"""
Unit and integration test suite for ModelShield Live Model Evaluation and Upload.
Tests checkpoint loading, architecture inference, forward validation, live snapshot generation,
and regression gate evaluation.
"""

import io
import json
import unittest
import torch
import torch.nn as nn

from model.checkpoint_loader import (
    SupportedArchitecture,
    CifarResNet18,
    create_model_architecture,
    clean_state_dict,
    infer_architecture_from_state_dict,
    load_checkpoint_safely,
    _validate_model_forward,
)
from evaluation.live_eval import (
    evaluate_model_live,
    is_snapshot_compatible_with_config,
    run_live_regression_pipeline,
)
from evaluation.regression_gate import (
    GateVerdict,
    MetricStatus,
    EvaluationMetadata,
    EvaluationSnapshot,
    RegressionThresholds,
    evaluate_regression_gate,
    export_gate_report_json,
    export_gate_report_markdown,
    validate_snapshot_integrity,
)


class DummyCifarModel(nn.Module):
    """Small deterministic model for rapid unit testing without downloading models."""
    def __init__(self, accuracy_bias: float = 1.0):
        super().__init__()
        torch.manual_seed(42)
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
        self.accuracy_bias = accuracy_bias

        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.conv(x))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        logits = self.fc(out) * self.accuracy_bias
        return logits


class TestLiveModelEvaluation(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(42)
        # 8 synthetic images and labels for offline evaluation testing
        self.dummy_images = torch.rand(8, 3, 32, 32).clamp(0.0, 1.0)
        self.dummy_labels = torch.randint(0, 10, (8,))

    def test_cifar_resnet18_instantiation_and_forward(self):
        """Verifies CifarResNet18 instantiates properly and produces shape (B, 10)."""
        model = CifarResNet18(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))
        self.assertFalse(torch.isnan(out).any())

    def test_clean_state_dict_prefix_stripping(self):
        """Verifies clean_state_dict removes 'module.' and 'model.' prefixes."""
        raw = {
            "module.conv1.weight": torch.randn(16, 3, 3, 3),
            "model.fc.bias": torch.randn(10),
            "epoch": 25,
            "best_acc": 85.5,
        }
        cleaned = clean_state_dict(raw)
        self.assertIn("conv1.weight", cleaned)
        self.assertIn("fc.bias", cleaned)
        self.assertNotIn("epoch", cleaned)
        self.assertNotIn("best_acc", cleaned)

    def test_infer_architecture_from_state_dict(self):
        """Verifies architecture heuristic detection."""
        # Carmon with sub_block1
        carmon_sd = {"sub_block1.layer.0.conv1.weight": torch.randn(16, 16, 3, 3)}
        self.assertEqual(
            infer_architecture_from_state_dict(carmon_sd),
            SupportedArchitecture.WIDERESNET_28_10_CARMON,
        )

        # Standard WideResNet-28
        wrn28_sd = {
            "block1.layer.0.conv1.weight": torch.randn(16, 16, 3, 3),
            "block3.layer.3.conv2.weight": torch.randn(64, 64, 3, 3),
        }
        self.assertEqual(
            infer_architecture_from_state_dict(wrn28_sd),
            SupportedArchitecture.WIDERESNET_28_10,
        )

        # ResNet18 CIFAR
        r18_sd = {
            "conv1.weight": torch.randn(64, 3, 3, 3),
            "layer1.0.conv1.weight": torch.randn(64, 64, 3, 3),
        }
        self.assertEqual(
            infer_architecture_from_state_dict(r18_sd),
            SupportedArchitecture.RESNET18_CIFAR,
        )

    def test_load_checkpoint_safely_bytes_and_buffer(self):
        """Verifies checkpoint loading from in-memory BytesIO buffer."""
        model = DummyCifarModel()
        buf = io.BytesIO()
        torch.save({"state_dict": model.state_dict()}, buf)
        buf.seek(0)

        # Loading into custom architecture or CifarResNet18 with incompatible weights should raise
        with self.assertRaises(ValueError):
            load_checkpoint_safely(buf, architecture=SupportedArchitecture.RESNET18_CIFAR)

    def test_load_checkpoint_safely_full_module(self):
        """Verifies loading a checkpoint that contains an instantiated nn.Module."""
        model = DummyCifarModel()
        buf = io.BytesIO()
        torch.save(model, buf)
        buf.seek(0)

        loaded_model, arch, info = load_checkpoint_safely(buf, device=self.device)
        self.assertIsInstance(loaded_model, nn.Module)
        self.assertEqual(info["type"], "nn.Module")

    def test_load_checkpoint_invalid_file_extension(self):
        """Verifies non-PyTorch extensions are rejected."""
        dummy_file = io.BytesIO(b"not a pt file")
        dummy_file.name = "malicious_script.sh"
        with self.assertRaises(ValueError) as cm:
            load_checkpoint_safely(dummy_file)
        self.assertIn("Expected .pt or .pth", str(cm.exception))

    def test_validate_model_forward_rejection(self):
        """Verifies that non-finite logits or shape mismatches fail forward validation."""
        class BadModel(nn.Module):
            def forward(self, x):
                return torch.tensor([[float("nan")] * 10])

        with self.assertRaises(ValueError) as cm:
            _validate_model_forward(BadModel(), self.device, 10)
        self.assertIn("NaN or infinite values", str(cm.exception))

    def test_evaluate_model_live_and_snapshot_generation(self):
        """Verifies evaluate_model_live executes genuine attack ensemble and builds valid snapshot."""
        model = DummyCifarModel()
        snapshot, raw_res = evaluate_model_live(
            model=model,
            model_name="TestModel",
            architecture="DummyCifarModel",
            dataset_name="CIFAR-10",
            n_samples=8,
            epsilon=8.0 / 255.0,
            pgd_steps=2,
            device=self.device,
            preloaded_data=(self.dummy_images, self.dummy_labels),
        )

        # Integrity validation
        self.assertIsInstance(snapshot, EvaluationSnapshot)
        is_ok, errs, val_snap = validate_snapshot_integrity(snapshot, "TestModel")
        self.assertTrue(is_ok, f"Snapshot integrity failed: {errs}")

        # Metrics presence and ranges
        m = snapshot.metrics
        for key in (
            "clean_accuracy",
            "fgsm_accuracy",
            "pgd_accuracy",
            "fgsm_attack_success_rate",
            "pgd_attack_success_rate",
            "worst_case_attack_success_rate",
        ):
            self.assertIn(key, m)
            self.assertGreaterEqual(m[key], 0.0)
            self.assertLessEqual(m[key], 100.0)

        # Metadata validation
        self.assertEqual(snapshot.metadata.num_samples, 8)
        self.assertEqual(snapshot.metadata.dataset_name, "CIFAR-10")
        self.assertAlmostEqual(snapshot.metadata.epsilon, 8.0 / 255.0, places=6)

        # Trust flag presence
        self.assertIsNotNone(snapshot.trust_flag)
        self.assertIn("status", snapshot.trust_flag)
        self.assertIn("passed", snapshot.trust_flag)

    def test_is_snapshot_compatible_with_config(self):
        """Verifies snapshot reuse compatibility logic."""
        model = DummyCifarModel()
        snapshot, _ = evaluate_model_live(
            model=model,
            model_name="BaseModel",
            n_samples=8,
            epsilon=8.0 / 255.0,
            pgd_steps=5,
            device=self.device,
            preloaded_data=(self.dummy_images, self.dummy_labels),
        )

        # Matching configuration
        self.assertTrue(
            is_snapshot_compatible_with_config(
                snapshot, "CIFAR-10", n_samples=8, epsilon=8.0 / 255.0, pgd_steps=5
            )
        )

        # Mismatched sample count
        self.assertFalse(
            is_snapshot_compatible_with_config(
                snapshot, "CIFAR-10", n_samples=16, epsilon=8.0 / 255.0, pgd_steps=5
            )
        )

        # Mismatched epsilon
        self.assertFalse(
            is_snapshot_compatible_with_config(
                snapshot, "CIFAR-10", n_samples=8, epsilon=4.0 / 255.0, pgd_steps=5
            )
        )

        # Mismatched PGD steps
        self.assertFalse(
            is_snapshot_compatible_with_config(
                snapshot, "CIFAR-10", n_samples=8, epsilon=8.0 / 255.0, pgd_steps=10
            )
        )

        # Mismatched dataset
        self.assertFalse(
            is_snapshot_compatible_with_config(
                snapshot, "MNIST", n_samples=8, epsilon=8.0 / 255.0, pgd_steps=5
            )
        )

    def test_live_regression_pipeline_pass_and_fail(self):
        """Verifies run_live_regression_pipeline produces PASS on identical models and FAIL on degraded models."""
        base_model = DummyCifarModel(accuracy_bias=1.0)
        cand_model_identical = DummyCifarModel(accuracy_bias=1.0)

        # 1. Identical model -> should PASS
        base_snap, cand_snap, gate_res = run_live_regression_pipeline(
            baseline_model=base_model,
            candidate_model=cand_model_identical,
            baseline_name="BaseModel",
            candidate_name="CandidateIdentical",
            candidate_architecture="DummyCifarModel",
            n_samples=8,
            epsilon=8.0 / 255.0,
            pgd_steps=2,
            device=self.device,
            thresholds=RegressionThresholds(),
        )
        self.assertEqual(gate_res.verdict, GateVerdict.PASS)
        self.assertEqual(len(gate_res.failures), 0)

        # 2. Degraded candidate model with zeroed weights -> should FAIL
        cand_model_degraded = DummyCifarModel()
        with torch.no_grad():
            cand_model_degraded.fc.weight.fill_(0.0)
            cand_model_degraded.fc.bias.fill_(0.0)
            cand_model_degraded.fc.bias[2] = 100.0  # Class 2 does not appear in test labels

        # Reuse existing base snapshot
        _, cand_snap_deg, gate_res_deg = run_live_regression_pipeline(
            baseline_model=None,
            candidate_model=cand_model_degraded,
            baseline_name="BaseModel",
            candidate_name="CandidateDegraded",
            candidate_architecture="DummyCifarModel",
            existing_baseline_snapshot=base_snap,
            n_samples=8,
            epsilon=8.0 / 255.0,
            pgd_steps=2,
            device=self.device,
            thresholds=RegressionThresholds(max_clean_accuracy_drop=1.0),
        )
        self.assertEqual(gate_res_deg.verdict, GateVerdict.FAIL)
        self.assertGreater(len(gate_res_deg.failures), 0)

    def test_json_and_markdown_reports_and_snapshot_serialization(self):
        """Verifies that generated snapshots and regression reports can be serialized and deserialized."""
        base_model = DummyCifarModel()
        cand_model = DummyCifarModel()

        base_snap, cand_snap, gate_res = run_live_regression_pipeline(
            baseline_model=base_model,
            candidate_model=cand_model,
            baseline_name="BaseModel",
            candidate_name="CandModel",
            candidate_architecture="DummyCifarModel",
            n_samples=8,
            epsilon=8.0 / 255.0,
            pgd_steps=2,
            device=self.device,
        )

        # Snapshots to JSON
        base_json = json.dumps(base_snap.to_dict(), indent=2)
        cand_json = json.dumps(cand_snap.to_dict(), indent=2)

        # Reload snapshots from JSON
        reloaded_base = EvaluationSnapshot.from_dict(json.loads(base_json))
        reloaded_cand = EvaluationSnapshot.from_dict(json.loads(cand_json))

        ok_b, _, val_b = validate_snapshot_integrity(reloaded_base, "Baseline")
        ok_c, _, val_c = validate_snapshot_integrity(reloaded_cand, "Candidate")
        self.assertTrue(ok_b)
        self.assertTrue(ok_c)

        # Gate reports
        report_json = export_gate_report_json(gate_res)
        report_md = export_gate_report_markdown(gate_res)

        self.assertIn("ModelShield Security Regression Gate", report_md)
        self.assertIn("summary", json.loads(report_json))


if __name__ == "__main__":
    unittest.main()
