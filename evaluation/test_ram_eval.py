"""
Unit Tests for RAM++ Live Evaluation and Regression Gate Pipeline.
"""

from __future__ import annotations

import io
import sys
import unittest
import zipfile
from unittest.mock import MagicMock, patch

# Ensure mock ram module exists if running in non-venv python
if "ram" not in sys.modules:
    mock_ram = MagicMock()
    mock_ram.inference = MagicMock()
    mock_ram.models = MagicMock()
    sys.modules["ram"] = mock_ram
    sys.modules["ram.inference"] = mock_ram.inference
    sys.modules["ram.models"] = mock_ram.models

import torch
import torch.nn as nn
from PIL import Image

from evaluation.ram_eval import (
    ensure_benchmark_dataset,
    evaluate_ram_model_live,
    load_dataset_images,
    perturb_image_tensor,
    run_live_ram_regression_pipeline,
)
from evaluation.regression_gate import (
    GateVerdict,
    RegressionThresholds,
    validate_snapshot_integrity,
)
from model.ram_loader import validate_ram_state_dict


class DummyRAMModel(nn.Module):
    """Mock RAM++ model for deterministic CPU testing."""
    def __init__(self, mode: str = "normal"):
        super().__init__()
        self.mode = mode
        # dummy param so parameters() is non-empty
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


class TestRAMLiveEvaluation(unittest.TestCase):
    """Tests the benchmark dataset loader, perturbation engine, and evaluation pipeline."""

    def test_ensure_benchmark_dataset(self):
        images = ensure_benchmark_dataset("data/test_ram_benchmark")
        self.assertGreaterEqual(len(images), 8)
        name, img = images[0]
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (384, 384))
        self.assertEqual(img.mode, "RGB")

    def test_perturb_image_tensor_budget(self):
        tensor = torch.full((1, 3, 384, 384), 0.5)
        eps = 8.0 / 255.0
        adv = perturb_image_tensor(tensor, epsilon=eps)

        self.assertEqual(adv.shape, tensor.shape)
        self.assertTrue(torch.all(adv >= 0.0))
        self.assertTrue(torch.all(adv <= 1.0))
        max_diff = torch.max(torch.abs(adv - tensor)).item()
        self.assertLessEqual(max_diff, eps + 1e-5)

    def test_load_dataset_images_zip(self):
        # Create an in-memory zip with 2 PNG images
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            img = Image.new("RGB", (100, 100), color="red")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            z.writestr("test1.png", img_byte_arr.getvalue())
            z.writestr("test2.jpg", img_byte_arr.getvalue())

        zip_buf.seek(0)
        images, label = load_dataset_images("zip", uploaded_zip=zip_buf, max_samples=2)
        self.assertEqual(len(images), 2)
        self.assertIn("Uploaded ZIP", label)

    def test_validate_ram_state_dict(self):
        # Valid RAM++ state dict mock
        valid_sd = {
            f"visual_encoder.layers.{i}.weight": torch.zeros(10) for i in range(25)
        }
        valid_sd["tag_list.weight"] = torch.zeros(10)
        ok, reason = validate_ram_state_dict(valid_sd)
        self.assertTrue(ok)

        # Incompatible state dict (e.g., standard ResNet)
        resnet_sd = {"conv1.weight": torch.zeros(10), "fc.weight": torch.zeros(10)}
        bad_ok, _ = validate_ram_state_dict(resnet_sd)
        self.assertFalse(bad_ok)

    @patch("ram.inference.inference_ram")
    def test_evaluate_ram_model_live_and_snapshot(self, mock_inference):
        # Mock inference to return predictable tags
        mock_inference.side_effect = [
            ("dog | grass | park", ""),  # clean image 1
            ("dog | grass", ""),         # adv image 1 (lost park)
            ("car | road", ""),          # clean image 2
            ("car | road | tree", ""),   # adv image 2 (new tree)
        ]

        model = DummyRAMModel()
        preprocess = lambda img: torch.zeros(3, 384, 384)
        dataset = [
            ("img1.png", Image.new("RGB", (384, 384))),
            ("img2.png", Image.new("RGB", (384, 384))),
        ]

        snapshot, raw = evaluate_ram_model_live(
            model=model,
            preprocess=preprocess,
            model_name="TestRAMModel",
            dataset_images=dataset,
            device=torch.device("cpu"),
        )

        # Verify snapshot integrity
        ok, errs, val_snap = validate_snapshot_integrity(snapshot, "Candidate")
        self.assertTrue(ok, f"Snapshot integrity failed: {errs}")

        m = snapshot.metrics
        self.assertIn("tag_agreement_rate", m)
        self.assertIn("mean_jaccard_similarity", m)
        self.assertIn("mean_tag_retention_rate", m)
        self.assertIn("avg_tags_detected", m)
        self.assertIn("tags_lost_per_image", m)
        self.assertIn("new_tags_per_image", m)
        self.assertIn("semantic_consistency", m)
        self.assertIn("mean_latency_ms", m)

        self.assertEqual(snapshot.trust_flag["status"], "PASS")

    @patch("ram.inference.inference_ram")
    def test_run_live_ram_regression_pipeline(self, mock_inference):
        # Baseline: clean & adv consistent
        # Candidate: identical
        mock_inference.return_value = ("airplane | sky | flight", "")

        base_model = DummyRAMModel()
        cand_model = DummyRAMModel()
        preprocess = lambda img: torch.zeros(3, 384, 384)
        dataset = [("plane.png", Image.new("RGB", (384, 384)))]

        with patch("evaluation.ram_eval.load_dataset_images", return_value=(dataset, "Test")):
            b_snap, c_snap, gate_res = run_live_ram_regression_pipeline(
                baseline_model=base_model,
                candidate_model=cand_model,
                preprocess=preprocess,
                baseline_name="BaselineRAM",
                candidate_name="CandidateRAM",
                max_samples=1,
                device=torch.device("cpu"),
            )

        self.assertEqual(gate_res.verdict, GateVerdict.PASS)
        self.assertIn("mean_jaccard_similarity", gate_res.metric_deltas)
        self.assertIn("tag_agreement_rate", gate_res.metric_deltas)

    def test_load_dataset_images_directory_and_aliases(self):
        import tempfile
        import shutil
        from pathlib import Path

        tmp = tempfile.mkdtemp()
        try:
            # Create files with supported extensions
            for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                img = Image.new("RGB", (384, 384), color="blue")
                img.save(Path(tmp) / f"sample{ext}")
            # Unsupported file
            with open(Path(tmp) / "ignored.txt", "w") as f:
                f.write("text file")

            # Test using directory= keyword
            imgs1, label1 = load_dataset_images(directory=tmp, max_samples=3)
            self.assertEqual(len(imgs1), 3)
            self.assertIn("Directory", label1)

            # Test using dir_path= keyword and source="directory"
            imgs2, label2 = load_dataset_images(source="directory", dir_path=tmp, max_samples=10)
            self.assertEqual(len(imgs2), 5)
            self.assertIn("Directory", label2)

            # Test passing preloaded list of tuples
            imgs3, label3 = load_dataset_images(directory=imgs2, max_samples=2)
            self.assertEqual(len(imgs3), 2)
            self.assertIn("Preloaded", label3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cross_family_rejection_cifar_vs_ram(self):
        from evaluation.regression_gate import (
            EvaluationMetadata,
            EvaluationSnapshot,
            evaluate_regression_gate,
        )

        cifar_snap = EvaluationSnapshot(
            metadata=EvaluationMetadata(
                model_name="Carmon2019",
                dataset_name="CIFAR-10",
                num_samples=16,
                epsilon=8.0 / 255.0,
                task_type="classification",
            ),
            metrics={
                "clean_accuracy": 89.5,
                "pgd_accuracy": 55.2,
                "pgd_attack_success_rate": 38.3,
            },
        )

        ram_snap = EvaluationSnapshot(
            metadata=EvaluationMetadata(
                model_name="RAMPlus_Official",
                dataset_name="RAM_Benchmark",
                num_samples=8,
                epsilon=8.0 / 255.0,
                task_type="ram_tagging",
            ),
            metrics={
                "tag_agreement_rate": 90.0,
                "mean_jaccard_similarity": 85.0,
                "mean_tag_retention_rate": 92.0,
                "semantic_consistency": 85.0,
            },
        )

        res = evaluate_regression_gate(cifar_snap, ram_snap)
        self.assertEqual(res.verdict, GateVerdict.FAIL)
        self.assertTrue(any("Task type mismatch" in f for f in res.failures))

    def test_compatible_ram_tagging_snapshots(self):
        from evaluation.regression_gate import (
            EvaluationMetadata,
            EvaluationSnapshot,
            evaluate_regression_gate,
        )

        base = EvaluationSnapshot(
            metadata=EvaluationMetadata(
                model_name="RAMPlus_Official",
                dataset_name="RAM_Benchmark",
                num_samples=8,
                epsilon=8.0 / 255.0,
                task_type="ram_tagging",
            ),
            metrics={
                "tag_agreement_rate": 90.0,
                "mean_jaccard_similarity": 85.0,
                "mean_tag_retention_rate": 92.0,
                "avg_tags_detected": 4.5,
                "tags_lost_per_image": 0.5,
                "new_tags_per_image": 0.5,
                "semantic_consistency": 85.0,
                "mean_latency_ms": 120.0,
            },
        )

        cand = EvaluationSnapshot(
            metadata=EvaluationMetadata(
                model_name="RAMPlus_FineTuned",
                dataset_name="RAM_Benchmark",
                num_samples=8,
                epsilon=8.0 / 255.0,
                task_type="image_tagging",  # alias for tagging
            ),
            metrics={
                "tag_agreement_rate": 90.0,
                "mean_jaccard_similarity": 85.0,
                "mean_tag_retention_rate": 92.0,
                "avg_tags_detected": 4.5,
                "tags_lost_per_image": 0.5,
                "new_tags_per_image": 0.5,
                "semantic_consistency": 85.0,
                "mean_latency_ms": 115.0,
            },
        )

        res = evaluate_regression_gate(base, cand)
        self.assertEqual(res.verdict, GateVerdict.PASS)
        self.assertFalse(any("Task type mismatch" in f for f in res.failures))

    @patch("ram.inference.inference_ram")
    def test_identical_tag_sets_produce_perfect_metrics(self, mock_inference):
        """Asserts that identical clean and adversarial tags produce 100% agreement, 100% Jaccard, 0 lost, 0 new."""
        mock_inference.return_value = ("cat | pet | animal | blanket", "")

        model = DummyRAMModel()
        preprocess = lambda img: torch.zeros(3, 384, 384)
        dataset = [
            ("img1.png", Image.new("RGB", (384, 384))),
            ("img2.png", Image.new("RGB", (384, 384))),
        ]

        snapshot, raw = evaluate_ram_model_live(
            model=model,
            preprocess=preprocess,
            model_name="IdenticalTestModel",
            dataset_images=dataset,
            device=torch.device("cpu"),
            num_latency_repetitions=1,
        )

        m = snapshot.metrics
        self.assertEqual(m["tag_agreement_rate"], 100.0)
        self.assertEqual(m["mean_jaccard_similarity"], 100.0)
        self.assertEqual(m["mean_tag_retention_rate"], 100.0)
        self.assertEqual(m["tags_lost_per_image"], 0.0)
        self.assertEqual(m["new_tags_per_image"], 0.0)
        self.assertEqual(m["semantic_consistency"], 100.0)
        self.assertEqual(m["avg_tags_detected"], 4.0)
        self.assertGreater(m["mean_latency_ms"], 0.0)

    @patch("ram.inference.inference_ram")
    def test_completely_disjoint_tag_sets(self, mock_inference):
        """Asserts that disjoint clean and adv tags produce 0% Jaccard, 0% agreement, expected lost & new."""
        # Clean: "cat | pet" (2 tags), Adv: "dog | bone | park" (3 tags)
        mock_inference.side_effect = [
            ("cat | pet", ""),
            ("dog | bone | park", ""),
        ]

        model = DummyRAMModel()
        preprocess = lambda img: torch.zeros(3, 384, 384)
        dataset = [("img1.png", Image.new("RGB", (384, 384)))]

        snapshot, raw = evaluate_ram_model_live(
            model=model,
            preprocess=preprocess,
            model_name="DisjointTestModel",
            dataset_images=dataset,
            device=torch.device("cpu"),
            num_latency_repetitions=1,
        )

        m = snapshot.metrics
        self.assertEqual(m["tag_agreement_rate"], 0.0)
        self.assertEqual(m["mean_jaccard_similarity"], 0.0)
        self.assertEqual(m["mean_tag_retention_rate"], 0.0)
        self.assertEqual(m["tags_lost_per_image"], 2.0)
        self.assertEqual(m["new_tags_per_image"], 3.0)
        self.assertEqual(m["semantic_consistency"], 0.0)

    @patch("ram.inference.inference_ram")
    def test_partially_overlapping_tag_sets(self, mock_inference):
        """Asserts partial overlap yields exact expected Jaccard, retention, and delta counts."""
        # Clean: dog, grass, ball (3 tags)
        # Adv: dog, grass, tree (3 tags)
        # Overlap: dog, grass (2)
        # Lost: ball (1)
        # New: tree (1)
        # Union: 4
        # Jaccard = 2/4 = 50.0%
        # Retention = 2/3 * 100 = 66.67%
        mock_inference.side_effect = [
            ("dog | grass | ball", ""),
            ("dog | grass | tree", ""),
        ]

        model = DummyRAMModel()
        preprocess = lambda img: torch.zeros(3, 384, 384)
        dataset = [("img1.png", Image.new("RGB", (384, 384)))]

        snapshot, raw = evaluate_ram_model_live(
            model=model,
            preprocess=preprocess,
            model_name="PartialTestModel",
            dataset_images=dataset,
            device=torch.device("cpu"),
            num_latency_repetitions=1,
        )

        m = snapshot.metrics
        self.assertEqual(m["tag_agreement_rate"], 100.0)  # 50% >= 50% threshold
        self.assertEqual(m["mean_jaccard_similarity"], 50.0)
        self.assertAlmostEqual(m["mean_tag_retention_rate"], 66.67, places=1)
        self.assertEqual(m["tags_lost_per_image"], 1.0)
        self.assertEqual(m["new_tags_per_image"], 1.0)
        self.assertEqual(m["semantic_consistency"], 50.0)

    @patch("ram.inference.inference_ram")
    def test_self_consistency_regression_gate_zero_regressions_clean_pass(self, mock_inference):
        """Asserts that running regression gate on identical models yields PASS and 0.0 deltas."""
        mock_inference.return_value = ("boat | sea | water | sky", "")

        base_model = DummyRAMModel()
        cand_model = DummyRAMModel()
        preprocess = lambda img: torch.zeros(3, 384, 384)
        dataset = [("boat.png", Image.new("RGB", (384, 384)))]

        with patch("evaluation.ram_eval.load_dataset_images", return_value=(dataset, "Test")):
            b_snap, c_snap, gate_res = run_live_ram_regression_pipeline(
                baseline_model=base_model,
                candidate_model=cand_model,
                preprocess=preprocess,
                baseline_name="OfficialRAM",
                candidate_name="SelfConsistencyCandidate",
                max_samples=1,
                device=torch.device("cpu"),
            )

        self.assertEqual(gate_res.verdict, GateVerdict.PASS)
        self.assertEqual(len(gate_res.failures), 0)
        self.assertEqual(len(gate_res.warnings), 0)

        # Verify every metric delta is exactly 0.0
        for metric_name in [
            "tag_agreement_rate",
            "mean_jaccard_similarity",
            "mean_tag_retention_rate",
            "avg_tags_detected",
            "tags_lost_per_image",
            "new_tags_per_image",
            "semantic_consistency",
        ]:
            delta_obj = gate_res.metric_deltas.get(metric_name)
            self.assertIsNotNone(delta_obj, f"Missing metric delta for {metric_name}")
            self.assertEqual(delta_obj.delta, 0.0, f"Delta for {metric_name} should be 0.0, got {delta_obj.delta}")

    @patch("ram.inference.inference_ram")
    def test_regression_gate_catches_degraded_candidate_model(self, mock_inference):
        """Asserts that candidate with degraded tag retention triggers WARNING or FAIL."""
        # Baseline: clean & adv 100% consistent
        # Candidate: clean has 4 tags, adv has 0 tags (catastrophic drop)
        mock_inference.side_effect = [
            # Baseline (clean, adv)
            ("car | wheel | road | drive", ""),
            ("car | wheel | road | drive", ""),
            # Candidate (clean, adv)
            ("car | wheel | road | drive", ""),
            ("", ""),
        ]

        base_model = DummyRAMModel()
        cand_model = DummyRAMModel()
        preprocess = lambda img: torch.zeros(3, 384, 384)
        dataset = [("car.png", Image.new("RGB", (384, 384)))]

        with patch("evaluation.ram_eval.load_dataset_images", return_value=(dataset, "Test")):
            b_snap, c_snap, gate_res = run_live_ram_regression_pipeline(
                baseline_model=base_model,
                candidate_model=cand_model,
                preprocess=preprocess,
                baseline_name="Baseline",
                candidate_name="DegradedCandidate",
                max_samples=1,
                device=torch.device("cpu"),
            )

        self.assertEqual(gate_res.verdict, GateVerdict.FAIL)
        self.assertGreater(len(gate_res.failures), 0)


if __name__ == "__main__":
    unittest.main()
