"""
Unit Tests for ModelShield Adaptive Red-Team Orchestrator
"""
import os
import shutil
import unittest
import torch
import torch.nn as nn

from orchestrator.copilot import InvestigationCopilot
from orchestrator.engine import RedTeamOrchestrator
from orchestrator.policy import AdaptiveDecisionPolicy
from orchestrator.preflight import run_preflight_checks
from orchestrator.types import (
    DecisionAction,
    EvaluationPhase,
)


class SimpleTestModel(nn.Module):
    """Lightweight 2-layer ConvNet for fast, deterministic unit testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        h = h.view(h.size(0), -1)
        return self.fc(h)


class TestAdaptiveRedTeamOrchestrator(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = SimpleTestModel().to(self.device)
        self.model.eval()

        # Synthetic CIFAR-10 like images [8, 3, 32, 32] in [0, 1]
        torch.manual_seed(42)
        self.images = torch.rand(8, 3, 32, 32).clamp(0.0, 1.0)
        self.labels = torch.randint(0, 10, (8,))

        self.test_results_dir = "results/test_runs"
        os.makedirs(self.test_results_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir, ignore_errors=True)

    def test_preflight_valid(self):
        result = run_preflight_checks(
            model=self.model,
            images=self.images,
            labels=self.labels,
            epsilon=8 / 255,
            pgd_steps=10,
            pgd_alpha=2 / 255,
            device=self.device,
        )
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)
        self.assertTrue(result.checks["model_callable"]["passed"])
        self.assertTrue(result.checks["image_tensor_type"]["passed"])
        self.assertTrue(result.checks["pixel_range"]["passed"])
        self.assertTrue(result.checks["prediction_outputs"]["passed"])

    def test_preflight_invalid_pixel_range(self):
        bad_images = self.images.clone()
        bad_images[0, 0, 0, 0] = 1.5  # Outside [0, 1]
        result = run_preflight_checks(
            model=self.model,
            images=bad_images,
            labels=self.labels,
            epsilon=8 / 255,
        )
        self.assertFalse(result.passed)
        self.assertIn("pixel_range", result.checks)
        self.assertFalse(result.checks["pixel_range"]["passed"])

    def test_preflight_invalid_dimensions(self):
        bad_images = torch.rand(8, 32, 32)  # 3D instead of 4D
        result = run_preflight_checks(
            model=self.model,
            images=bad_images,
            labels=self.labels,
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["image_tensor_type"]["passed"])

    def test_policy_budget_exhaustion(self):
        policy = AdaptiveDecisionPolicy()
        decision = policy.decide_next_action(
            step_number=5,
            current_phase=EvaluationPhase.ATTACK_EVALUATION,
            preflight_passed=True,
            preflight_errors=[],
            completed_actions=["RUN_PREFLIGHT", "RUN_BASELINE", "RUN_FGSM"],
            metrics_by_action={"RUN_BASELINE": {"clean_accuracy": 80.0}},
            trust_flags_by_action={},
            remaining_budget=0,  # Zero budget left
            anomalies_detected=[],
        )
        self.assertEqual(decision.action, DecisionAction.STOP_BUDGET_EXHAUSTED)
        self.assertEqual(decision.budget_cost, 0)
        self.assertIn("exhausted", decision.reason.lower())

    def test_policy_escalation_flow(self):
        policy = AdaptiveDecisionPolicy()

        # Step 1: Preflight
        d1 = policy.decide_next_action(
            step_number=1,
            current_phase=EvaluationPhase.PREFLIGHT,
            preflight_passed=None,
            preflight_errors=[],
            completed_actions=[],
            metrics_by_action={},
            trust_flags_by_action={},
            remaining_budget=5,
            anomalies_detected=[],
        )
        self.assertEqual(d1.action, DecisionAction.RUN_PREFLIGHT)

        # Step 2: Baseline
        d2 = policy.decide_next_action(
            step_number=2,
            current_phase=EvaluationPhase.BASELINE,
            preflight_passed=True,
            preflight_errors=[],
            completed_actions=["RUN_PREFLIGHT"],
            metrics_by_action={},
            trust_flags_by_action={},
            remaining_budget=4,
            anomalies_detected=[],
        )
        self.assertEqual(d2.action, DecisionAction.RUN_BASELINE)

        # Step 3: FGSM Probe
        d3 = policy.decide_next_action(
            step_number=3,
            current_phase=EvaluationPhase.ATTACK_EVALUATION,
            preflight_passed=True,
            preflight_errors=[],
            completed_actions=["RUN_PREFLIGHT", "RUN_BASELINE"],
            metrics_by_action={"RUN_BASELINE": {"clean_accuracy": 90.0}},
            trust_flags_by_action={},
            remaining_budget=3,
            anomalies_detected=[],
        )
        self.assertEqual(d3.action, DecisionAction.RUN_FGSM)

        # Step 4: PGD Escalation
        d4 = policy.decide_next_action(
            step_number=4,
            current_phase=EvaluationPhase.ATTACK_EVALUATION,
            preflight_passed=True,
            preflight_errors=[],
            completed_actions=["RUN_PREFLIGHT", "RUN_BASELINE", "RUN_FGSM"],
            metrics_by_action={
                "RUN_BASELINE": {"clean_accuracy": 90.0},
                "RUN_FGSM": {"adversarial_accuracy": 60.0, "attack_success_rate": 33.3},
            },
            trust_flags_by_action={},
            remaining_budget=2,
            anomalies_detected=[],
        )
        self.assertEqual(d4.action, DecisionAction.ESCALATE_PGD)
        self.assertIn("FGSM revealed measurable vulnerability", d4.reason)

    def test_full_orchestrator_execution(self):
        orchestrator = RedTeamOrchestrator(
            model=self.model,
            device=self.device,
            images=self.images,
            labels=self.labels,
            max_budget=6,
            results_dir=self.test_results_dir,
        )

        assessment = orchestrator.run()

        # Verify assessment object
        self.assertIsNotNone(assessment)
        self.assertIn(assessment.overall_status, [
            "CRITICAL_VULNERABILITY",
            "MODERATE_VULNERABILITY",
            "HIGH_ROBUSTNESS_OBSERVED",
            "INCONCLUSIVE",
        ])
        self.assertGreaterEqual(len(assessment.key_findings), 1)
        self.assertGreaterEqual(len(assessment.scientific_limitations), 1)
        self.assertGreaterEqual(len(assessment.untested_vectors), 1)

        # Verify step and decision logs
        self.assertGreaterEqual(len(orchestrator.step_history), 3)
        self.assertGreaterEqual(len(orchestrator.decision_history), 3)

        # Verify persistence file created
        files = os.listdir(self.test_results_dir)
        self.assertTrue(any(f.endswith(".json") for f in files))

    def test_copilot_explanations(self):
        copilot = InvestigationCopilot()
        # Test empty state response
        empty_res = copilot.answer_inquiry("Why did testing stop?")
        self.assertIn("No active evaluation run data available", empty_res)

        # Feed real run data
        run_data = {
            "run_id": "test-run-123",
            "model_name": "SimpleTestModel",
            "config": {"max_budget": 5, "quantization_levels": 16},
            "completed_actions": ["RUN_PREFLIGHT", "RUN_BASELINE", "RUN_FGSM", "ESCALATE_PGD"],
            "decisions": [
                {
                    "action": "ESCALATE_PGD",
                    "reason": "FGSM revealed measurable vulnerability with ASR 40.0%.",
                    "supporting_evidence": {"clean_accuracy": 90.0, "fgsm_accuracy": 54.0, "fgsm_attack_success_rate": 40.0},
                    "remaining_budget": 2,
                },
                {
                    "action": "STOP_BUDGET_EXHAUSTED",
                    "reason": "Configured budget of steps was fully consumed.",
                    "remaining_budget": 0,
                }
            ],
            "step_history": [],
            "final_assessment": {
                "overall_status": "MODERATE_VULNERABILITY",
                "clean_accuracy": 90.0,
                "worst_adversarial_accuracy": 54.0,
                "max_attack_success_rate": 40.0,
                "key_findings": ["Clean accuracy was 90.00%."],
                "scientific_limitations": ["Empirical Linf robustness does not certify complete security."],
                "untested_vectors": ["AutoAttack", "Carlini & Wagner L2"],
            }
        }
        copilot.update_run_data(run_data)

        # Test "Why PGD" inquiry
        pgd_res = copilot.answer_inquiry("Why did the orchestrator choose PGD?")
        self.assertIn("FGSM revealed measurable vulnerability", pgd_res)
        self.assertIn("90.0", pgd_res)

        # Test ASR vs Accuracy explanation
        diff_res = copilot.answer_inquiry("What is the difference between attack success rate and adversarial accuracy?")
        self.assertIn("initially correct", diff_res.lower())
        self.assertIn("40.00%", diff_res)

        # Test limitations inquiry
        limit_res = copilot.answer_inquiry("What limitations should I present?")
        self.assertIn("Empirical Linf robustness", limit_res)

    def test_preflight_failure_halts_orchestrator(self):
        # Corrupt images with NaN values
        bad_images = self.images.clone()
        bad_images[0, 0, 0, 0] = float("nan")

        orchestrator = RedTeamOrchestrator(
            model=self.model,
            device=self.device,
            images=bad_images,
            labels=self.labels,
            max_budget=5,
            results_dir=self.test_results_dir,
        )
        assessment = orchestrator.run()

        # The run should abort and flag INTEGRITY_FAILED, NOT falsely claim model robustness or vulnerability
        self.assertEqual(assessment.overall_status, "INTEGRITY_FAILED")
        self.assertEqual(orchestrator.current_phase, EvaluationPhase.FAILED)
        self.assertIn("RUN_PREFLIGHT", orchestrator.completed_actions)
        # Attacks must NOT have executed on corrupted inputs
        self.assertNotIn("RUN_FGSM", orchestrator.completed_actions)
        self.assertNotIn("ESCALATE_PGD", orchestrator.completed_actions)

    def test_defense_quantization_evaluation(self):
        orchestrator = RedTeamOrchestrator(
            model=self.model,
            device=self.device,
            images=self.images,
            labels=self.labels,
            max_budget=6,
            results_dir=self.test_results_dir,
        )
        assessment = orchestrator.run()
        self.assertIn("TEST_DEFENSE", orchestrator.completed_actions)

        defense_step = next(s for s in orchestrator.step_history if s.action_type == "TEST_DEFENSE")
        self.assertEqual(defense_step.phase, EvaluationPhase.DEFENSE_VERIFICATION)
        self.assertIn("defense_clean_accuracy", defense_step.metrics)
        self.assertIn("defended_fgsm_accuracy", defense_step.metrics)
        self.assertIn("defended_pgd_accuracy", defense_step.metrics)

    def test_load_model_refuses_untrained_random_fallback(self):
        from unittest.mock import patch
        from model.load_model import load_model, ModelConfigurationError
        # When RobustBench is missing and no local verified checkpoint exists,
        # load_model must raise ModelConfigurationError rather than silently returning an untrained model.
        with patch("model.load_model._ROBUSTBENCH_AVAILABLE", False):
            with self.assertRaises(ModelConfigurationError) as ctx:
                load_model()
            err_msg = str(ctx.exception)
            self.assertIn("Carmon2019Unlabeled", err_msg)
            self.assertIn("Security Policy Enforcement", err_msg)
            self.assertIn("pip install", err_msg)

    def test_load_model_local_checkpoint_provenance(self):
        from model.load_model import load_model
        # Create a temporary verified checkpoint
        chkpt_path = os.path.join(self.test_results_dir, "test_cifar_chkpt.pt")
        test_net = SimpleTestModel()
        torch.save(test_net, chkpt_path)

        loaded_model, dev = load_model(checkpoint_path=chkpt_path)
        self.assertIsNotNone(loaded_model)
        self.assertTrue(hasattr(loaded_model, "model_metadata"))
        self.assertTrue(loaded_model.model_metadata.get("is_pretrained"))
        self.assertIn("Local Verified Checkpoint", loaded_model.model_metadata.get("checkpoint_source", ""))

    def test_preflight_model_provenance(self):
        # Attach provenance metadata and test preflight records it
        self.model.model_metadata = {
            "architecture": "Carmon2019Unlabeled",
            "checkpoint_source": "RobustBench",
            "is_pretrained": True,
        }
        result = run_preflight_checks(
            model=self.model,
            images=self.images,
            labels=self.labels,
            device=self.device,
        )
        self.assertTrue(result.passed)
        self.assertIn("model_provenance", result.checks)
        self.assertTrue(result.checks["model_provenance"]["passed"])
        self.assertEqual(result.checks["model_provenance"]["metadata"]["architecture"], "Carmon2019Unlabeled")

    def test_neural_attack_visualizer_detailed_samples(self):
        orchestrator = RedTeamOrchestrator(
            model=self.model,
            device=self.device,
            images=self.images,
            labels=self.labels,
            max_budget=5,
            results_dir=self.test_results_dir,
        )
        orchestrator.run()

        # Check detailed samples extracted
        self.assertGreater(len(orchestrator.detailed_samples), 0)
        sample0 = orchestrator.detailed_samples[0]
        self.assertIn("clean_img_b64", sample0)
        self.assertTrue(sample0["clean_img_b64"].startswith("data:image/png;base64,"))
        self.assertIn("clean_feature_b64", sample0)
        self.assertTrue(sample0["clean_feature_b64"].startswith("data:image/png;base64,"))
        self.assertIn("fgsm_img_b64", sample0)
        self.assertIn("pgd_heatmap_b64", sample0)
        self.assertIn("pgd_feature_b64", sample0)
        self.assertIn("feature_distortion_b64", sample0)
        self.assertIn("trajectory", sample0)
        self.assertGreater(len(sample0["trajectory"]), 1)

        # Check class metrics and flip matrix
        self.assertIsInstance(orchestrator.class_metrics, dict)
        self.assertIsInstance(orchestrator.flip_matrix, dict)

        # Check persistence file contains detailed_samples
        persisted_file = os.path.join(self.test_results_dir, f"{orchestrator.run_id}.json")
        self.assertTrue(os.path.exists(persisted_file))
        import json
        with open(persisted_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        self.assertIn("detailed_samples", saved_data)
        self.assertIn("class_metrics", saved_data)
        self.assertIn("flip_matrix", saved_data)

    def test_copilot_sample_and_class_inquiries(self):
        mock_run = {
            "run_id": "test-run",
            "model_name": "Carmon2019Unlabeled",
            "detailed_samples": [{
                "sample_id": 1,
                "true_label": 6,
                "true_class": "frog",
                "clean_pred": 6,
                "clean_class": "frog",
                "clean_confidence": 97.5,
                "fgsm_pred": 3,
                "fgsm_class": "cat",
                "fgsm_confidence": 65.0,
                "fgsm_l_inf": 0.03137,
                "pgd_pred": 3,
                "pgd_class": "cat",
                "pgd_confidence": 88.2,
                "pgd_l_inf": 0.03137,
                "fooled": True,
            }],
            "class_metrics": {
                "frog": {"clean_acc": 100.0, "fgsm_acc": 0.0, "pgd_acc": 0.0},
                "automobile": {"clean_acc": 100.0, "fgsm_acc": 100.0, "pgd_acc": 100.0},
            },
            "final_assessment": {"clean_accuracy": 100.0, "worst_adversarial_accuracy": 50.0},
        }
        copilot = InvestigationCopilot(mock_run)

        # Test sample query
        sample_resp = copilot.answer_inquiry("What happened to sample 1?")
        self.assertIn("Technical Dossier for Sample 1", sample_resp)
        self.assertIn("frog", sample_resp)
        self.assertIn("cat", sample_resp)
        self.assertIn("FOOLED BY ADVERSARIAL ASCENT", sample_resp)

        # Test class query
        class_resp = copilot.answer_inquiry("Which class was most vulnerable?")
        self.assertIn("Class-wise Robustness & Vulnerability Analysis", class_resp)
        self.assertIn("frog", class_resp)


if __name__ == "__main__":
    unittest.main()
