"""
Unit test suite for ModelShield Feature 15: Model Security Regression Gate.

Runs offline with zero network requests and zero model/dataset downloads.
Runnable with both `python -m unittest evaluation/test_regression_gate.py` and `pytest`.
"""

import math
import json
import unittest

from evaluation.regression_gate import (
    GateVerdict,
    MetricStatus,
    EvaluationMetadata,
    EvaluationSnapshot,
    RegressionThresholds,
    evaluate_regression_gate,
    export_gate_report_json,
    export_gate_report_markdown,
)


def get_base_fixture() -> EvaluationSnapshot:
    """Creates a standard healthy baseline snapshot fixture."""
    return EvaluationSnapshot(
        metadata=EvaluationMetadata(
            model_name="Carmon2019Unlabeled_Baseline",
            dataset_name="CIFAR-10",
            num_samples=32,
            epsilon=8.0 / 255.0,
            alpha=2.0 / 255.0,
            pgd_steps=10,
            random_start=True,
            timestamp="2026-10-29T12:00:00Z",
        ),
        metrics={
            "clean_accuracy": 90.0,
            "fgsm_accuracy": 62.0,
            "pgd_accuracy": 58.0,
            "fgsm_attack_success_rate": 30.0,
            "pgd_attack_success_rate": 35.0,
            "worst_case_attack_success_rate": 38.0,
        },
        trust_flag={
            "status": "PASS",
            "summary": "All configured evaluation checks passed.",
            "passed": 4,
            "failed": 0,
            "total": 4,
        },
        gradient_masking={
            "status": "NO_STRONG_SIGNAL",
            "summary": "No strong gradient-masking signal detected.",
            "warning_reasons": [],
        },
    )


def get_candidate_fixture(metric_overrides=None, meta_overrides=None, **kwargs) -> EvaluationSnapshot:
    """Creates a candidate snapshot fixture derived from baseline with overrides."""
    base = get_base_fixture()
    metrics = dict(base.metrics)
    if metric_overrides:
        metrics.update(metric_overrides)

    meta_dict = base.metadata.to_dict()
    if meta_overrides:
        meta_dict.update(meta_overrides)
    meta_dict["model_name"] = "CandidateModel_v2"
    metadata = EvaluationMetadata.from_dict(meta_dict)

    trust_flag = kwargs.get("trust_flag", base.trust_flag)
    gradient_masking = kwargs.get("gradient_masking", base.gradient_masking)
    raw_checks = kwargs.get("raw_checks", None)

    return EvaluationSnapshot(
        metadata=metadata,
        metrics=metrics,
        trust_flag=trust_flag,
        gradient_masking=gradient_masking,
        raw_checks=raw_checks,
    )


class TestModelSecurityRegressionGate(unittest.TestCase):

    # -------------------------------------------------------------
    # Existing Tests 1-16
    # -------------------------------------------------------------

    def test_pass_clean_equivalent_or_improved_candidate(self):
        """Candidate with identical or improved metrics should achieve PASS."""
        base = get_base_fixture()
        cand = get_candidate_fixture(
            metric_overrides={
                "clean_accuracy": 91.0,
                "pgd_accuracy": 60.0,
                "pgd_attack_success_rate": 33.0,
                "worst_case_attack_success_rate": 36.0,
            }
        )

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.PASS)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(len(result.warnings), 0)
        self.assertGreater(len(result.passed_checks), 0)
        self.assertIn("clean_accuracy", result.metric_deltas)
        self.assertEqual(result.metric_deltas["clean_accuracy"].status, MetricStatus.OK)
        self.assertEqual(result.metric_deltas["pgd_accuracy"].status, MetricStatus.OK)

    def test_warning_level_degradation(self):
        """Degradation inside the warning band should trigger WARN without failing."""
        base = get_base_fixture()
        cand = get_candidate_fixture(
            metric_overrides={
                "clean_accuracy": 89.2,
                "pgd_attack_success_rate": 36.2,
            }
        )

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.WARN)
        self.assertEqual(len(result.failures), 0)
        self.assertGreater(len(result.warnings), 0)
        self.assertEqual(result.metric_deltas["clean_accuracy"].status, MetricStatus.WARN)
        self.assertEqual(result.metric_deltas["pgd_attack_success_rate"].status, MetricStatus.WARN)

    def test_clean_accuracy_failure(self):
        """Clean accuracy drop exceeding max_clean_accuracy_drop should FAIL."""
        base = get_base_fixture()
        cand = get_candidate_fixture(metric_overrides={"clean_accuracy": 87.0})

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertGreater(len(result.failures), 0)
        self.assertTrue(any("clean_accuracy dropped by 3.00%" in f for f in result.failures))
        self.assertEqual(result.metric_deltas["clean_accuracy"].status, MetricStatus.FAIL)

    def test_adversarial_accuracy_failure(self):
        """PGD robust accuracy drop exceeding max_pgd_accuracy_drop should FAIL."""
        base = get_base_fixture()
        cand = get_candidate_fixture(metric_overrides={"pgd_accuracy": 53.0})

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("pgd_accuracy dropped by 5.00%" in f for f in result.failures))
        self.assertEqual(result.metric_deltas["pgd_accuracy"].status, MetricStatus.FAIL)

    def test_asr_increase_failure(self):
        """Surge in PGD or worst-case ASR exceeding max threshold should FAIL."""
        base = get_base_fixture()
        cand = get_candidate_fixture(metric_overrides={"worst_case_attack_success_rate": 43.0})

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("worst_case_attack_success_rate increased by +5.00%" in f for f in result.failures))
        self.assertEqual(result.metric_deltas["worst_case_attack_success_rate"].status, MetricStatus.FAIL)

    def test_condition_mismatch_epsilon(self):
        """Mismatch in attack epsilon between baseline and candidate should FAIL."""
        base = get_base_fixture()
        cand = get_candidate_fixture(meta_overrides={"epsilon": 16.0 / 255.0})

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("Perturbation budget (epsilon) mismatch" in f for f in result.failures))
        self.assertFalse(result.condition_compatibility["epsilon"]["matched"])

    def test_condition_mismatch_sample_count(self):
        """Mismatch in sample counts should FAIL."""
        base = get_base_fixture()
        cand = get_candidate_fixture(meta_overrides={"num_samples": 16})

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("Sample count mismatch" in f for f in result.failures))

    def test_missing_required_metric(self):
        """Snapshots missing clean_accuracy or adversarial metrics must FAIL."""
        base = get_base_fixture()
        cand_no_clean = get_candidate_fixture()
        del cand_no_clean.metrics["clean_accuracy"]

        result = evaluate_regression_gate(base, cand_no_clean)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("missing mandatory 'clean_accuracy'" in f for f in result.failures))

        cand_no_adv = EvaluationSnapshot(
            metadata=base.metadata,
            metrics={"clean_accuracy": 92.0},
        )
        result_no_adv = evaluate_regression_gate(base, cand_no_adv)
        self.assertEqual(result_no_adv.verdict, GateVerdict.FAIL)
        self.assertTrue(any("at least one adversarial metric" in f for f in result_no_adv.failures))

    def test_invalid_nan_and_infinity_metrics(self):
        """NaN or Inf metrics in candidate snapshot must be rejected with FAIL."""
        base = get_base_fixture()

        cand_nan = get_candidate_fixture(metric_overrides={"clean_accuracy": float("nan")})
        res_nan = evaluate_regression_gate(base, cand_nan)
        self.assertEqual(res_nan.verdict, GateVerdict.FAIL)
        self.assertTrue(any("invalid non-finite value" in f for f in res_nan.failures))

        cand_inf = get_candidate_fixture(metric_overrides={"pgd_accuracy": float("inf")})
        res_inf = evaluate_regression_gate(base, cand_inf)
        self.assertEqual(res_inf.verdict, GateVerdict.FAIL)
        self.assertTrue(any("invalid non-finite value" in f for f in res_inf.failures))

    def test_out_of_range_and_non_numeric_metrics(self):
        """Out of range metrics (<0 or >100) or non-numeric types must FAIL."""
        base = get_base_fixture()

        cand_out_of_range = get_candidate_fixture(metric_overrides={"clean_accuracy": 105.0})
        res_oor = evaluate_regression_gate(base, cand_out_of_range)
        self.assertEqual(res_oor.verdict, GateVerdict.FAIL)
        self.assertTrue(any("out of valid percentage range" in f for f in res_oor.failures))

        cand_string_val = get_candidate_fixture(metric_overrides={"pgd_accuracy": "eighty percent"})
        res_str = evaluate_regression_gate(base, cand_string_val)
        self.assertEqual(res_str.verdict, GateVerdict.FAIL)
        self.assertTrue(any("is non-numeric" in f for f in res_str.failures))

    def test_trust_check_failure_propagation(self):
        """Candidate trust checks failure must propagate as a gate FAIL."""
        base = get_base_fixture()
        cand = get_candidate_fixture(
            trust_flag={
                "status": "FAIL",
                "summary": "Perturbation budget exceeded by 0.005.",
                "passed": 3,
                "failed": 1,
                "total": 4,
            }
        )

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("failed ModelShield trust checks" in f for f in result.failures))

    def test_gradient_masking_warning_rejection(self):
        """Candidate showing gradient-masking WARNING must FAIL to prevent false security."""
        base = get_base_fixture()
        cand = get_candidate_fixture(
            metric_overrides={"pgd_accuracy": 75.0, "pgd_attack_success_rate": 15.0},
            gradient_masking={
                "status": "WARNING",
                "summary": "Potential gradient masking detected.",
                "warning_reasons": ["Loss landscape ray non-monotonic", "PGD restart variance > 20%"],
            },
        )

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("suspicious gradient masking signals" in f for f in result.failures))

    def test_missing_optional_diagnostics_not_falsely_reported(self):
        """Missing optional trust checks or gradient masking are not reported as passed."""
        base = get_base_fixture()
        cand = get_candidate_fixture(trust_flag=None, gradient_masking=None)

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.PASS)
        self.assertFalse(any("passed ModelShield trust checks" in p for p in result.passed_checks))
        self.assertFalse(any("passed gradient-masking diagnostics" in p for p in result.passed_checks))

    def test_custom_thresholds_override(self):
        """Strict custom thresholds must be obeyed."""
        base = get_base_fixture()
        cand = get_candidate_fixture(metric_overrides={"clean_accuracy": 89.6})

        res_default = evaluate_regression_gate(base, cand)
        self.assertEqual(res_default.verdict, GateVerdict.PASS)

        strict_th = RegressionThresholds(max_clean_accuracy_drop=0.3, warn_clean_accuracy_drop=0.1)
        res_strict = evaluate_regression_gate(base, cand, thresholds=strict_th)
        self.assertEqual(res_strict.verdict, GateVerdict.FAIL)
        self.assertTrue(any("clean_accuracy dropped by 0.40%" in f for f in res_strict.failures))

    def test_threshold_boundaries(self):
        """
        Tests exact boundary edges for warning and failure thresholds.
        Convention: drop <= threshold is allowed; drop > threshold breaches.
        """
        base = get_base_fixture()
        th = RegressionThresholds(
            max_clean_accuracy_drop=2.0,
            warn_clean_accuracy_drop=0.5,
            max_pgd_asr_increase=3.0,
            warn_pgd_asr_increase=1.0,
        )

        # 1. Exactly at warning boundary: drop == 0.50% -> OK
        cand_at_warn = get_candidate_fixture(metric_overrides={"clean_accuracy": 89.5})
        res_at_warn = evaluate_regression_gate(base, cand_at_warn, thresholds=th)
        self.assertEqual(res_at_warn.metric_deltas["clean_accuracy"].status, MetricStatus.OK)
        self.assertEqual(res_at_warn.verdict, GateVerdict.PASS)

        # 2. Just beyond warning boundary: drop == 0.501% -> WARN
        cand_over_warn = get_candidate_fixture(metric_overrides={"clean_accuracy": 89.499})
        res_over_warn = evaluate_regression_gate(base, cand_over_warn, thresholds=th)
        self.assertEqual(res_over_warn.metric_deltas["clean_accuracy"].status, MetricStatus.WARN)
        self.assertEqual(res_over_warn.verdict, GateVerdict.WARN)

        # 3. Exactly at fail boundary: drop == 2.00% -> WARN
        cand_at_fail = get_candidate_fixture(metric_overrides={"clean_accuracy": 88.0})
        res_at_fail = evaluate_regression_gate(base, cand_at_fail, thresholds=th)
        self.assertEqual(res_at_fail.metric_deltas["clean_accuracy"].status, MetricStatus.WARN)
        self.assertEqual(res_at_fail.verdict, GateVerdict.WARN)

        # 4. Just beyond fail boundary: drop == 2.001% -> FAIL
        cand_over_fail = get_candidate_fixture(metric_overrides={"clean_accuracy": 87.999})
        res_over_fail = evaluate_regression_gate(base, cand_over_fail, thresholds=th)
        self.assertEqual(res_over_fail.metric_deltas["clean_accuracy"].status, MetricStatus.FAIL)
        self.assertEqual(res_over_fail.verdict, GateVerdict.FAIL)

        # 5. Exactly at ASR surge fail boundary: increase == +3.00% -> WARN
        cand_asr_at_fail = get_candidate_fixture(metric_overrides={"pgd_attack_success_rate": 38.0})
        res_asr_at_fail = evaluate_regression_gate(base, cand_asr_at_fail, thresholds=th)
        self.assertEqual(res_asr_at_fail.metric_deltas["pgd_attack_success_rate"].status, MetricStatus.WARN)

        # 6. Just beyond ASR surge fail boundary: increase == +3.01% -> FAIL
        cand_asr_over_fail = get_candidate_fixture(metric_overrides={"pgd_attack_success_rate": 38.01})
        res_asr_over_fail = evaluate_regression_gate(base, cand_asr_over_fail, thresholds=th)
        self.assertEqual(res_asr_over_fail.metric_deltas["pgd_attack_success_rate"].status, MetricStatus.FAIL)

    def test_json_and_markdown_exports(self):
        """Verifies JSON and Markdown report generation."""
        base = get_base_fixture()
        cand = get_candidate_fixture(metric_overrides={"clean_accuracy": 87.0})
        result = evaluate_regression_gate(base, cand)

        json_str = export_gate_report_json(result)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["verdict"], "FAIL")
        self.assertIn("metric_deltas", parsed)
        self.assertIn("clean_accuracy", parsed["metric_deltas"])
        self.assertEqual(parsed["metric_deltas"]["clean_accuracy"]["status"], "FAIL")

        md_str = export_gate_report_markdown(result)
        self.assertIn("# 🛡️ ModelShield Security Regression Gate: FAIL ❌", md_str)
        self.assertIn("| `clean_accuracy` |", md_str)
        self.assertIn("### ❌ Critical Violations (FAIL)", md_str)

    # -------------------------------------------------------------
    # New Targeted Regression Tests (Review Phase)
    # -------------------------------------------------------------

    def test_condition_mismatch_dataset_name(self):
        """Mismatch in dataset name (e.g. CIFAR-10 vs MNIST) should FAIL."""
        base = get_base_fixture()
        cand = get_candidate_fixture(meta_overrides={"dataset_name": "MNIST"})

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("Dataset mismatch" in f for f in result.failures))

    def test_trust_status_warning(self):
        """Candidate trust status of WARNING should trigger gate WARN."""
        base = get_base_fixture()
        cand = get_candidate_fixture(
            trust_flag={
                "status": "WARNING",
                "summary": "1 check has invalid results.",
                "passed": 3,
                "failed": 0,
                "total": 4,
            }
        )

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.WARN)
        self.assertEqual(len(result.failures), 0)
        self.assertTrue(any("trust checks warning" in w for w in result.warnings))

    def test_gradient_masking_status_inconclusive(self):
        """Candidate gradient masking status of INCONCLUSIVE should trigger gate WARN."""
        base = get_base_fixture()
        cand = get_candidate_fixture(
            gradient_masking={
                "status": "INCONCLUSIVE",
                "summary": "Insufficient clean-correct samples to evaluate masking.",
                "warning_reasons": [],
            }
        )

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.WARN)
        self.assertEqual(len(result.failures), 0)
        self.assertTrue(any("diagnostics were inconclusive" in w for w in result.warnings))

    def test_enforce_condition_equality_false(self):
        """When enforce_condition_equality=False, condition mismatches become WARN instead of FAIL."""
        base = get_base_fixture()
        # Epsilon mismatch (16/255 vs 8/255)
        cand = get_candidate_fixture(meta_overrides={"epsilon": 16.0 / 255.0})

        relaxed_th = RegressionThresholds(enforce_condition_equality=False)
        result = evaluate_regression_gate(base, cand, thresholds=relaxed_th)

        self.assertEqual(result.verdict, GateVerdict.WARN)
        self.assertEqual(len(result.failures), 0)
        self.assertTrue(any("Condition Compatibility Warning" in w for w in result.warnings))

    def test_negative_metric_values(self):
        """Negative accuracy or ASR values should be strictly rejected with FAIL."""
        base = get_base_fixture()

        cand_neg_clean = get_candidate_fixture(metric_overrides={"clean_accuracy": -1.0})
        res_neg = evaluate_regression_gate(base, cand_neg_clean)
        self.assertEqual(res_neg.verdict, GateVerdict.FAIL)
        self.assertTrue(any("out of valid percentage range" in f for f in res_neg.failures))

        cand_neg_asr = get_candidate_fixture(metric_overrides={"pgd_attack_success_rate": -0.05})
        res_asr = evaluate_regression_gate(base, cand_neg_asr)
        self.assertEqual(res_asr.verdict, GateVerdict.FAIL)
        self.assertTrue(any("out of valid percentage range" in f for f in res_asr.failures))

    def test_missing_or_empty_metadata_and_zero_samples(self):
        """Snapshots with missing metadata, empty dataset, or num_samples <= 0 must FAIL."""
        base = get_base_fixture()

        # Raw dict snapshot with missing metadata dict
        raw_no_meta = {
            "metrics": dict(base.metrics),
        }
        res_no_meta = evaluate_regression_gate(base, raw_no_meta)
        self.assertEqual(res_no_meta.verdict, GateVerdict.FAIL)
        self.assertTrue(any("missing mandatory 'metadata'" in f for f in res_no_meta.failures))

        # Snapshot with num_samples = 0
        cand_zero_samples = get_candidate_fixture(meta_overrides={"num_samples": 0})
        res_zero = evaluate_regression_gate(base, cand_zero_samples)
        self.assertEqual(res_zero.verdict, GateVerdict.FAIL)
        self.assertTrue(any("num_samples' must be an integer greater than zero" in f for f in res_zero.failures))

        # Snapshot with empty dataset_name
        cand_empty_ds = get_candidate_fixture(meta_overrides={"dataset_name": ""})
        res_ds = evaluate_regression_gate(base, cand_empty_ds)
        self.assertEqual(res_ds.verdict, GateVerdict.FAIL)
        self.assertTrue(any("valid non-empty 'dataset_name'" in f for f in res_ds.failures))

    def test_invalid_epsilon_values(self):
        """Epsilon <= 0.0, NaN, Inf, or non-numeric must FAIL."""
        base = get_base_fixture()

        cand_zero_eps = get_candidate_fixture(meta_overrides={"epsilon": 0.0})
        res_zero = evaluate_regression_gate(base, cand_zero_eps)
        self.assertEqual(res_zero.verdict, GateVerdict.FAIL)
        self.assertTrue(any("epsilon' must be a finite number greater than zero" in f for f in res_zero.failures))

        cand_neg_eps = get_candidate_fixture(meta_overrides={"epsilon": -0.05})
        res_neg = evaluate_regression_gate(base, cand_neg_eps)
        self.assertEqual(res_neg.verdict, GateVerdict.FAIL)
        self.assertTrue(any("epsilon' must be a finite number greater than zero" in f for f in res_neg.failures))

        cand_nan_eps = get_candidate_fixture(meta_overrides={"epsilon": float("nan")})
        res_nan = evaluate_regression_gate(base, cand_nan_eps)
        self.assertEqual(res_nan.verdict, GateVerdict.FAIL)
        self.assertTrue(any("epsilon' must be a finite number greater than zero" in f for f in res_nan.failures))

    def test_pgd_parameters_missing_when_pgd_metrics_present(self):
        """When PGD metrics are present in both snapshots, missing pgd_steps must FAIL."""
        base = get_base_fixture()
        # Candidate has PGD metrics but pgd_steps=None
        cand = get_candidate_fixture(meta_overrides={"pgd_steps": None})

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("Candidate metadata is missing 'pgd_steps'" in f for f in result.failures))

    def test_pgd_steps_mismatch_when_pgd_metrics_present(self):
        """When PGD metrics are present in both, different pgd_steps must FAIL."""
        base = get_base_fixture()
        cand = get_candidate_fixture(meta_overrides={"pgd_steps": 20})

        result = evaluate_regression_gate(base, cand)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("PGD steps mismatch: Baseline=10, Candidate=20" in f for f in result.failures))

    def test_pgd_parameters_not_required_when_no_pgd_metrics(self):
        """When snapshots only have FGSM metrics (no PGD metrics), missing pgd_steps is acceptable."""
        base_fgsm_only = EvaluationSnapshot(
            metadata=EvaluationMetadata(
                model_name="BaseModel",
                dataset_name="CIFAR-10",
                num_samples=32,
                epsilon=8.0 / 255.0,
                pgd_steps=None,
            ),
            metrics={
                "clean_accuracy": 90.0,
                "fgsm_accuracy": 62.0,
                "fgsm_attack_success_rate": 30.0,
            },
        )
        cand_fgsm_only = EvaluationSnapshot(
            metadata=EvaluationMetadata(
                model_name="CandModel",
                dataset_name="CIFAR-10",
                num_samples=32,
                epsilon=8.0 / 255.0,
                pgd_steps=None,
            ),
            metrics={
                "clean_accuracy": 90.5,
                "fgsm_accuracy": 63.0,
                "fgsm_attack_success_rate": 29.0,
            },
        )

        result = evaluate_regression_gate(base_fgsm_only, cand_fgsm_only)

        self.assertEqual(result.verdict, GateVerdict.PASS)
        self.assertEqual(len(result.failures), 0)
        self.assertIn("fgsm_accuracy", result.metric_deltas)


if __name__ == "__main__":
    unittest.main()
