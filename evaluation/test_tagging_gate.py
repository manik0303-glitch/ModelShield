"""
Unit test suite for ModelShield Multi-Label Image-Tagging Regression Gate (RAM++ Support).

Tests validation, condition compatibility, delta calculations, thresholds,
verdicts, and report serialization for image-tagging evaluations.
"""

import json
import math
import unittest

from evaluation.regression_gate import (
    GateVerdict,
    MetricDirection,
    MetricStatus,
    TaskType,
    EvaluationMetadata,
    EvaluationSnapshot,
    RegressionThresholds,
    evaluate_regression_gate,
    export_gate_report_json,
    export_gate_report_markdown,
    validate_snapshot_integrity,
    CLASSIFICATION_METRICS,
    IMAGE_TAGGING_METRICS,
    SUPPORTED_METRICS,
)


def get_base_tagging_fixture() -> EvaluationSnapshot:
    """Creates a standard healthy baseline snapshot fixture for RAM++ image tagging."""
    return EvaluationSnapshot(
        metadata=EvaluationMetadata(
            model_name="RAM_Plus_SwinL_Baseline",
            dataset_name="COCO-Tagging",
            num_samples=50,
            epsilon=8.0 / 255.0,
            alpha=2.0 / 255.0,
            pgd_steps=10,
            random_start=True,
            timestamp="2026-11-23T12:00:00Z",
            task_type="image_tagging",
            extra={"architecture": "Swin-Large", "tag_vocabulary_size": 4585},
        ),
        metrics={
            "clean_map": 85.0,
            "clean_precision": 82.0,
            "clean_recall": 80.0,
            "clean_f1": 81.0,
            "fgsm_map": 65.0,
            "pgd_map": 60.0,
            "tag_retention_rate": 75.0,
            "tag_drop_rate": 25.0,
            "tag_injection_rate": 15.0,
        },
        trust_flag={
            "status": "PASS",
            "summary": "All evaluation trust checks passed.",
            "passed": 4,
            "failed": 0,
            "total": 4,
        },
        gradient_masking={
            "status": "NO_STRONG_SIGNAL",
            "summary": "No gradient-masking signal detected.",
            "warning_reasons": [],
        },
    )


def get_candidate_tagging_fixture(metric_overrides=None, meta_overrides=None, **kwargs) -> EvaluationSnapshot:
    """Creates a candidate tagging snapshot fixture derived from baseline with overrides."""
    base = get_base_tagging_fixture()
    metrics = dict(base.metrics)
    if metric_overrides:
        metrics.update(metric_overrides)

    meta_dict = base.metadata.to_dict()
    if meta_overrides:
        meta_dict.update(meta_overrides)
    meta_dict["model_name"] = "Candidate_RAM_Plus_v2"
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


def get_classification_fixture() -> EvaluationSnapshot:
    """Standard healthy classification snapshot fixture."""
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
            task_type="classification",
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
            "summary": "All checks passed.",
            "passed": 4,
            "failed": 0,
            "total": 4,
        },
        gradient_masking={
            "status": "NO_STRONG_SIGNAL",
            "summary": "No gradient-masking signal detected.",
            "warning_reasons": [],
        },
    )


class TestTaggingRegressionGate(unittest.TestCase):

    def test_metric_directions_and_definitions(self):
        """Verifies that all image-tagging metrics have correct directions defined."""
        # Higher is better
        self.assertEqual(IMAGE_TAGGING_METRICS["clean_map"], MetricDirection.HIGHER_IS_BETTER)
        self.assertEqual(IMAGE_TAGGING_METRICS["clean_precision"], MetricDirection.HIGHER_IS_BETTER)
        self.assertEqual(IMAGE_TAGGING_METRICS["clean_recall"], MetricDirection.HIGHER_IS_BETTER)
        self.assertEqual(IMAGE_TAGGING_METRICS["clean_f1"], MetricDirection.HIGHER_IS_BETTER)
        self.assertEqual(IMAGE_TAGGING_METRICS["fgsm_map"], MetricDirection.HIGHER_IS_BETTER)
        self.assertEqual(IMAGE_TAGGING_METRICS["pgd_map"], MetricDirection.HIGHER_IS_BETTER)
        self.assertEqual(IMAGE_TAGGING_METRICS["tag_retention_rate"], MetricDirection.HIGHER_IS_BETTER)

        # Lower is better
        self.assertEqual(IMAGE_TAGGING_METRICS["tag_drop_rate"], MetricDirection.LOWER_IS_BETTER)
        self.assertEqual(IMAGE_TAGGING_METRICS["tag_injection_rate"], MetricDirection.LOWER_IS_BETTER)

        # SUPPORTED_METRICS contains both
        for k, v in IMAGE_TAGGING_METRICS.items():
            self.assertIn(k, SUPPORTED_METRICS)
            self.assertEqual(SUPPORTED_METRICS[k], v)

    def test_valid_image_tagging_snapshot_integrity(self):
        """Valid image-tagging snapshot passes integrity validation without clean_accuracy."""
        snap = get_base_tagging_fixture()
        is_valid, errors, obj = validate_snapshot_integrity(snap, "Baseline")
        self.assertTrue(is_valid, f"Validation failed unexpectedly: {errors}")
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(obj)
        self.assertEqual(obj.metadata.task_type, "image_tagging")

    def test_tagging_does_not_require_clean_accuracy(self):
        """Image-tagging snapshots must NOT require clean_accuracy."""
        snap = get_base_tagging_fixture()
        self.assertNotIn("clean_accuracy", snap.metrics)
        is_valid, errors, _ = validate_snapshot_integrity(snap, "Candidate")
        self.assertTrue(is_valid)
        self.assertFalse(any("clean_accuracy" in e for e in errors))

    def test_invalid_image_tagging_missing_clean_map(self):
        """Image-tagging snapshot missing clean_map must fail integrity validation."""
        snap = get_base_tagging_fixture()
        del snap.metrics["clean_map"]
        is_valid, errors, _ = validate_snapshot_integrity(snap, "Candidate")
        self.assertFalse(is_valid)
        self.assertTrue(any("clean_map" in e for e in errors))

    def test_invalid_image_tagging_missing_adversarial_metric(self):
        """Image-tagging snapshot with clean_map but no adversarial metrics must fail integrity."""
        snap = get_base_tagging_fixture()
        snap.metrics = {"clean_map": 85.0, "clean_precision": 82.0}
        is_valid, errors, _ = validate_snapshot_integrity(snap, "Candidate")
        self.assertFalse(is_valid)
        self.assertTrue(any("at least one adversarial metric" in e for e in errors))

    def test_invalid_unsupported_task_type(self):
        """Metadata with an unknown/unsupported task type must fail integrity validation."""
        snap = get_base_tagging_fixture()
        snap.metadata.task_type = "speech_recognition"
        is_valid, errors, _ = validate_snapshot_integrity(snap, "Candidate")
        self.assertFalse(is_valid)
        self.assertTrue(any("task_type" in e for e in errors))

    def test_pass_clean_equivalent_or_improved_tagging_candidate(self):
        """Tagging candidate with equivalent or superior metrics should achieve PASS."""
        base = get_base_tagging_fixture()
        cand = get_candidate_tagging_fixture(
            metric_overrides={
                "clean_map": 86.5,
                "pgd_map": 62.0,
                "tag_retention_rate": 78.0,
                "tag_drop_rate": 22.0,
                "tag_injection_rate": 13.0,
            }
        )
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.PASS)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(len(result.warnings), 0)
        self.assertIn("clean_map", result.metric_deltas)
        self.assertEqual(result.metric_deltas["clean_map"].status, MetricStatus.OK)

    def test_task_type_mismatch_cross_task_rejection(self):
        """Comparing a classification snapshot with an image-tagging snapshot must FAIL."""
        class_base = get_classification_fixture()
        tag_cand = get_base_tagging_fixture()

        result = evaluate_regression_gate(class_base, tag_cand)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("Task type mismatch" in f for f in result.failures))

    def test_task_type_mismatch_warning_when_not_enforced(self):
        """When enforce_condition_equality is False, task type mismatch becomes a warning."""
        class_base = get_classification_fixture()
        tag_cand = get_base_tagging_fixture()
        thresholds = RegressionThresholds(enforce_condition_equality=False)

        result = evaluate_regression_gate(class_base, tag_cand, thresholds)
        self.assertTrue(any("Task type mismatch" in w for w in result.warnings))

    def test_tagging_clean_map_drop_failure(self):
        """Drop in clean_map exceeding max_clean_map_drop (default 3.0) must FAIL."""
        base = get_base_tagging_fixture()  # clean_map: 85.0
        cand = get_candidate_tagging_fixture(metric_overrides={"clean_map": 81.5})  # drop 3.5%
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertEqual(result.metric_deltas["clean_map"].status, MetricStatus.FAIL)
        self.assertTrue(any("clean_map dropped" in f for f in result.failures))

    def test_tagging_pgd_map_drop_failure(self):
        """Drop in pgd_map exceeding max_pgd_map_drop (default 4.0) must FAIL."""
        base = get_base_tagging_fixture()  # pgd_map: 60.0
        cand = get_candidate_tagging_fixture(metric_overrides={"pgd_map": 55.0})  # drop 5.0%
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertEqual(result.metric_deltas["pgd_map"].status, MetricStatus.FAIL)
        self.assertTrue(any("pgd_map dropped" in f for f in result.failures))

    def test_tagging_tag_retention_drop_failure(self):
        """Drop in tag_retention_rate exceeding max_tag_retention_drop (default 5.0) must FAIL."""
        base = get_base_tagging_fixture()  # retention: 75.0
        cand = get_candidate_tagging_fixture(metric_overrides={"tag_retention_rate": 68.0})  # drop 7.0%
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertEqual(result.metric_deltas["tag_retention_rate"].status, MetricStatus.FAIL)

    def test_tagging_tag_drop_rate_surge_failure(self):
        """Surge in tag_drop_rate exceeding max_tag_drop_rate_increase (default 5.0) must FAIL."""
        base = get_base_tagging_fixture()  # drop_rate: 25.0
        cand = get_candidate_tagging_fixture(metric_overrides={"tag_drop_rate": 32.0})  # surge +7.0%
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertEqual(result.metric_deltas["tag_drop_rate"].status, MetricStatus.FAIL)

    def test_tagging_tag_injection_surge_failure(self):
        """Surge in tag_injection_rate exceeding max_tag_injection_rate_increase (default 5.0) must FAIL."""
        base = get_base_tagging_fixture()  # injection: 15.0
        cand = get_candidate_tagging_fixture(metric_overrides={"tag_injection_rate": 22.0})  # surge +7.0%
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertEqual(result.metric_deltas["tag_injection_rate"].status, MetricStatus.FAIL)

    def test_tagging_warning_level_degradation(self):
        """Degradation within the warning band must produce WARN without FAIL."""
        base = get_base_tagging_fixture()  # clean_map: 85.0, warn: 1.0, max: 3.0
        cand = get_candidate_tagging_fixture(metric_overrides={"clean_map": 83.5})  # drop 1.5%
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.WARN)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(result.metric_deltas["clean_map"].status, MetricStatus.WARN)
        self.assertTrue(any("clean_map dropped" in w for w in result.warnings))

    def test_tagging_non_finite_metric_rejected(self):
        """NaN or Inf metrics in tagging snapshot must be rejected with FAIL."""
        base = get_base_tagging_fixture()
        cand = get_candidate_tagging_fixture(metric_overrides={"clean_map": float("nan")})
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("invalid non-finite value" in f for f in result.failures))

    def test_tagging_out_of_range_metrics_rejected(self):
        """Tagging metrics outside [0.0, 100.0] must be rejected with FAIL."""
        base = get_base_tagging_fixture()
        cand = get_candidate_tagging_fixture(metric_overrides={"tag_retention_rate": 105.0})
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("out of valid percentage range" in f for f in result.failures))

    def test_tagging_pgd_parameters_compatibility(self):
        """When both tagging snapshots have pgd_map, PGD steps mismatch must FAIL."""
        base = get_base_tagging_fixture()  # pgd_steps: 10
        cand = get_candidate_tagging_fixture(meta_overrides={"pgd_steps": 20})
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertTrue(any("PGD steps mismatch" in f for f in result.failures))

    def test_tagging_json_and_markdown_roundtrip(self):
        """Verifies JSON and Markdown report generation for image-tagging evaluations."""
        base = get_base_tagging_fixture()
        cand = get_candidate_tagging_fixture(
            metric_overrides={
                "clean_map": 86.0,
                "pgd_map": 61.0,
            }
        )
        result = evaluate_regression_gate(base, cand)
        self.assertEqual(result.verdict, GateVerdict.PASS)

        # JSON Roundtrip
        json_str = export_gate_report_json(result)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["verdict"], "PASS")
        self.assertEqual(parsed["baseline_metadata"]["task_type"], "image_tagging")
        self.assertEqual(parsed["candidate_metadata"]["task_type"], "image_tagging")
        self.assertIn("clean_map", parsed["metric_deltas"])
        self.assertIn("pgd_map", parsed["metric_deltas"])
        self.assertEqual(parsed["metric_deltas"]["clean_map"]["delta"], 1.0)

        # Markdown Report
        md_str = export_gate_report_markdown(result)
        self.assertIn("# 🛡️ ModelShield Security Regression Gate: PASS ✅", md_str)
        self.assertIn("Task: image_tagging", md_str)
        self.assertIn("| `clean_map` |", md_str)
        self.assertIn("| `pgd_map` |", md_str)
        self.assertIn("| `tag_retention_rate` |", md_str)


if __name__ == "__main__":
    unittest.main()

