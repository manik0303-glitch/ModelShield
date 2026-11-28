"""
Unit Tests for Semantic Consistency Evaluation and Regression Gate Integration.
"""

from __future__ import annotations

import unittest
from evaluation.semantic_consistency import (
    SemanticConsistencyResult,
    compare_tag_sets,
    evaluate_semantic_regression,
)
from evaluation.regression_gate import (
    EvaluationMetadata,
    EvaluationSnapshot,
    GateVerdict,
    RegressionGateResult,
    RegressionThresholds,
    evaluate_regression_gate,
    export_gate_report_markdown,
)


class TestSemanticConsistency(unittest.TestCase):
    """Tests tag comparison logic, metric derivations, and regression thresholds."""

    def test_identical_tags(self):
        clean = ["dog", "golden retriever", "grass", "park"]
        adv = ["golden retriever", "dog", "park", "grass"]
        res = compare_tag_sets(clean, adv)

        self.assertEqual(res.overlap_ratio, 1.0)
        self.assertEqual(res.semantic_consistency, 100.0)
        self.assertEqual(res.tag_retention_rate, 100.0)
        self.assertEqual(res.tags_lost_count, 0)
        self.assertEqual(res.new_tags_count, 0)
        self.assertEqual(res.lost_tags, [])
        self.assertEqual(res.new_tags, [])
        self.assertEqual(len(res.overlap_tags), 4)

    def test_completely_disjoint_tags(self):
        clean = ["dog", "puppy"]
        adv = ["cat", "kitten", "sofa"]
        res = compare_tag_sets(clean, adv)

        self.assertEqual(res.overlap_ratio, 0.0)
        self.assertEqual(res.semantic_consistency, 0.0)
        self.assertEqual(res.tag_retention_rate, 0.0)
        self.assertEqual(res.tags_lost_count, 2)
        self.assertEqual(res.new_tags_count, 3)
        self.assertEqual(res.lost_tags, ["dog", "puppy"])
        self.assertEqual(res.new_tags, ["cat", "kitten", "sofa"])
        self.assertEqual(res.overlap_tags, [])

    def test_partial_overlap(self):
        clean = ["dog", "grass", "frisbee", "sun"]
        adv = ["dog", "grass", "cloud", "tree"]
        res = compare_tag_sets(clean, adv)

        # Overlap: dog, grass (2)
        # Lost: frisbee, sun (2)
        # New: cloud, tree (2)
        # Union: dog, grass, frisbee, sun, cloud, tree (6)
        # Jaccard = 2/6 = 0.3333 -> 33.33%
        self.assertEqual(res.overlap_tags, ["dog", "grass"])
        self.assertEqual(res.lost_tags, ["frisbee", "sun"])
        self.assertEqual(res.new_tags, ["cloud", "tree"])
        self.assertEqual(res.tags_lost_count, 2)
        self.assertEqual(res.new_tags_count, 2)
        self.assertAlmostEqual(res.overlap_ratio, 2.0 / 6.0, places=4)
        self.assertAlmostEqual(res.semantic_consistency, (2.0 / 6.0) * 100.0, places=1)
        self.assertEqual(res.tag_retention_rate, 50.0)

    def test_empty_both_tags(self):
        res = compare_tag_sets([], [])
        self.assertEqual(res.overlap_ratio, 1.0)
        self.assertEqual(res.semantic_consistency, 100.0)
        self.assertEqual(res.tags_lost_count, 0)
        self.assertEqual(res.new_tags_count, 0)

    def test_empty_clean_only(self):
        res = compare_tag_sets([], ["automobile", "wheel"])
        self.assertEqual(res.overlap_ratio, 0.0)
        self.assertEqual(res.semantic_consistency, 0.0)
        self.assertEqual(res.tags_lost_count, 0)
        self.assertEqual(res.new_tags_count, 2)

    def test_empty_adversarial_only(self):
        res = compare_tag_sets(["automobile", "wheel"], [])
        self.assertEqual(res.overlap_ratio, 0.0)
        self.assertEqual(res.semantic_consistency, 0.0)
        self.assertEqual(res.tags_lost_count, 2)
        self.assertEqual(res.new_tags_count, 0)

    def test_normalization_and_formatting(self):
        clean = [" DOG ", "Golden Retriever", "GRASS", "dog"]  # Has spaces, mixed case, duplicate
        adv = "golden retriever, grass, park | outdoor / field"  # Comma, pipe, and slash separated string
        res = compare_tag_sets(clean, adv)

        self.assertIn("dog", res.lost_tags)
        self.assertIn("golden retriever", res.overlap_tags)
        self.assertIn("grass", res.overlap_tags)
        self.assertIn("park", res.new_tags)
        self.assertIn("outdoor", res.new_tags)
        self.assertIn("field", res.new_tags)

    def test_none_and_edge_inputs(self):
        res = compare_tag_sets(None, None)
        self.assertEqual(res.overlap_ratio, 1.0)
        self.assertEqual(res.semantic_consistency, 100.0)
        self.assertEqual(res.tags_lost_count, 0)
        self.assertEqual(res.new_tags_count, 0)

        res2 = compare_tag_sets(["cat", None, "  "], ["cat"])
        self.assertEqual(res2.overlap_ratio, 1.0)
        self.assertEqual(res2.semantic_consistency, 100.0)
        self.assertEqual(res2.tags_lost_count, 0)
        self.assertEqual(res2.new_tags_count, 0)

    def test_serialization_roundtrip(self):
        res = compare_tag_sets(["car", "road"], ["car", "tree"])
        d = res.to_dict()
        reconstituted = SemanticConsistencyResult.from_dict(d)

        self.assertEqual(res.overlap_ratio, reconstituted.overlap_ratio)
        self.assertEqual(res.semantic_consistency, reconstituted.semantic_consistency)
        self.assertEqual(res.overlap_tags, reconstituted.overlap_tags)
        self.assertEqual(res.lost_tags, reconstituted.lost_tags)
        self.assertEqual(res.new_tags, reconstituted.new_tags)

    def test_semantic_regression_disabled(self):
        res = compare_tag_sets(["a"], ["b"])  # 0% consistency
        fail, warn, assessment = evaluate_semantic_regression(
            candidate_result=res,
            enabled=False,
        )
        self.assertIsNone(fail)
        self.assertIsNone(warn)
        self.assertIsNone(assessment)

    def test_semantic_regression_pass(self):
        base_res = compare_tag_sets(["a", "b", "c"], ["a", "b"])  # 66.67%
        cand_res = compare_tag_sets(["a", "b", "c"], ["a", "b", "c"])  # 100%
        fail, warn, assessment = evaluate_semantic_regression(
            candidate_result=cand_res,
            baseline_result=base_res,
            min_semantic_consistency=60.0,
            max_semantic_consistency_drop=10.0,
            enabled=True,
        )
        self.assertIsNone(fail)
        self.assertIsNone(warn)
        self.assertIsNotNone(assessment)
        self.assertEqual(assessment["status"], "PASS")

    def test_semantic_regression_below_min_threshold_fail(self):
        cand_res = compare_tag_sets(["a", "b", "c", "d"], ["a"])  # 25% consistency
        fail, warn, assessment = evaluate_semantic_regression(
            candidate_result=cand_res,
            min_semantic_consistency=60.0,
            enabled=True,
        )
        self.assertIsNotNone(fail)
        self.assertIn("fell below minimum allowable threshold", fail)
        self.assertEqual(assessment["status"], "FAIL")

    def test_semantic_regression_warn_threshold(self):
        # 66.67% consistency: above min 60.0% but below warn 75.0%
        cand_res = compare_tag_sets(["a", "b", "c"], ["a", "b"])
        fail, warn, assessment = evaluate_semantic_regression(
            candidate_result=cand_res,
            min_semantic_consistency=60.0,
            warn_semantic_consistency=75.0,
            enabled=True,
        )
        self.assertIsNone(fail)
        self.assertIsNotNone(warn)
        self.assertIn("fell below warning threshold", warn)
        self.assertEqual(assessment["status"], "WARN")

    def test_semantic_regression_excessive_drop_fail(self):
        # Baseline = 100%, Candidate = 80%, drop = 20% > max_drop = 10%
        base_res = SemanticConsistencyResult(semantic_consistency=100.0)
        cand_res = SemanticConsistencyResult(semantic_consistency=80.0)
        fail, warn, assessment = evaluate_semantic_regression(
            candidate_result=cand_res,
            baseline_result=base_res,
            min_semantic_consistency=60.0,
            max_semantic_consistency_drop=10.0,
            enabled=True,
        )
        self.assertIsNotNone(fail)
        self.assertIn("exceeding max allowable drop", fail)
        self.assertEqual(assessment["status"], "FAIL")

    def test_semantic_regression_warning_drop(self):
        # Baseline = 100%, Candidate = 92%, drop = 8% > warn_drop = 5% but <= max_drop = 10%
        base_res = SemanticConsistencyResult(semantic_consistency=100.0)
        cand_res = SemanticConsistencyResult(semantic_consistency=92.0)
        fail, warn, assessment = evaluate_semantic_regression(
            candidate_result=cand_res,
            baseline_result=base_res,
            min_semantic_consistency=60.0,
            warn_semantic_consistency=75.0,
            max_semantic_consistency_drop=10.0,
            warn_semantic_consistency_drop=5.0,
            enabled=True,
        )
        self.assertIsNone(fail)
        self.assertIsNotNone(warn)
        self.assertIn("exceeding warning drop threshold", warn)
        self.assertEqual(assessment["status"], "WARN")

    def test_semantic_regression_missing_candidate_warns(self):
        fail, warn, assessment = evaluate_semantic_regression(
            candidate_result=None,
            enabled=True,
        )
        self.assertIsNone(fail)
        self.assertIsNotNone(warn)
        self.assertIn("does not contain semantic consistency", warn)
        self.assertEqual(assessment["status"], "MISSING")


class TestRegressionGateSemanticIntegration(unittest.TestCase):
    """Tests end-to-end integration of semantic consistency with evaluate_regression_gate."""

    def _make_snapshot(self, model_name: str, clean_acc: float = 90.0, pgd_acc: float = 60.0, sc_dict: dict = None):
        meta = EvaluationMetadata(
            model_name=model_name,
            dataset_name="CIFAR-10",
            num_samples=16,
            epsilon=8.0 / 255.0,
            pgd_steps=10,
            task_type="classification",
        )
        metrics = {
            "clean_accuracy": clean_acc,
            "pgd_accuracy": pgd_acc,
            "fgsm_accuracy": 70.0,
            "pgd_attack_success_rate": 40.0,
            "fgsm_attack_success_rate": 30.0,
            "worst_case_attack_success_rate": 40.0,
        }
        return EvaluationSnapshot(
            metadata=meta,
            metrics=metrics,
            semantic_consistency=sc_dict,
        )

    def test_gate_ignores_semantic_consistency_when_disabled(self):
        base_snap = self._make_snapshot("BaseModel", clean_acc=90.0, pgd_acc=60.0)
        # Candidate has terrible 10% semantic consistency, but semantic check is disabled
        cand_sc = SemanticConsistencyResult(semantic_consistency=10.0).to_dict()
        cand_snap = self._make_snapshot("CandModel", clean_acc=90.0, pgd_acc=60.0, sc_dict=cand_sc)

        thresholds = RegressionThresholds(enable_semantic_consistency=False)
        result = evaluate_regression_gate(base_snap, cand_snap, thresholds=thresholds)

        self.assertEqual(result.verdict, GateVerdict.PASS)
        self.assertIsNone(result.semantic_consistency_assessment)

    def test_gate_fails_when_candidate_semantic_consistency_breached(self):
        base_snap = self._make_snapshot("BaseModel", clean_acc=90.0, pgd_acc=60.0)
        # Candidate has 40% semantic consistency, min is 60%
        cand_sc = SemanticConsistencyResult(semantic_consistency=40.0).to_dict()
        cand_snap = self._make_snapshot("CandModel", clean_acc=90.0, pgd_acc=60.0, sc_dict=cand_sc)

        thresholds = RegressionThresholds(
            enable_semantic_consistency=True,
            min_semantic_consistency=60.0,
        )
        result = evaluate_regression_gate(base_snap, cand_snap, thresholds=thresholds)

        self.assertEqual(result.verdict, GateVerdict.FAIL)
        self.assertIsNotNone(result.semantic_consistency_assessment)
        self.assertEqual(result.semantic_consistency_assessment["status"], "FAIL")
        self.assertTrue(any("semantic consistency" in f.lower() for f in result.failures))

    def test_gate_passes_when_semantic_consistency_satisfies_thresholds(self):
        base_sc = SemanticConsistencyResult(semantic_consistency=85.0).to_dict()
        cand_sc = SemanticConsistencyResult(semantic_consistency=88.0).to_dict()
        base_snap = self._make_snapshot("BaseModel", clean_acc=90.0, pgd_acc=60.0, sc_dict=base_sc)
        cand_snap = self._make_snapshot("CandModel", clean_acc=90.0, pgd_acc=60.0, sc_dict=cand_sc)

        thresholds = RegressionThresholds(
            enable_semantic_consistency=True,
            min_semantic_consistency=60.0,
            warn_semantic_consistency=75.0,
            max_semantic_consistency_drop=10.0,
        )
        result = evaluate_regression_gate(base_snap, cand_snap, thresholds=thresholds)

        self.assertEqual(result.verdict, GateVerdict.PASS)
        self.assertIsNotNone(result.semantic_consistency_assessment)
        self.assertEqual(result.semantic_consistency_assessment["status"], "PASS")

    def test_markdown_report_includes_semantic_consistency(self):
        sc_dict = compare_tag_sets(["dog", "park"], ["dog", "cat"]).to_dict()
        snap = self._make_snapshot("CandModel", sc_dict=sc_dict)
        thresholds = RegressionThresholds(enable_semantic_consistency=True)
        res = evaluate_regression_gate(snap, snap, thresholds=thresholds)

        md = export_gate_report_markdown(res)
        self.assertIn("Semantic Consistency (RAM++)", md)
        self.assertIn("Candidate Consistency", md)


if __name__ == "__main__":
    unittest.main()

