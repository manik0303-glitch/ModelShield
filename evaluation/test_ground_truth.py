"""
Unit tests for evaluation/ground_truth.py

Covers:
  - compute_precision_recall_f1: edge cases and partial matches
  - evaluate_benchmark_quality: per-image and aggregate values, GT-only filtering
  - limitation_note presence
  - Non-benchmark images silently skipped
"""
import unittest
from evaluation.ground_truth import (
    BENCHMARK_GROUND_TRUTH,
    BENCHMARK_CLASS_NAMES,
    compute_precision_recall_f1,
    evaluate_benchmark_quality,
)


class TestComputePrecisionRecallF1(unittest.TestCase):

    def test_identical_sets(self):
        """Perfect predictions → (1, 1, 1)."""
        gt = frozenset({"cat", "animal", "pet"})
        pred = ["cat", "animal", "pet"]
        p, r, f = compute_precision_recall_f1(pred, gt)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 1.0)

    def test_disjoint_sets(self):
        """Completely wrong predictions → (0, 0, 0)."""
        gt = frozenset({"cat", "animal"})
        pred = ["table", "chair"]
        p, r, f = compute_precision_recall_f1(pred, gt)
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(f, 0.0)

    def test_partial_overlap(self):
        """Half the GT matched → recall=0.5; precision depends on pred size."""
        gt = frozenset({"cat", "animal"})
        pred = ["cat", "table"]  # 1 TP out of 2 pred, 1 TP out of 2 GT
        p, r, f = compute_precision_recall_f1(pred, gt)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(f, 0.5)

    def test_empty_predictions(self):
        """No predictions at all → (0, 0, 0)."""
        gt = frozenset({"cat", "animal"})
        p, r, f = compute_precision_recall_f1([], gt)
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(f, 0.0)

    def test_empty_ground_truth(self):
        """No ground truth (vacuously true) → (1, 1, 1)."""
        p, r, f = compute_precision_recall_f1(["cat", "animal"], frozenset())
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 1.0)

    def test_both_empty(self):
        """Both empty → vacuously (1, 1, 1) since there is no GT."""
        p, r, f = compute_precision_recall_f1([], frozenset())
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 1.0)

    def test_superset_predictions(self):
        """Extra tags beyond GT → precision < 1, recall = 1."""
        gt = frozenset({"cat"})
        pred = ["cat", "animal", "pet", "kitten"]  # 1 TP / 4 pred
        p, r, f = compute_precision_recall_f1(pred, gt)
        self.assertAlmostEqual(p, 0.25)
        self.assertAlmostEqual(r, 1.0)
        expected_f1 = 2 * 0.25 * 1.0 / (0.25 + 1.0)
        self.assertAlmostEqual(f, round(expected_f1, 4), places=3)

    def test_single_exact_match(self):
        """Single tag, exact match → (1, 1, 1) even with large GT set."""
        gt = frozenset({"cat", "animal", "pet", "kitten", "tabby"})
        pred = ["cat"]  # 1/5 GT matched, 1/1 pred matched
        p, r, f = compute_precision_recall_f1(pred, gt)
        self.assertAlmostEqual(p, 1.0)      # 1 TP / 1 pred
        self.assertAlmostEqual(r, 0.2)      # 1 TP / 5 GT
        expected_f1 = 2 * 1.0 * 0.2 / (1.0 + 0.2)
        self.assertAlmostEqual(f, round(expected_f1, 4), places=3)

    def test_return_values_bounded(self):
        """All returned values must lie in [0.0, 1.0]."""
        import random
        random.seed(0)
        for _ in range(50):
            n_pred = random.randint(0, 10)
            n_gt = random.randint(0, 10)
            pred = [f"tag_{i}" for i in random.sample(range(20), n_pred)]
            gt = frozenset(f"tag_{i}" for i in random.sample(range(20), n_gt))
            p, r, f = compute_precision_recall_f1(pred, gt)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)


class TestEvaluateBenchmarkQuality(unittest.TestCase):

    def test_known_benchmark_images_exact_match(self):
        """
        Feed one benchmark image with its full GT set as predictions.
        Expect precision=1 (all preds in GT), recall=1 (all GT in preds), f1=1.
        """
        filename = "04_cat.png"
        gt = BENCHMARK_GROUND_TRUTH[filename]
        image_name_to_tags = {filename: sorted(list(gt))}
        result = evaluate_benchmark_quality(image_name_to_tags)

        self.assertIn("per_image", result)
        self.assertIn("aggregate", result)
        self.assertIn(filename, result["per_image"])

        pi = result["per_image"][filename]
        self.assertAlmostEqual(pi["precision"], 1.0)
        self.assertAlmostEqual(pi["recall"], 1.0)
        self.assertAlmostEqual(pi["f1"], 1.0)
        self.assertTrue(pi["subject_found"])

    def test_all_eight_benchmark_images_in_gt(self):
        """All 8 benchmark filenames must be defined in BENCHMARK_GROUND_TRUTH."""
        expected = {
            "01_airplane.png", "02_automobile.png", "03_bird.png", "04_cat.png",
            "05_deer.png", "06_dog.png", "07_frog.png", "08_ship.png",
        }
        self.assertEqual(set(BENCHMARK_GROUND_TRUTH.keys()), expected)
        self.assertEqual(set(BENCHMARK_CLASS_NAMES.keys()), expected)

    def test_non_benchmark_images_silently_skipped(self):
        """
        Images not in BENCHMARK_GROUND_TRUTH are silently skipped.
        Returns empty per_image dict and None aggregates.
        """
        result = evaluate_benchmark_quality({
            "user_upload_01.jpg": ["cat", "dog"],
            "my_photo.png": ["tree", "sky"],
        })
        self.assertEqual(result["per_image"], {})
        agg = result["aggregate"]
        self.assertIsNone(agg["mean_precision"])
        self.assertIsNone(agg["mean_recall"])
        self.assertIsNone(agg["mean_f1"])
        self.assertIsNone(agg["subject_recall_rate"])
        self.assertEqual(agg["n_images_with_gt"], 0)
        self.assertIn("limitation_note", agg)
        self.assertIn("No ground-truth", agg["limitation_note"])

    def test_empty_input(self):
        """Empty dict returns empty result with appropriate limitation note."""
        result = evaluate_benchmark_quality({})
        self.assertEqual(result["per_image"], {})
        self.assertEqual(result["aggregate"]["n_images_with_gt"], 0)

    def test_disjoint_predictions_all_benchmarks(self):
        """
        Completely wrong predictions for all benchmark images.
        Recall = 0 → subject_found = False for all; f1 = 0 for all.
        """
        image_name_to_tags = {k: ["wrong_tag_xyz"] for k in BENCHMARK_GROUND_TRUTH}
        result = evaluate_benchmark_quality(image_name_to_tags)
        agg = result["aggregate"]
        self.assertAlmostEqual(agg["mean_recall"], 0.0)
        self.assertAlmostEqual(agg["mean_f1"], 0.0)
        self.assertAlmostEqual(agg["subject_recall_rate"], 0.0)
        self.assertEqual(agg["n_images_with_gt"], 8)
        for pi in result["per_image"].values():
            self.assertFalse(pi["subject_found"])

    def test_aggregate_values_bounded(self):
        """Aggregate metrics must lie in [0.0, 1.0] (or 0–100 for subject_recall_rate)."""
        image_name_to_tags = {k: ["cat", "animal", "jet", "ship"] for k in BENCHMARK_GROUND_TRUTH}
        result = evaluate_benchmark_quality(image_name_to_tags)
        agg = result["aggregate"]
        for key in ("mean_precision", "mean_recall", "mean_f1"):
            val = agg[key]
            self.assertIsNotNone(val)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)
        self.assertGreaterEqual(agg["subject_recall_rate"], 0.0)
        self.assertLessEqual(agg["subject_recall_rate"], 100.0)

    def test_limitation_note_present_for_benchmark(self):
        """When GT images are found, the limitation note must be present."""
        image_name_to_tags = {"04_cat.png": ["cat"]}
        result = evaluate_benchmark_quality(image_name_to_tags)
        agg = result["aggregate"]
        self.assertIn("limitation_note", agg)
        note = agg["limitation_note"]
        self.assertIn("8-image", note)
        self.assertIn("indicative", note)

    def test_per_image_contains_expected_keys(self):
        """Per-image dict must contain all required keys."""
        image_name_to_tags = {"01_airplane.png": ["airplane", "jet"]}
        result = evaluate_benchmark_quality(image_name_to_tags)
        pi = result["per_image"]["01_airplane.png"]
        for key in ("class_name", "precision", "recall", "f1",
                    "predicted_tags", "ground_truth_tags",
                    "matched_tags", "subject_found"):
            self.assertIn(key, pi, msg=f"Missing key: {key}")

    def test_matched_tags_subset_of_gt_and_pred(self):
        """matched_tags must be a subset of both GT and predictions."""
        filename = "06_dog.png"
        pred = ["dog", "pet", "unrelated_tag"]
        result = evaluate_benchmark_quality({filename: pred})
        pi = result["per_image"][filename]
        matched = set(pi["matched_tags"])
        gt_set = set(pi["ground_truth_tags"])
        pred_set = set(pred)
        self.assertTrue(matched.issubset(gt_set), f"{matched} not subset of GT {gt_set}")
        self.assertTrue(matched.issubset(pred_set), f"{matched} not subset of pred {pred_set}")

    def test_mixed_benchmark_and_non_benchmark(self):
        """Benchmark images evaluated; non-benchmark images silently skipped."""
        image_name_to_tags = {
            "04_cat.png": ["cat", "animal"],       # benchmark → included
            "my_custom_photo.jpg": ["cat", "dog"],  # not benchmark → skipped
        }
        result = evaluate_benchmark_quality(image_name_to_tags)
        self.assertIn("04_cat.png", result["per_image"])
        self.assertNotIn("my_custom_photo.jpg", result["per_image"])
        self.assertEqual(result["aggregate"]["n_images_with_gt"], 1)


if __name__ == "__main__":
    unittest.main()
