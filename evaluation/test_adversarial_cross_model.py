"""
Adversarial Cross-Model Comparison Validation (Item 3 — Final Audit).

Verifies that the RAM++ Regression Gate correctly detects robustness differences
between baseline and candidate snapshots WHEN evaluate_adversarial=True.

DESIGN:
  - Uses mock inference so no GPU / checkpoint is needed.
  - Identical mock → all Δ metrics ≈ 0, gate PASS.
  - 75% degraded candidate mock → lower Jaccard/retention for candidate, gate FAIL.

Run from project root:
    python -m evaluation.adversarial_cross_model_validation
"""
from __future__ import annotations

import sys
import types
import unittest
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Mock RAM inference helpers
# ---------------------------------------------------------------------------

CLEAN_TAGS = "cat | animal | pet | kitten | tabby"
ADV_TAGS_ROBUST = "cat | animal | pet | kitten"           # loses 1 of 5 (80 % retention)
ADV_TAGS_DEGRADED = "cat"                                  # loses 4 of 5 (20 % retention)


def _make_mock_model() -> torch.nn.Module:
    m = MagicMock(spec=torch.nn.Module)
    m.to = MagicMock(return_value=m)
    m.eval = MagicMock(return_value=m)
    m.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    return m


def _build_inference_module(clean_response: str, adv_response: str) -> types.ModuleType:
    """Build a fake ``ram.inference`` module that alternates clean/adv responses."""
    call_count = [0]

    def inference_ram(tensor, model):
        call_count[0] += 1
        # Odd calls → clean, even calls → adversarial
        if call_count[0] % 2 == 1:
            return clean_response, ""
        else:
            return adv_response, ""

    inference_mod = types.ModuleType("ram.inference")
    inference_mod.inference_ram = inference_ram
    return inference_mod


def _build_ram_package(inference_mod: types.ModuleType) -> types.ModuleType:
    ram_mod = types.ModuleType("ram")
    ram_mod.inference = inference_mod
    return ram_mod


# ---------------------------------------------------------------------------
# Shared benchmark images (tiny 384×384 PIL images)
# ---------------------------------------------------------------------------

def _make_benchmark_images(n: int = 4) -> List[Tuple[str, Image.Image]]:
    return [
        (f"0{i+1}_cat.png", Image.new("RGB", (384, 384), color=(i * 30, 100, 200)))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _run_comparison(
    baseline_inference_mod: types.ModuleType,
    candidate_inference_mod: types.ModuleType,
    benchmark_images: List[Tuple[str, Image.Image]],
    label: str,
) -> dict:
    """
    Runs evaluate_ram_model_live independently for baseline and candidate,
    then compares their metrics directly (mimicking the regression gate comparison).
    Returns a summary dict with per-metric deltas.
    """
    from evaluation.ram_eval import evaluate_ram_model_live

    baseline_model = _make_mock_model()
    candidate_model = _make_mock_model()

    with patch.dict(sys.modules, {
        "ram": _build_ram_package(baseline_inference_mod),
        "ram.inference": baseline_inference_mod,
    }):
        base_snap, base_raw = evaluate_ram_model_live(
            model=baseline_model,
            preprocess=None,
            model_name="Baseline",
            dataset_images=benchmark_images,
            dataset_name="Benchmark Suite",
            evaluate_adversarial=True,
            device=torch.device("cpu"),
        )

    # Swap to candidate inference module
    with patch.dict(sys.modules, {
        "ram": _build_ram_package(candidate_inference_mod),
        "ram.inference": candidate_inference_mod,
    }):
        cand_snap, cand_raw = evaluate_ram_model_live(
            model=candidate_model,
            preprocess=None,
            model_name="Candidate",
            dataset_images=benchmark_images,
            dataset_name="Benchmark Suite",
            evaluate_adversarial=True,
            device=torch.device("cpu"),
        )

    # Compute deltas for the key robustness metrics
    base_m = base_snap.metrics
    cand_m = cand_snap.metrics
    deltas = {
        k: round(cand_m.get(k, 0.0) - base_m.get(k, 0.0), 2)
        for k in ("mean_jaccard_similarity", "mean_tag_retention_rate",
                   "tag_agreement_rate", "avg_tags_detected",
                   "tags_lost_per_image", "new_tags_per_image")
    }

    print(f"\n── {label} ──")
    print(f"  Baseline  Jaccard={base_m.get('mean_jaccard_similarity'):.1f}%  "
          f"Retention={base_m.get('mean_tag_retention_rate'):.1f}%  "
          f"Agreement={base_m.get('tag_agreement_rate'):.1f}%  "
          f"AvgTags={base_m.get('avg_tags_detected'):.1f}")
    print(f"  Candidate Jaccard={cand_m.get('mean_jaccard_similarity'):.1f}%  "
          f"Retention={cand_m.get('mean_tag_retention_rate'):.1f}%  "
          f"Agreement={cand_m.get('tag_agreement_rate'):.1f}%  "
          f"AvgTags={cand_m.get('avg_tags_detected'):.1f}")
    print(f"  Δ metrics: {deltas}")
    return {
        "baseline_metrics": base_m,
        "candidate_metrics": cand_m,
        "deltas": deltas,
        "base_raw": base_raw,
        "cand_raw": cand_raw,
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestAdversarialCrossModelComparison(unittest.TestCase):
    """
    Item 3 — Adversarial Cross-Model Comparison.

    With evaluate_adversarial=True:
      - Identical mock  → all robustness Δ ≈ 0.
      - Degraded candidate → Jaccard/retention Δ clearly negative.
    """

    def setUp(self):
        self.benchmark = _make_benchmark_images(n=4)

    def test_identical_models_adversarial_all_deltas_near_zero(self):
        """
        SCENARIO 1: Baseline and candidate use identical mock inference.
        Expected: all metric deltas ≈ 0 (within ±5 percentage points).
        """
        # Both models produce the same adv response (robust)
        base_inf = _build_inference_module(CLEAN_TAGS, ADV_TAGS_ROBUST)
        cand_inf = _build_inference_module(CLEAN_TAGS, ADV_TAGS_ROBUST)

        result = _run_comparison(base_inf, cand_inf, self.benchmark,
                                 "Scenario 1: Identical models (adversarial=True)")
        deltas = result["deltas"]

        # Jaccard and retention deltas must be near zero
        self.assertAlmostEqual(deltas["mean_jaccard_similarity"], 0.0, delta=5.0,
                               msg="Identical models should have ~0 Jaccard delta")
        self.assertAlmostEqual(deltas["mean_tag_retention_rate"], 0.0, delta=5.0,
                               msg="Identical models should have ~0 retention delta")
        self.assertAlmostEqual(deltas["tag_agreement_rate"], 0.0, delta=5.0,
                               msg="Identical models should have ~0 agreement delta")
        print("  ✅ Scenario 1 PASS: Identical models → ~0 Δ for all robustness metrics")

    def test_degraded_candidate_adversarial_metrics_diverge(self):
        """
        SCENARIO 2: Candidate has severely degraded adversarial response.
        Expected: candidate Jaccard < baseline Jaccard; delta clearly negative.
        """
        base_inf = _build_inference_module(CLEAN_TAGS, ADV_TAGS_ROBUST)
        cand_inf = _build_inference_module(CLEAN_TAGS, ADV_TAGS_DEGRADED)

        result = _run_comparison(base_inf, cand_inf, self.benchmark,
                                 "Scenario 2: Degraded candidate (adversarial=True)")
        deltas = result["deltas"]
        base_m = result["baseline_metrics"]
        cand_m = result["candidate_metrics"]

        # Candidate must have lower Jaccard than baseline
        self.assertLess(cand_m["mean_jaccard_similarity"],
                        base_m["mean_jaccard_similarity"],
                        msg="Degraded candidate must have lower Jaccard than baseline")

        # Delta must be clearly negative (at least -10 pp)
        self.assertLess(deltas["mean_jaccard_similarity"], -10.0,
                        msg="Jaccard delta must be clearly negative (< -10 pp) for 75% degraded candidate")
        self.assertLess(deltas["mean_tag_retention_rate"], -10.0,
                        msg="Retention delta must be clearly negative (< -10 pp) for 75% degraded candidate")

        print("  ✅ Scenario 2 PASS: Degraded candidate → clearly negative Δ for Jaccard and retention")

    def test_adv_disabled_all_metrics_equal(self):
        """
        SANITY CHECK: With evaluate_adversarial=False, robustness metrics both default
        to 100%; deltas are all 0 regardless of adversarial response quality.
        """
        from evaluation.ram_eval import evaluate_ram_model_live

        base_inf = _build_inference_module(CLEAN_TAGS, ADV_TAGS_ROBUST)
        cand_inf = _build_inference_module(CLEAN_TAGS, ADV_TAGS_DEGRADED)

        base_model = _make_mock_model()
        cand_model = _make_mock_model()

        with patch.dict(sys.modules, {
            "ram": _build_ram_package(base_inf),
            "ram.inference": base_inf,
        }):
            base_snap, _ = evaluate_ram_model_live(
                model=base_model, preprocess=None, model_name="Baseline",
                dataset_images=self.benchmark, dataset_name="Benchmark Suite",
                evaluate_adversarial=False, device=torch.device("cpu"),
            )

        with patch.dict(sys.modules, {
            "ram": _build_ram_package(cand_inf),
            "ram.inference": cand_inf,
        }):
            cand_snap, _ = evaluate_ram_model_live(
                model=cand_model, preprocess=None, model_name="Candidate",
                dataset_images=self.benchmark, dataset_name="Benchmark Suite",
                evaluate_adversarial=False, device=torch.device("cpu"),
            )

        for metric in ("mean_jaccard_similarity", "mean_tag_retention_rate", "tag_agreement_rate"):
            self.assertAlmostEqual(base_snap.metrics[metric], 100.0, delta=0.1,
                                   msg=f"adversarial=False: {metric} must default to 100")
            self.assertAlmostEqual(cand_snap.metrics[metric], 100.0, delta=0.1,
                                   msg=f"adversarial=False: {metric} must default to 100 (even degraded)")

        print("  ✅ Sanity check PASS: adversarial=False → both models report 100% robustness metrics")


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("ModelShield — Adversarial Cross-Model Comparison (Audit Item 3)")
    print("=" * 70)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAdversarialCrossModelComparison)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
