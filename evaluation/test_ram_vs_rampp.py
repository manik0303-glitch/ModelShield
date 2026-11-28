"""
Unit & Integration Tests for RAM (ram) vs RAM++ (ram_plus) Support,
Architecture Identity, Cross-Family Rejection, and Export Roundtrips.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from evaluation.ram_eval import (
    SimulatedDegradedRAMWrapper,
    evaluate_ram_model_live,
    run_live_ram_regression_pipeline,
)
from evaluation.regression_gate import (
    EvaluationMetadata,
    EvaluationSnapshot,
    GateVerdict,
    RegressionThresholds,
    TaskType,
    evaluate_regression_gate,
    export_gate_report_json,
    export_gate_report_markdown,
)


def _make_mock_model() -> torch.nn.Module:
    m = MagicMock(spec=torch.nn.Module)
    m.to = MagicMock(return_value=m)
    m.eval = MagicMock(return_value=m)
    m.parameters = MagicMock(return_value=iter([torch.zeros(1000)]))
    return m


def _make_images(n: int = 4):
    return [
        (f"0{i+1}_cat.png", Image.new("RGB", (384, 384), color=(i * 40, 120, 180)))
        for i in range(n)
    ]


class TestRamVsRamPlusSupport(unittest.TestCase):

    def test_ram_original_loader_uses_ram_constructor(self):
        """load_ram_original_model must call ram(), not ram_plus(), with vit='swin_l', image_size=384, stage='eval'."""
        import model.ram_loader as rl

        fake_model = _make_mock_model()
        mock_ram_fn = MagicMock(return_value=fake_model)
        mock_ram_plus_fn = MagicMock()
        mock_ram_pkg = MagicMock()
        mock_ram_pkg.get_transform = MagicMock(return_value=lambda img: torch.zeros(3, 384, 384))

        rl.load_ram_original_model.clear()
        with patch.object(rl, "ram_original", mock_ram_fn), \
             patch.object(rl, "ram_plus", mock_ram_plus_fn), \
             patch.object(rl, "ram", mock_ram_pkg), \
             patch("pathlib.Path.is_file", return_value=True):
            loaded_m, transform, dev = rl.load_ram_original_model("pretrained/ram_swin_large_14m.pth")

        mock_ram_fn.assert_called_once()
        mock_ram_plus_fn.assert_not_called()
        _, kwargs = mock_ram_fn.call_args
        self.assertEqual(kwargs["vit"], "swin_l")
        self.assertEqual(kwargs["image_size"], 384)
        self.assertEqual(kwargs["stage"], "eval")
        self.assertIs(loaded_m, fake_model)
        self.assertIsNotNone(transform)
        self.assertIsInstance(dev, torch.device)

    def test_ram_original_loader_missing_checkpoint_raises(self):
        import model.ram_loader as rl
        rl.load_ram_original_model.clear()
        with self.assertRaises(FileNotFoundError):
            rl.load_ram_original_model("nonexistent_dir/ram_swin_large_14m.pth")

    def test_rampp_identical_baseline_vs_candidate_passes(self):
        """RAM++ baseline vs identical RAM++ candidate must PASS with zero deltas."""
        inf_mod = types.ModuleType("ram.inference")
        inf_mod.inference_ram = lambda tensor, model: ("cat | animal | pet", "猫")
        ram_pkg = types.ModuleType("ram")
        ram_pkg.inference = inf_mod

        base_m = _make_mock_model()
        cand_m = _make_mock_model()
        imgs = _make_images(4)
        forensic_out = {}

        with patch.dict(sys.modules, {"ram": ram_pkg, "ram.inference": inf_mod}):
            b_snap, c_snap, res = run_live_ram_regression_pipeline(
                baseline_model=base_m,
                candidate_model=cand_m,
                preprocess=lambda x: torch.zeros(3, 384, 384),
                baseline_name="RAMPlus_Official",
                candidate_name="RAMPlus_Identical",
                dataset_images=imgs,
                dataset_name="RAM++ Benchmark Suite",
                baseline_architecture="ram_plus_swin_large",
                candidate_architecture="ram_plus_swin_large",
                forensic_out=forensic_out,
                device=torch.device("cpu"),
            )

        self.assertEqual(res.verdict, GateVerdict.PASS)
        self.assertEqual(b_snap.metadata.architecture, "ram_plus_swin_large")
        self.assertEqual(c_snap.metadata.architecture, "ram_plus_swin_large")
        self.assertEqual(res.metric_deltas["mean_jaccard_similarity"].delta, 0.0)
        self.assertEqual(len(forensic_out["candidate_per_image"]), 4)

    def test_rampp_baseline_vs_ram_candidate_distinct_architectures(self):
        """
        RAM++ baseline vs RAM candidate:
          - Must report distinct architectures (ram_plus_swin_large vs ram_swin_large).
          - Must not fabricate identical architecture strings.
          - Must populate forensic_out and produce valid JSON/Markdown exports.
        """
        base_m = _make_mock_model()
        cand_m = _make_mock_model()

        def _fake_inference(tensor, model):
            if model is base_m:
                return "cat | animal | pet | tabby", ""
            return "cat | animal", ""

        inf_mod = types.ModuleType("ram.inference")
        inf_mod.inference_ram = _fake_inference
        ram_pkg = types.ModuleType("ram")
        ram_pkg.inference = inf_mod

        imgs = _make_images(4)
        forensic_out = {}

        with patch.dict(sys.modules, {"ram": ram_pkg, "ram.inference": inf_mod}):
            b_snap, c_snap, res = run_live_ram_regression_pipeline(
                baseline_model=base_m,
                candidate_model=cand_m,
                preprocess=lambda x: torch.zeros(3, 384, 384),
                baseline_name="RAMPlus_SwinLarge_Official",
                candidate_name="RAM_SwinLarge_Official",
                dataset_images=imgs,
                dataset_name="RAM++ Benchmark Suite",
                baseline_architecture="ram_plus_swin_large",
                candidate_architecture="ram_swin_large",
                baseline_checkpoint_name="ram_plus_swin_large_14m.pth",
                candidate_checkpoint_name="ram_swin_large_14m.pth",
                forensic_out=forensic_out,
                device=torch.device("cpu"),
            )

        self.assertEqual(b_snap.metadata.architecture, "ram_plus_swin_large")
        self.assertEqual(c_snap.metadata.architecture, "ram_swin_large")
        self.assertNotEqual(b_snap.metadata.architecture, c_snap.metadata.architecture)
        self.assertEqual(b_snap.metadata.extra["architecture_display"], "RAM++ (ram_plus)")
        self.assertEqual(c_snap.metadata.extra["architecture_display"], "RAM (ram)")
        self.assertEqual(b_snap.metadata.extra["checkpoint_name"], "ram_plus_swin_large_14m.pth")
        self.assertEqual(c_snap.metadata.extra["checkpoint_name"], "ram_swin_large_14m.pth")

        # Exports must succeed and contain both model names
        json_str = export_gate_report_json(res)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["baseline_metadata"]["architecture"], "ram_plus_swin_large")
        self.assertEqual(parsed["candidate_metadata"]["architecture"], "ram_swin_large")

        md_str = export_gate_report_markdown(res)
        self.assertIn("RAMPlus_SwinLarge_Official", md_str)
        self.assertIn("RAM_SwinLarge_Official", md_str)
        self.assertIn("Metric Category Reference", md_str)

    def test_cross_family_rejection_ram_vs_classification(self):
        """Comparing a CIFAR-10 classification snapshot with a RAM tagging snapshot must FAIL."""
        base_snap = EvaluationSnapshot(
            metadata=EvaluationMetadata(
                model_name="Carmon2019Unlabeled",
                dataset_name="CIFAR-10",
                num_samples=100,
                epsilon=8.0 / 255.0,
                task_type=TaskType.CLASSIFICATION.value,
            ),
            metrics={"clean_accuracy": 89.0, "pgd_accuracy": 60.0},
        )
        cand_snap = EvaluationSnapshot(
            metadata=EvaluationMetadata(
                model_name="RAM_SwinLarge_Official",
                dataset_name="CIFAR-10",
                num_samples=100,
                epsilon=8.0 / 255.0,
                task_type=TaskType.RAM_TAGGING.value,
                architecture="ram_swin_large",
            ),
            metrics={
                "mean_jaccard_similarity": 85.0,
                "mean_tag_retention_rate": 85.0,
                "tag_agreement_rate": 90.0,
                "semantic_consistency": 85.0,
            },
        )
        res = evaluate_regression_gate(base_snap, cand_snap, thresholds=RegressionThresholds())
        self.assertEqual(res.verdict, GateVerdict.FAIL)
        self.assertTrue(any("Task type mismatch" in f for f in res.failures))

    def test_visuals_module_helpers_importable(self):
        """Ensure ram_gate_visuals imports cleanly and formats parameters accurately."""
        from evaluation.ram_gate_visuals import _format_params
        self.assertEqual(_format_params(174_500_000), "174.5M (174,500,000)")
        self.assertEqual(_format_params(None), "Unavailable")

    def test_simulated_degradation_demo_three_verdicts(self):
        """
        Verify the 3 demo scenarios without downloading another model:
          - Identical (0% drop) -> PASS
          - Simulated moderate degradation (25% drop) -> WARN
          - Simulated severe degradation (75% drop) -> FAIL
        """
        base_m = _make_mock_model()
        base_m.generate_tag = MagicMock(
            return_value=(["cat | animal | pet | kitten | tabby | fur | whisker | indoor"], ["猫"])
        )

        inf_mod = types.ModuleType("ram.inference")
        inf_mod.inference_ram = lambda tensor, model: (
            model.generate_tag(tensor)[0][0],
            model.generate_tag(tensor)[1][0],
        )
        ram_pkg = types.ModuleType("ram")
        ram_pkg.inference = inf_mod

        imgs = _make_images(4)
        expected = [
            (0.0, GateVerdict.PASS),
            (0.25, GateVerdict.WARN),
            (0.75, GateVerdict.FAIL),
        ]

        with patch.dict(sys.modules, {"ram": ram_pkg, "ram.inference": inf_mod}):
            for drop_frac, expected_verdict in expected:
                cand_m = (
                    base_m
                    if drop_frac == 0.0
                    else SimulatedDegradedRAMWrapper(base_m, drop_fraction=drop_frac, seed=42)
                )
                _, _, res = run_live_ram_regression_pipeline(
                    baseline_model=base_m,
                    candidate_model=cand_m,
                    preprocess=lambda x: torch.zeros(3, 384, 384),
                    baseline_name="RAMPlus_Official",
                    candidate_name=f"RAMPlus_Demo_{int(drop_frac * 100)}",
                    dataset_images=imgs,
                    dataset_name="RAM++ Benchmark Suite",
                    device=torch.device("cpu"),
                )
                self.assertEqual(
                    res.verdict,
                    expected_verdict,
                    msg=f"drop_fraction={drop_frac} expected {expected_verdict}, got {res.verdict} ({res.summary})",
                )

    def test_multi_image_gallery_summary_4_and_8_images(self):
        """
        Verify multi-image forensic gallery summary with 4 and 8 benchmark images:
          - All images represented
          - Aggregate metrics (retention, jaccard, lost, new, regressions) accurate
          - Per-image row contains corresponding predictions and real tensor perturbation stats
        """
        from evaluation.ram_gate_visuals import compute_forensic_gallery_summary

        base_m = _make_mock_model()
        cand_m = _make_mock_model()

        def _fake_inf(tensor, model):
            if model is base_m:
                return "cat | animal | pet | tabby", ""
            return "cat | animal", ""  # loses 2 of 4 tags vs baseline -> 50% vs base

        inf_mod = types.ModuleType("ram.inference")
        inf_mod.inference_ram = _fake_inf
        ram_pkg = types.ModuleType("ram")
        ram_pkg.inference = inf_mod

        for n_imgs in (4, 8):
            imgs = _make_images(n_imgs)
            forensic_out = {}
            with patch.dict(sys.modules, {"ram": ram_pkg, "ram.inference": inf_mod}):
                run_live_ram_regression_pipeline(
                    baseline_model=base_m,
                    candidate_model=cand_m,
                    preprocess=lambda x: torch.zeros(3, 384, 384),
                    dataset_images=imgs,
                    dataset_name="RAM++ Benchmark Suite",
                    evaluate_adversarial=True,
                    forensic_out=forensic_out,
                    device=torch.device("cpu"),
                )

            summary = compute_forensic_gallery_summary(forensic_out)
            self.assertIsNotNone(summary)
            self.assertEqual(summary["total_images"], n_imgs)
            self.assertEqual(len(summary["rows"]), n_imgs)
            self.assertEqual(summary["images_with_regressions"], n_imgs)
            self.assertAlmostEqual(summary["avg_retention"], 100.0)
            self.assertAlmostEqual(summary["avg_jaccard"], 100.0)
            self.assertAlmostEqual(summary["avg_lost_tags"], 2.0)
            self.assertAlmostEqual(summary["avg_new_tags"], 0.0)

            for i, row in enumerate(summary["rows"]):
                self.assertEqual(row["image_name"], imgs[i][0])
                self.assertEqual(row["clean_tags"], ["animal", "cat"])
                self.assertEqual(row["base_clean_tags"], ["animal", "cat", "pet", "tabby"])
                self.assertEqual(row["lost_vs_base"], ["pet", "tabby"])
                self.assertIsNotNone(row["perturbation_linf"])
                self.assertGreater(row["perturbation_linf"], 0.0)


if __name__ == "__main__":
    unittest.main()
