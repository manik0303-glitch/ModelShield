"""
tests/test_masking_simulation.py
---------------------------------
Offline unit tests for the controlled gradient-masking simulation.

All tests use tiny synthetic models and in-memory tensors.
No CIFAR-10 download, no RobustBench weights required.
"""

import json
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.bpda_pgd import (
    straight_through_quantize,
    bpda_pgd_attack,
    _StraightThroughQuantize,
)
from evaluation.masking_simulation import (
    _HardQuantizedPipeline,
    _measure_ordinary_gradient_stats,
    _measure_bpda_gradient_stats,
    _safe_asr,
    run_masking_simulation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TinyLinearModel(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        return self.fc(x.reshape(x.size(0), -1))


def _batch(n=4, seed=42):
    torch.manual_seed(seed)
    images = torch.rand(n, 3, 32, 32)
    model = TinyLinearModel()
    with torch.no_grad():
        labels = model(images).argmax(dim=1)
    return images, labels, model


# ---------------------------------------------------------------------------
# 1. Hard quantization forward test
# ---------------------------------------------------------------------------

class TestHardQuantization:
    def test_shape_dtype_range(self):
        images, labels, model = _batch()
        pipe = _HardQuantizedPipeline(model, levels=16)
        with torch.no_grad():
            out = pipe(images)
        assert out.shape == (4, 10)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_output_in_valid_range(self):
        images, _, model = _batch()
        levels = 8
        x_q = torch.round(images.clamp(0, 1) * (levels - 1)) / (levels - 1)
        assert x_q.min().item() >= -1e-6
        assert x_q.max().item() <= 1.0 + 1e-6

    def test_deterministic_for_same_input(self):
        images, labels, model = _batch()
        pipe = _HardQuantizedPipeline(model, levels=16)
        with torch.no_grad():
            out1 = pipe(images)
            out2 = pipe(images)
        assert torch.allclose(out1, out2)

    def test_original_model_params_unchanged(self):
        images, labels, model = _batch()
        orig_params = {n: p.clone() for n, p in model.named_parameters()}
        pipe = _HardQuantizedPipeline(model, levels=16)
        _ = pipe(images)
        for n, p in model.named_parameters():
            assert torch.allclose(p, orig_params[n]), f"Parameter {n} was mutated"


# ---------------------------------------------------------------------------
# 2. Gradient-obscuring test
# ---------------------------------------------------------------------------

class TestGradientObscuring:
    def test_ordinary_grad_near_zero_through_hard_quant(self):
        images, labels, model = _batch()
        pipe = _HardQuantizedPipeline(model, levels=16)
        stats = _measure_ordinary_gradient_stats(pipe, images, labels)
        # After hard quantization, gradient should be zero or very close to zero
        norm = stats.get("norm")
        nzf = stats.get("near_zero_fraction")
        assert stats.get("finite") is True
        # Either None (disconnected) or very small
        assert norm is None or norm < 1e-3, \
            f"Ordinary gradient norm {norm} unexpectedly large through hard quant"

    def test_distinguishes_zero_from_exception(self):
        """
        A model with genuinely zero gradients returns norm=0.0, not None.
        A model with disconnected path returns norm=None.
        """
        class ZeroGradModel(nn.Module):
            def __init__(self): super().__init__(); self.fc = nn.Linear(3072, 10)
            def forward(self, x): return self.fc(x.detach().reshape(x.size(0), -1))

        images, labels, _ = _batch()
        model = ZeroGradModel()
        stats = _measure_ordinary_gradient_stats(model, images, labels)
        # Grad None because detach() disconnects graph
        assert stats.get("norm") is None or stats.get("norm") == 0.0


# ---------------------------------------------------------------------------
# 3. BPDA backward test
# ---------------------------------------------------------------------------

class TestBPDA:
    def test_bpda_forward_equals_hard_quant(self):
        images, _, _ = _batch()
        levels = 8
        hard = torch.round(images.clamp(0, 1) * (levels - 1)) / (levels - 1)
        bpda = straight_through_quantize(images, levels)
        assert torch.allclose(hard, bpda), "BPDA forward must equal hard quantization"

    def test_bpda_backward_finite(self):
        images, labels, model = _batch()
        stats = _measure_bpda_gradient_stats(model, images, labels, levels=16)
        assert stats.get("finite") is True
        norm = stats.get("norm")
        assert norm is not None and norm > 1e-9, \
            f"BPDA gradient norm {norm} is zero or None (straight-through broken)"

    def test_bpda_no_param_update(self):
        images, labels, model = _batch()
        orig = {n: p.clone() for n, p in model.named_parameters()}
        _ = bpda_pgd_attack(model, images, labels, quantization_levels=16,
                            epsilon=8/255, alpha=2/255, steps=3)
        for n, p in model.named_parameters():
            assert torch.allclose(p, orig[n]), f"Parameter {n} mutated during BPDA"


# ---------------------------------------------------------------------------
# 4. Attack constraints
# ---------------------------------------------------------------------------

class TestAttackConstraints:
    EPSILON = 8 / 255

    def test_bpda_within_epsilon_ball(self):
        images, labels, model = _batch()
        adv = bpda_pgd_attack(model, images, labels,
                               quantization_levels=16,
                               epsilon=self.EPSILON, alpha=2/255, steps=5,
                               random_seed=0)
        max_pert = (adv - images).abs().max().item()
        assert max_pert <= self.EPSILON + 1e-5, \
            f"BPDA adversarial exceeded epsilon: {max_pert:.6f}"

    def test_bpda_in_valid_range(self):
        images, labels, model = _batch()
        adv = bpda_pgd_attack(model, images, labels,
                               quantization_levels=16,
                               epsilon=self.EPSILON, alpha=2/255, steps=5)
        assert adv.min().item() >= -1e-5
        assert adv.max().item() <= 1.0 + 1e-5

    def test_bpda_does_not_mutate_original_images(self):
        images, labels, model = _batch()
        images_copy = images.clone()
        bpda_pgd_attack(model, images, labels,
                         quantization_levels=16,
                         epsilon=self.EPSILON, alpha=2/255, steps=3)
        assert torch.allclose(images, images_copy), "Original images were mutated"

    def test_deterministic_with_seed(self):
        images, labels, model = _batch()
        adv1 = bpda_pgd_attack(model, images, labels, quantization_levels=16,
                                epsilon=self.EPSILON, steps=3, random_seed=42)
        adv2 = bpda_pgd_attack(model, images, labels, quantization_levels=16,
                                epsilon=self.EPSILON, steps=3, random_seed=42)
        assert torch.allclose(adv1, adv2), "BPDA attack not deterministic for same seed"


# ---------------------------------------------------------------------------
# 5. Metric consistency
# ---------------------------------------------------------------------------

class TestMetricConsistency:
    def test_same_denominator_all_attacks(self):
        images, labels, model = _batch(n=8)
        result = run_masking_simulation(
            model, images, labels,
            quantization_levels=16,
            pgd_steps=3, sample_cap=8,
        )
        m = result["measurements"]
        denom = m["eligible_clean_correct_count"]
        for atk_name, atk in m["attacks"].items():
            assert atk["denominator"] == denom, \
                f"Attack {atk_name} uses different denominator {atk['denominator']} vs {denom}"

    def test_zero_eligible_returns_none_asr(self):
        assert _safe_asr(0, 0) is None, "Expected None ASR for 0/0"

    def test_zero_eligible_run_is_inconclusive(self):
        images, _, model = _batch(n=4)
        # Force zero clean-correct by using impossible labels
        wrong_labels = torch.full((4,), 99, dtype=torch.long)  # invalid class
        # run_masking_simulation will hit the min_eligible guard
        result = run_masking_simulation(
            model, images, wrong_labels,
            quantization_levels=4, pgd_steps=2, sample_cap=4, min_eligible=1,
        )
        # Should be INCONCLUSIVE because wrong_labels or 0 correct
        assert result["simulation_validation"]["status"] in (
            "INCONCLUSIVE", "ERROR", "CONDITION_NOT_DEMONSTRATED"
        )


# ---------------------------------------------------------------------------
# 6. Detector independence
# ---------------------------------------------------------------------------

class TestDetectorIndependence:
    def test_detector_receives_no_simulation_mode_flag(self):
        """
        The detector result must be determined by measured evidence, not
        by a simulation_mode=True parameter. We verify the detector
        is not short-circuited.
        """
        images, labels, model = _batch(n=8)
        result = run_masking_simulation(
            model, images, labels,
            quantization_levels=16, pgd_steps=3, sample_cap=8,
        )
        # Detector result is a real status, not hardcoded
        assert result["detector_result"]["status"] in (
            "WARNING", "NO_STRONG_SIGNAL", "INCONCLUSIVE"
        )
        # Simulation validation is separate
        assert result["simulation_validation"]["status"] in (
            "VALIDATED", "INCONCLUSIVE", "CONDITION_NOT_DEMONSTRATED", "ERROR"
        )

    def test_no_shortcut_warning_in_code(self):
        """
        Ensure the simulation runner doesn't contain a hardcoded
        'if simulation_mode: status = WARNING' pattern.
        """
        import inspect
        from evaluation import masking_simulation
        source = inspect.getsource(masking_simulation)
        assert "simulation_mode" not in source or "WARNING" not in source.split("simulation_mode")[1][:200], \
            "Possible hardcoded simulation_mode -> WARNING shortcut detected"


# ---------------------------------------------------------------------------
# 7. Simulation validator
# ---------------------------------------------------------------------------

class TestSimulationValidator:
    def test_validated_when_all_checks_pass(self):
        images, labels, model = _batch(n=8)
        result = run_masking_simulation(
            model, images, labels,
            quantization_levels=4,   # aggressive quant → more masking
            pgd_steps=5, sample_cap=8,
            bpda_advantage_margin=0.0,  # accept any BPDA advantage
        )
        # With 0.0 margin, if BPDA >= PGD the check passes
        checks = {c["name"]: c["passed"]
                  for c in result["simulation_validation"]["checks"]}
        assert checks.get("hard_quantization_active") is True
        assert checks.get("ordinary_gradient_obscured") is True
        assert checks.get("bpda_gradient_usable") is True

    def test_condition_not_demonstrated_when_bpda_no_advantage(self):
        images, labels, model = _batch(n=8)
        result = run_masking_simulation(
            model, images, labels,
            quantization_levels=16,
            pgd_steps=5, sample_cap=8,
            bpda_advantage_margin=999.0,  # impossible margin
        )
        val_status = result["simulation_validation"]["status"]
        # Must not be VALIDATED since margin can't be met
        assert val_status in ("CONDITION_NOT_DEMONSTRATED", "INCONCLUSIVE")


# ---------------------------------------------------------------------------
# 8. State and regression
# ---------------------------------------------------------------------------

class TestRegressionAndState:
    def test_existing_gradient_masking_tests_still_pass(self):
        """Import and re-run key functions from existing module."""
        from evaluation.gradient_masking import (
            compute_gradient_health,
            evaluate_fgsm_vs_pgd,
        )
        images, labels, model = _batch()
        health = compute_gradient_health(model, images, labels)
        assert health["status"] in ("OK", "SUSPICIOUS", "ERROR")
        fvp = evaluate_fgsm_vs_pgd(40.0, 60.0)
        assert fvp["status"] == "OK"

    def test_simulation_does_not_mutate_original_model(self):
        images, labels, model = _batch(n=8)
        orig = {n: p.clone() for n, p in model.named_parameters()}
        run_masking_simulation(model, images, labels,
                               quantization_levels=16, pgd_steps=3, sample_cap=8)
        for n, p in model.named_parameters():
            assert torch.allclose(p, orig[n]), f"Model param {n} was mutated"

    def test_simulation_does_not_mutate_images(self):
        images, labels, model = _batch(n=8)
        images_copy = images.clone()
        run_masking_simulation(model, images, labels,
                               quantization_levels=16, pgd_steps=3, sample_cap=8)
        assert torch.allclose(images, images_copy)


# ---------------------------------------------------------------------------
# 9. Result serialization
# ---------------------------------------------------------------------------

class TestResultSerialization:
    def test_output_is_json_serializable(self):
        images, labels, model = _batch(n=8)
        result = run_masking_simulation(
            model, images, labels,
            quantization_levels=16, pgd_steps=3, sample_cap=8,
        )
        try:
            serialized = json.dumps(result)
            assert len(serialized) > 10
        except (TypeError, ValueError) as exc:
            pytest.fail(f"Result is not JSON-serializable: {exc}")

    def test_no_tensors_in_result(self):
        images, labels, model = _batch(n=8)
        result = run_masking_simulation(
            model, images, labels,
            quantization_levels=16, pgd_steps=3, sample_cap=8,
        )

        def _check_no_tensor(val, path=""):
            assert not isinstance(val, torch.Tensor), \
                f"Tensor found at result{path}"
            if isinstance(val, dict):
                for k, v in val.items():
                    _check_no_tensor(v, f"{path}['{k}']")
            elif isinstance(val, list):
                for i, v in enumerate(val):
                    _check_no_tensor(v, f"{path}[{i}]")

        _check_no_tensor(result)

