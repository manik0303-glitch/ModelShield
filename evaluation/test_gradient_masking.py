import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.gradient_masking import (
    compute_gradient_health,
    evaluate_loss_landscape_ray,
    evaluate_pgd_restarts,
    evaluate_attack_strength_consistency,
    evaluate_fgsm_vs_pgd,
    evaluate_random_search_baseline,
    analyze_gradient_masking,
)


class DummyLinearModel(nn.Module):
    """Simple differentiable linear model for CIFAR-like images (3, 32, 32) -> 10 classes."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        flat = x.view(x.size(0), -1)
        return self.fc(flat)


class DummyZeroGradModel(nn.Module):
    """Model that detaches input, producing zero gradients (classic gradient masking)."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        flat = x.detach().view(x.size(0), -1)
        return self.fc(flat)


class DummyNonFiniteGradModel(nn.Module):
    """Model that introduces NaN into the loss graph."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        # Taking sqrt of negative number connected to x produces NaN gradients
        nan_val = (x - 100.0).sqrt().view(x.size(0), -1)
        return self.fc(nan_val)


def _assert_json_serializable(val):
    if isinstance(val, dict):
        for k, v in val.items():
            assert isinstance(k, str), f"Dictionary key must be string, got {type(k)}"
            _assert_json_serializable(v)
    elif isinstance(val, list):
        for item in val:
            _assert_json_serializable(item)
    elif isinstance(val, tuple):
        for item in val:
            _assert_json_serializable(item)
    else:
        assert isinstance(val, (int, float, str, bool, type(None))), (
            f"Expected primitive JSON-serializable type, got {type(val)}: {val}"
        )
        assert not isinstance(val, torch.Tensor), "PyTorch Tensor found in result!"


def test_gradient_health_normal():
    model = DummyLinearModel()
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])

    res = compute_gradient_health(model, images, labels)
    assert res["status"] == "OK"
    assert res["non_finite"] is False
    assert res["median_l2_norm"] is not None
    assert res["median_l2_norm"] > 0.0


def test_gradient_health_zero_grad():
    model = DummyZeroGradModel()
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])

    res = compute_gradient_health(model, images, labels)
    assert res["status"] == "SUSPICIOUS"
    assert res["median_l2_norm"] == 0.0
    assert res["median_near_zero_fraction"] == 1.0


def test_gradient_health_non_finite():
    model = DummyNonFiniteGradModel()
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])

    res = compute_gradient_health(model, images, labels)
    assert res["status"] == "ERROR"
    assert res["non_finite"] is True


def test_loss_landscape_ray():
    model = DummyLinearModel()
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    mask = torch.tensor([True, True, True, True])

    res = evaluate_loss_landscape_ray(model, images, labels, mask, epsilon=8/255)
    assert res["status"] in ("OK", "SUSPICIOUS")
    assert len(res["losses"]) == 3
    assert res["slope"] is not None


def test_pgd_restarts():
    model = DummyLinearModel()
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    mask = torch.tensor([True, True, True, True])

    res = evaluate_pgd_restarts(
        model=model,
        images=images,
        labels=labels,
        clean_correct_mask=mask,
        restarts=2,
        baseline_pgd_asr=50.0,
    )
    assert res["status"] in ("OK", "SUSPICIOUS")
    assert len(res["restart_success_rates"]) == 2
    assert res["restart_success_rates"][0] == 50.0


def test_fgsm_vs_pgd_discrepancy():
    # PGD >= FGSM (Normal)
    ok_res = evaluate_fgsm_vs_pgd(fgsm_asr=30.0, pgd_asr=50.0)
    assert ok_res["status"] == "OK"

    # FGSM significantly outperforms PGD (Suspicious)
    susp_res = evaluate_fgsm_vs_pgd(fgsm_asr=60.0, pgd_asr=30.0)
    assert susp_res["status"] == "SUSPICIOUS"


def test_random_search_baseline():
    model = DummyLinearModel()
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    mask = torch.tensor([True, True, True, True])

    res = evaluate_random_search_baseline(
        model=model,
        images=images,
        labels=labels,
        clean_correct_mask=mask,
        epsilon=8/255,
        pgd_asr=50.0,
        num_samples=5,
    )
    assert res["status"] in ("OK", "SUSPICIOUS")
    assert "random_search_success_rate" in res


def test_analyze_gradient_masking_zero_clean_correct():
    model = DummyLinearModel()
    images = torch.rand(4, 3, 32, 32)
    # Ensure no label matches the actual prediction
    with torch.no_grad():
        preds = model(images).argmax(dim=1)
    wrong_labels = (preds + 1) % 10

    res = analyze_gradient_masking(
        model=model,
        images=images,
        labels=wrong_labels,
    )
    assert res["status"] == "INCONCLUSIVE"
    assert res["metadata"]["clean_correct_count"] == 0
    _assert_json_serializable(res)


def test_analyze_gradient_masking_masked_model():
    model = DummyZeroGradModel()
    images = torch.rand(8, 3, 32, 32)
    with torch.no_grad():
        preds = model(images).argmax(dim=1)
    # Give it matching labels so clean accuracy is 100%
    labels = preds.clone()

    res = analyze_gradient_masking(
        model=model,
        images=images,
        labels=labels,
        pgd_steps=5,
        pgd_restarts=1,
    )
    assert res["status"] == "WARNING"
    assert len(res["warning_reasons"]) > 0
    _assert_json_serializable(res)


def test_analyze_gradient_masking_output_serializability():
    model = DummyLinearModel()
    images = torch.rand(8, 3, 32, 32)
    with torch.no_grad():
        preds = model(images).argmax(dim=1)
    labels = preds.clone()

    res = analyze_gradient_masking(
        model=model,
        images=images,
        labels=labels,
        pgd_steps=5,
        pgd_restarts=2,
    )
    assert res["status"] in ("WARNING", "NO_STRONG_SIGNAL")
    _assert_json_serializable(res)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
