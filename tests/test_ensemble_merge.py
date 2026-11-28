import pytest
import torch

import attacks.ensemble as ensemble
from attacks.fgsm import fgsm_attack
from attacks.loss_utils import InputGradientUnavailableError
from attacks.pgd import pgd_attack


def make_model():
    model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -0.5, 0.25], [-0.5, 0.75, -1.0]]))
    return model


def test_default_compares_labels_without_casting(monkeypatch):
    monkeypatch.setattr(ensemble, "fgsm_attack", lambda model, images, labels, **kwargs: images)
    monkeypatch.setattr(ensemble, "pgd_attack", lambda model, images, labels, **kwargs: images)
    images = torch.tensor([[0.7, 0.3, 0.5], [0.1, 0.9, 0.1]])
    labels = torch.tensor([0.2, 1.0])

    result = ensemble.evaluate_attack_ensemble(make_model(), images, labels)

    assert result["clean_predictions"].tolist() == [0, 1]
    assert result["clean_correct"].tolist() == [False, True]
    assert result["clean_accuracy"] == 50.0


@pytest.mark.parametrize("threshold", [0.5, torch.tensor([0.45, 0.55])], ids=["scalar", "per-tag"])
def test_multilabel_ensemble_uses_exact_tag_matching(threshold):
    model = make_model()
    images = torch.tensor([[0.207, 0.630, 0.365], [0.851, 0.855, 0.551]])
    with torch.no_grad():
        labels = (model(images).sigmoid() > threshold).long()
        labels[1, 0] = 1 - labels[1, 0]

    fgsm = fgsm_attack(model, images, labels, epsilon=0.1, multi_label=True)
    torch.manual_seed(17)
    pgd = pgd_attack(
        model, images, labels, epsilon=0.1, alpha=0.025, steps=3, multi_label=True
    )
    with torch.no_grad():
        fgsm_predictions = (model(fgsm).sigmoid() > threshold).long()
        pgd_predictions = (model(pgd).sigmoid() > threshold).long()
    clean_correct = torch.tensor([True, False])
    fgsm_correct = (fgsm_predictions == labels).all(dim=1)
    pgd_correct = (pgd_predictions == labels).all(dim=1)
    fgsm_fooled = clean_correct & ~fgsm_correct
    pgd_fooled = clean_correct & ~pgd_correct

    torch.manual_seed(17)
    result = ensemble.evaluate_attack_ensemble(
        model, images, labels, epsilon=0.1, alpha=0.025, pgd_steps=3,
        multi_label=True, threshold=threshold,
    )

    torch.testing.assert_close(result["fgsm_images"], fgsm, rtol=0, atol=0)
    torch.testing.assert_close(result["pgd_images"], pgd, rtol=0, atol=0)
    torch.testing.assert_close(result["fgsm_predictions"], fgsm_predictions)
    torch.testing.assert_close(result["pgd_predictions"], pgd_predictions)
    torch.testing.assert_close(result["clean_correct"], clean_correct)
    assert result["fgsm_accuracy"] == 100 * fgsm_correct.float().mean().item()
    assert result["pgd_accuracy"] == 100 * pgd_correct.float().mean().item()
    assert result["fgsm_attack_success_rate"] == 100 * fgsm_fooled.sum().item()
    assert result["pgd_attack_success_rate"] == 100 * pgd_fooled.sum().item()
    assert result["worst_case_attack_success_rate"] == 100 * (fgsm_fooled | pgd_fooled).sum().item()
    assert result["worst_case_attack_success_rate"] == 100.0


def test_multilabel_attack_success_uses_only_clean_correct_samples(monkeypatch):
    images = torch.tensor([[2.0, -2.0], [-2.0, 2.0], [-2.0, -2.0]])
    labels = torch.tensor([[1, 0], [0, 1], [1, 1]])
    fgsm = torch.tensor([[2.0, -2.0], [2.0, 2.0], [2.0, 2.0]])
    pgd = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]])
    monkeypatch.setattr(ensemble, "fgsm_attack", lambda model, images, labels, **kwargs: fgsm)
    monkeypatch.setattr(ensemble, "pgd_attack", lambda model, images, labels, **kwargs: pgd)

    result = ensemble.evaluate_attack_ensemble(
        torch.nn.Identity(), images, labels, multi_label=True, threshold=0.5
    )

    assert result["initially_correct_count"] == 2
    assert result["fgsm_attack_success_rate"] == 50.0
    assert result["pgd_attack_success_rate"] == 50.0
    assert result["worst_case_attack_success_rate"] == 100.0


def test_unavailable_gradient_is_not_reported_as_success():
    class DisconnectedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[0.0, 1.0]]))

        def forward(self, images):
            return self.logits.expand(images.shape[0], -1)

    with pytest.raises(InputGradientUnavailableError, match="gradient"):
        ensemble.evaluate_attack_ensemble(
            DisconnectedModel(), torch.tensor([[0.5, 0.5, 0.5]]), torch.tensor([1])
        )
