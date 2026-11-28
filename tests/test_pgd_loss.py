import pytest
import torch
import torch.nn.functional as F

from attacks.pgd import pgd_attack
from attacks.loss_utils import InputGradientUnavailableError


def make_model():
    model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -0.5, 0.25], [-0.5, 0.75, -1.0]]))
    return model


@pytest.mark.parametrize(
    "labels",
    [torch.tensor([0, 1]), torch.tensor([[0.7, 0.3], [0.2, 0.8]])],
    ids=["class-indices", "soft-labels"],
)
def test_default_attack_matches_main(labels):
    model = make_model()
    images = torch.tensor([[0.01, 0.4, 0.99], [0.7, 0.3, 0.5]])
    expected = torch.tensor([[0.0, 0.475, 0.915], [0.775, 0.225, 0.575]])

    actual = pgd_attack(model, images, labels, epsilon=0.1, alpha=0.025, steps=3, random_start=False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert not actual.requires_grad


def test_random_start_is_reproducible_and_preserves_model():
    model = make_model()
    images = torch.tensor([[0.01, 0.4, 0.99], [0.7, 0.3, 0.5]])
    labels = torch.tensor([0, 1])
    weights = model.weight.detach().clone()
    clean_logits = model(images).detach().clone()

    torch.manual_seed(17)
    first = pgd_attack(model, images, labels, epsilon=0.1, alpha=0.025, steps=3)
    torch.manual_seed(17)
    second = pgd_attack(model, images, labels, epsilon=0.1, alpha=0.025, steps=3)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert (first - images).abs().max() <= 0.1 + 1e-6
    assert (first >= 0).all() and (first <= 1).all()
    torch.testing.assert_close(model.weight, weights, rtol=0, atol=0)
    torch.testing.assert_close(model(images), clean_logits, rtol=0, atol=0)


def test_multilabel_attack_uses_binary_cross_entropy():
    model = make_model()
    images = torch.tensor([[0.01, 0.4, 0.99], [0.7, 0.3, 0.5]])
    labels = torch.tensor([[1, 0], [1, 1]])
    inputs = images.clone().requires_grad_(True)
    gradient = torch.autograd.grad(
        F.binary_cross_entropy_with_logits(model(inputs), labels.float()), inputs
    )[0]
    expected = (images + 0.025 * gradient.sign()).clamp(0, 1)

    actual = pgd_attack(
        model, images, labels, epsilon=0.1, alpha=0.025, steps=1,
        random_start=False, multi_label=True,
    )

    torch.testing.assert_close(actual, expected)
    assert (actual - images).abs().max() <= 0.1 + 1e-6


@pytest.mark.parametrize("multi_label", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_disconnected_input_gradient_raises(multi_label, frozen):
    class DisconnectedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[0.0, 1.0]]))

        def forward(self, images):
            return self.logits.expand(images.shape[0], -1)

    labels = torch.tensor([[1, 0]]) if multi_label else torch.tensor([1])
    model = DisconnectedModel()
    if frozen:
        model.requires_grad_(False)
    with pytest.raises(InputGradientUnavailableError, match="gradient"):
        pgd_attack(
            model, torch.tensor([[0.5, 0.5, 0.5]]), labels,
            steps=1, multi_label=multi_label,
        )
