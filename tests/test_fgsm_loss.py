import pytest
import torch
import torch.nn.functional as F

from attacks.fgsm import fgsm_attack
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
def test_default_attack_preserves_cross_entropy(labels):
    model = make_model()
    images = torch.tensor([[0.01, 0.4, 0.99], [0.7, 0.3, 0.5]])
    epsilon = 0.1
    inputs = images.clone().requires_grad_(True)
    gradient = torch.autograd.grad(F.cross_entropy(model(inputs), labels), inputs)[0]
    expected = (images + epsilon * gradient.sign()).clamp(0, 1)

    actual = fgsm_attack(model, images, labels, epsilon=epsilon)

    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_multilabel_attack_uses_binary_cross_entropy():
    model = make_model()
    images = torch.tensor([[0.01, 0.4, 0.99], [0.7, 0.3, 0.5]])
    labels = torch.tensor([[1, 0], [1, 1]])
    epsilon = 0.1
    inputs = images.clone().requires_grad_(True)
    gradient = torch.autograd.grad(
        F.binary_cross_entropy_with_logits(model(inputs), labels.float()), inputs
    )[0]
    expected = (images + epsilon * gradient.sign()).clamp(0, 1)

    actual = fgsm_attack(model, images, labels, epsilon=epsilon, multi_label=True)

    torch.testing.assert_close(actual, expected)
    assert (actual - images).abs().max() <= epsilon + 1e-6


@pytest.mark.parametrize("frozen", [False, True])
def test_disconnected_input_gradient_raises(frozen):
    class DisconnectedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[0.0, 1.0]]))

        def forward(self, images):
            return self.logits.expand(images.shape[0], -1)

    model = DisconnectedModel()
    if frozen:
        model.requires_grad_(False)
    with pytest.raises(InputGradientUnavailableError, match="gradient"):
        fgsm_attack(model, torch.tensor([[0.5, 0.5, 0.5]]), torch.tensor([1]))
