import pytest
import torch
import torch.nn as nn
from evaluation.prediction_utils import (
    predict_multilabel,
    is_correct_multilabel,
    is_fooled_multilabel,
)
from attacks.loss_utils import compute_attack_loss
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.bpda_pgd import bpda_pgd_attack
from attacks.ensemble import evaluate_attack_ensemble
from evaluation.gradient_masking import analyze_gradient_masking
from evaluation.masking_simulation import run_masking_simulation
from model.dataset import TaggedImageDataset, load_dataset


class SimpleMultiLabelModel(nn.Module):
    def __init__(self, num_tags=8):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_tags)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        pooled = self.pool(h).flatten(1)
        return self.fc(pooled)


def test_prediction_utils():
    # 2 samples, 4 tags
    pred_binary = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]])
    labels_exact = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])

    # Exact match: sample 0 passes, sample 1 fails
    correct_exact = is_correct_multilabel(pred_binary, labels_exact, mode="exact")
    assert correct_exact[0].item() is True
    assert correct_exact[1].item() is False

    # Fooled mask check
    adv_binary = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 1]])  # sample 0 flipped tag 2
    fooled = is_fooled_multilabel(pred_binary, adv_binary, correct_exact)
    assert fooled[0].item() is True
    assert fooled[1].item() is False  # sample 1 was not initially correct


def test_multilabel_attacks():
    torch.manual_seed(42)
    model = SimpleMultiLabelModel(num_tags=6)
    model.eval()

    images = torch.rand(3, 3, 32, 32)
    labels = torch.zeros(3, 6)
    labels[0, [0, 2]] = 1.0
    labels[1, [1, 3]] = 1.0
    labels[2, [4, 5]] = 1.0

    # FGSM
    adv_fgsm = fgsm_attack(model, images, labels, epsilon=4 / 255, multi_label=True)
    assert adv_fgsm.shape == images.shape
    assert (adv_fgsm >= 0.0).all() and (adv_fgsm <= 1.0).all()
    assert (adv_fgsm - images).abs().max() <= (4 / 255 + 1e-5)

    # PGD
    adv_pgd = pgd_attack(model, images, labels, epsilon=4 / 255, alpha=1 / 255, steps=5, multi_label=True)
    assert adv_pgd.shape == images.shape
    assert (adv_pgd >= 0.0).all() and (adv_pgd <= 1.0).all()
    assert (adv_pgd - images).abs().max() <= (4 / 255 + 1e-5)

    # BPDA PGD
    adv_bpda = bpda_pgd_attack(model, images, labels, epsilon=4 / 255, alpha=1 / 255, steps=5, multi_label=True)
    assert adv_bpda.shape == images.shape
    assert (adv_bpda >= 0.0).all() and (adv_bpda <= 1.0).all()


def test_multilabel_gradient_masking():
    torch.manual_seed(42)
    model = SimpleMultiLabelModel(num_tags=5)
    model.eval()

    images = torch.rand(4, 3, 32, 32)
    # Give ground-truth labels that match model output to ensure clean accuracy > 0
    with torch.no_grad():
        clean_logits = model(images)
        labels = (clean_logits.sigmoid() > 0.5).float()

    res = analyze_gradient_masking(
        model=model,
        images=images,
        labels=labels,
        epsilon=4 / 255,
        pgd_steps=3,
        multi_label=True,
        threshold=0.5,
    )
    assert res["status"] in ["NO_STRONG_SIGNAL", "WARNING", "INCONCLUSIVE"]
    assert "signals" in res
    assert "gradient_health" in res["signals"]


def test_multilabel_masking_simulation():
    torch.manual_seed(42)
    model = SimpleMultiLabelModel(num_tags=4)
    model.eval()

    images = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        labels = (model(images).sigmoid() > 0.5).float()

    sim_res = run_masking_simulation(
        model=model,
        images=images,
        labels=labels,
        quantization_levels=8,
        epsilon=4 / 255,
        pgd_steps=3,
        multi_label=True,
        threshold=0.5,
    )
    assert "simulation_validation" in sim_res
    assert "detector_result" in sim_res
    assert "measurements" in sim_res
