import torch
import torch.nn as nn

import evaluation.defense_evaluation as defense_evaluation


class FixedOutputModel(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.register_buffer("outputs", outputs)

    def forward(self, images):
        return self.outputs[: images.shape[0]]


def test_single_label_accuracy_and_attack_success_remain_unchanged():
    images = torch.zeros(2, 1)
    labels = torch.tensor([1, 0])
    model = FixedOutputModel(torch.tensor([[0.0, 2.0], [3.0, 0.0]]))

    accuracy, predictions = defense_evaluation.calculate_accuracy(
        model, images, labels, lambda x: x
    )
    attack_success = defense_evaluation.calculate_attack_success(
        predictions, torch.tensor([0, 0]), labels
    )

    assert accuracy == 100.0
    assert predictions.tolist() == [1, 0]
    assert attack_success == 50.0


def test_multi_label_accuracy_uses_thresholds_and_exact_match():
    images = torch.zeros(3, 2)
    labels = torch.tensor([[1, 0], [0, 1], [1, 1]])
    model = FixedOutputModel(
        torch.tensor([[2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]])
    )

    accuracy, predictions = defense_evaluation.calculate_accuracy(
        model,
        images,
        labels,
        lambda x: x,
        multi_label=True,
        threshold=torch.tensor([0.5, 0.5]),
    )

    assert accuracy == 100.0
    assert predictions.tolist() == labels.tolist()


def test_single_label_defense_evaluation_keeps_default_attack_calls(monkeypatch):
    images = torch.zeros(2, 1)
    labels = torch.tensor([1, 0])
    model = FixedOutputModel(torch.tensor([[0.0, 2.0], [3.0, 0.0]]))
    calls = []

    def fake_fgsm(model, attack_images, attack_labels, **kwargs):
        calls.append(("fgsm", kwargs))
        return attack_images

    def fake_pgd(model, attack_images, attack_labels, **kwargs):
        calls.append(("pgd", kwargs))
        return attack_images

    monkeypatch.setattr(defense_evaluation, "fgsm_attack", fake_fgsm)
    monkeypatch.setattr(defense_evaluation, "pgd_attack", fake_pgd)

    results = defense_evaluation.evaluate_defense(model, images, labels)

    assert calls == [
        ("fgsm", {"epsilon": 8 / 255}),
        (
            "pgd",
            {
                "epsilon": 8 / 255,
                "alpha": 2 / 255,
                "steps": 10,
                "random_start": True,
            },
        ),
    ]
    assert results["Original"] == {
        "clean_accuracy": 100.0,
        "fgsm_accuracy": 100.0,
        "pgd_accuracy": 100.0,
        "fgsm_attack_success_rate": 0.0,
        "pgd_attack_success_rate": 0.0,
    }


def test_multi_label_attack_success_counts_any_tag_change_on_clean_correct_items():
    labels = torch.tensor([[1, 0], [0, 1], [1, 1]])
    clean_predictions = labels.clone()
    adversarial_predictions = torch.tensor([[0, 0], [0, 1], [1, 0]])

    attack_success = defense_evaluation.calculate_attack_success(
        clean_predictions,
        adversarial_predictions,
        labels,
        multi_label=True,
    )

    assert attack_success == 100.0 * 2 / 3


def test_multi_label_mode_is_passed_to_attacks_and_preserves_result_shape(monkeypatch):
    images = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    labels = torch.tensor([[1, 0], [0, 1]])
    model = FixedOutputModel(images.clone())
    calls = []

    def fake_attack(model, attack_images, attack_labels, **kwargs):
        calls.append((attack_labels.clone(), kwargs))
        return attack_images

    monkeypatch.setattr(defense_evaluation, "fgsm_attack", fake_attack)
    monkeypatch.setattr(defense_evaluation, "pgd_attack", fake_attack)

    results = defense_evaluation.evaluate_defense(
        model, images, labels, multi_label=True, threshold=0.5
    )

    assert set(results) == {"Original", "Quantization"}
    assert set(results["Original"]) == {
        "clean_accuracy",
        "fgsm_accuracy",
        "pgd_accuracy",
        "fgsm_attack_success_rate",
        "pgd_attack_success_rate",
    }
    assert len(calls) == 2
    for attack_labels, kwargs in calls:
        assert attack_labels.dtype == torch.float32
        assert attack_labels.tolist() == labels.tolist()
        assert kwargs["multi_label"] is True
