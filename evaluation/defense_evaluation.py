import torch

from defenses.defense_methods import (
    identity_defense,
    quantization_defense,
)

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def calculate_accuracy(model, images, labels, defense):
    """Calculate model accuracy after applying a defense."""

    defended_images = defense(images)

    with torch.no_grad():
        outputs = model(defended_images)
        predictions = outputs.argmax(dim=1)

    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / len(labels)

    return accuracy, predictions


def calculate_attack_success(
    clean_predictions,
    adversarial_predictions,
    labels,
):
    """Calculate attack success among initially correct examples."""

    initially_correct = clean_predictions == labels

    fooled = (
        initially_correct
        & (adversarial_predictions != labels)
    )

    denominator = initially_correct.sum().item()

    if denominator == 0:
        return 0.0

    return 100.0 * fooled.sum().item() / denominator


def evaluate_defense(
    model,
    images,
    labels,
    epsilon=8 / 255,
    pgd_steps=10,
    quantization_levels=16,
):
    """
    Compare the original pipeline with input quantization.

    Note: The initial attack examples are generated against
    the original model. This is a preliminary evaluation,
    not a robustness certification.
    """

    model.eval()

    defenses = {
        "Original": identity_defense,
        "Quantization": lambda x: quantization_defense(
            x,
            levels=quantization_levels,
        ),
    }

    with torch.no_grad():
        clean_outputs = model(images)
        clean_predictions = clean_outputs.argmax(dim=1)

    # Use the model's clean predictions as attack reference labels.
    attack_labels = clean_predictions.detach()

    # Generate adversarial examples against the original model.
    fgsm_images = fgsm_attack(
        model,
        images,
        attack_labels,
        epsilon=epsilon,
    )

    pgd_images = pgd_attack(
        model,
        images,
        attack_labels,
        epsilon=epsilon,
        alpha=min(2 / 255, epsilon),
        steps=pgd_steps,
        random_start=True,
    )

    results = {}

    for name, defense in defenses.items():

        clean_accuracy, clean_preds = calculate_accuracy(
            model,
            images,
            labels,
            defense,
        )

        fgsm_accuracy, fgsm_preds = calculate_accuracy(
            model,
            fgsm_images,
            labels,
            defense,
        )

        pgd_accuracy, pgd_preds = calculate_accuracy(
            model,
            pgd_images,
            labels,
            defense,
        )

        fgsm_asr = calculate_attack_success(
            clean_preds,
            fgsm_preds,
            labels,
        )

        pgd_asr = calculate_attack_success(
            clean_preds,
            pgd_preds,
            labels,
        )

        results[name] = {
            "clean_accuracy": clean_accuracy,
            "fgsm_accuracy": fgsm_accuracy,
            "pgd_accuracy": pgd_accuracy,
            "fgsm_attack_success_rate": fgsm_asr,
            "pgd_attack_success_rate": pgd_asr,
        }

    return results