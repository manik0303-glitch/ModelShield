import torch

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

def evaluate_attack_ensemble(
    model,
    images,
    labels,
    epsilon=8 / 255,
    alpha=2 / 255,
    pgd_steps=10,
    multi_label=False,
    threshold=0.68,
    random_start=True,
):
    """
    Evaluates clean performance and generates an attack ensemble (FGSM + PGD).

    Supports both single-label and multi-label models.
    """
    model.eval()

    # 1. Evaluate the original images.
    with torch.no_grad():
        clean_outputs = model(images)
        if multi_label:
            thresh = threshold if not isinstance(threshold, torch.Tensor) else threshold.to(clean_outputs.device)
            clean_predictions = (clean_outputs.sigmoid() > thresh).long()
            clean_correct = (clean_predictions == labels.long()).all(dim=1)
        else:
            clean_predictions = clean_outputs.argmax(dim=1)
            clean_correct = clean_predictions.eq(labels)

    # 2. Generate FGSM adversarial images.
    fgsm_images = fgsm_attack(
        model,
        images,
        labels,
        epsilon=epsilon,
        multi_label=multi_label,
    )

    # 3. Generate PGD adversarial images.
    pgd_images = pgd_attack(
        model,
        images,
        labels,
        epsilon=epsilon,
        alpha=alpha,
        steps=pgd_steps,
        random_start=random_start,
        multi_label=multi_label,
    )

    # 4. Evaluate adversarial predictions.
    with torch.no_grad():
        fgsm_outputs = model(fgsm_images)
        pgd_outputs = model(pgd_images)

        if multi_label:
            thresh = threshold if not isinstance(threshold, torch.Tensor) else threshold.to(fgsm_outputs.device)
            fgsm_predictions = (fgsm_outputs.sigmoid() > thresh).long()
            pgd_predictions = (pgd_outputs.sigmoid() > thresh).long()
            fgsm_correct = (fgsm_predictions == labels.long()).all(dim=1)
            pgd_correct = (pgd_predictions == labels.long()).all(dim=1)
        else:
            fgsm_predictions = fgsm_outputs.argmax(dim=1)
            pgd_predictions = pgd_outputs.argmax(dim=1)
            fgsm_correct = fgsm_predictions.eq(labels)
            pgd_correct = pgd_predictions.eq(labels)

    # 5. An attack succeeds only if clean was correct and attacked is incorrect.
    fgsm_fooled = clean_correct & ~fgsm_correct
    pgd_fooled = clean_correct & ~pgd_correct

    # Worst-case result: either attack fooled the model.
    worst_case_fooled = fgsm_fooled | pgd_fooled

    eligible = int(clean_correct.sum().item())
    fgsm_successes = int(fgsm_fooled.sum().item())
    pgd_successes = int(pgd_fooled.sum().item())
    worst_case_successes = int(worst_case_fooled.sum().item())

    def percentage(numerator, denominator):
        return (
            100.0 * numerator / denominator
            if denominator > 0
            else 0.0
        )

    return {
        "clean_predictions": clean_predictions,
        "fgsm_predictions": fgsm_predictions,
        "pgd_predictions": pgd_predictions,
        "clean_correct": clean_correct,
        "fgsm_correct": fgsm_correct,
        "pgd_correct": pgd_correct,
        "fgsm_images": fgsm_images,
        "pgd_images": pgd_images,
        "fgsm_accuracy": percentage(
            int(fgsm_correct.sum().item()), len(labels)
        ),
        "pgd_accuracy": percentage(
            int(pgd_correct.sum().item()), len(labels)
        ),
        "fgsm_attack_success_rate": percentage(
            fgsm_successes, eligible
        ),
        "pgd_attack_success_rate": percentage(
            pgd_successes, eligible
        ),
        "worst_case_attack_success_rate": percentage(
            worst_case_successes, eligible
        ),
        "clean_accuracy": percentage(
            eligible, len(labels)
        ),
        "initially_correct_count": eligible,
        "fgsm_fooled_count": fgsm_successes,
        "pgd_fooled_count": pgd_successes,
        "worst_case_fooled_count": worst_case_successes,
        "total_images": len(labels),
    }
