import torch
import torch.nn.functional as F
from attacks.loss_utils import compute_attack_loss


def pgd_attack(
    model,
    images,
    labels,
    epsilon=8 / 255,
    alpha=2 / 255,
    steps=10,
    random_start=True,
    multi_label=False,
):
    """
    Projected Gradient Descent (PGD) attack.

    Supports both single-label and multi-label pipelines via multi_label parameter.

    Args:
        model: Target PyTorch model.
        images: Input images tensor in range [0, 1].
        labels: Ground-truth labels.
        epsilon: Maximum L-infinity perturbation budget.
        alpha: Step size per iteration.
        steps: Number of PGD iterations.
        random_start: If True, initializes with random uniform perturbation.
        multi_label: If True, uses BCE with logits loss; else cross_entropy.

    Returns:
        Adversarial images tensor in range [0, 1].
    """
    images = images.detach()
    labels = labels.detach()

    # Keep the original images for the perturbation constraint.
    original_images = images.clone().detach()

    # Optionally start from a random perturbation.
    if random_start:
        adversarial_images = original_images + torch.empty_like(
            original_images
        ).uniform_(-epsilon, epsilon)

        adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)
    else:
        adversarial_images = original_images.clone()

    # Iteratively increase the classification loss.
    for _ in range(steps):
        adversarial_images = adversarial_images.detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)

        outputs = model(adversarial_images)
        loss = compute_attack_loss(outputs, labels, multi_label=multi_label)

        gradients = torch.autograd.grad(
            loss, adversarial_images, allow_unused=True
        )[0]

        if gradients is not None:
            # Take a step in the direction that increases the loss.
            adversarial_images = (
                adversarial_images.detach()
                + alpha * gradients.sign()
            )

        # Project back into the permitted epsilon neighbourhood.
        perturbation = torch.clamp(
            adversarial_images - original_images,
            min=-epsilon,
            max=epsilon,
        )

        adversarial_images = torch.clamp(
            original_images + perturbation, 0.0, 1.0
        )

    return adversarial_images.detach()
