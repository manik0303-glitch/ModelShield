import torch
import torch.nn.functional as F

from attacks.loss_utils import InputGradientUnavailableError, compute_attack_loss


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
    """Generate an L-infinity PGD perturbation in pixel space."""
    images = images.detach()
    labels = labels.detach()
    original_images = images.clone().detach()

    if random_start:
        adversarial_images = original_images + torch.empty_like(
            original_images
        ).uniform_(-epsilon, epsilon)
        adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)
    else:
        adversarial_images = original_images.clone()

    for _ in range(steps):
        adversarial_images = adversarial_images.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs = model(adversarial_images)
        if multi_label:
            loss = compute_attack_loss(outputs, labels, multi_label=True)
        else:
            loss = F.cross_entropy(outputs, labels)

        if not loss.requires_grad:
            raise InputGradientUnavailableError("Cannot generate PGD attack: input gradient is unavailable.")
        gradients = torch.autograd.grad(loss, adversarial_images, allow_unused=True)[0]
        if gradients is None:
            raise InputGradientUnavailableError("Cannot generate PGD attack: input gradient is unavailable.")
        adversarial_images = adversarial_images.detach() + alpha * gradients.sign()
        perturbation = torch.clamp(
            adversarial_images - original_images,
            min=-epsilon,
            max=epsilon,
        )
        adversarial_images = torch.clamp(
            original_images + perturbation, 0.0, 1.0
        )

    return adversarial_images.detach()
