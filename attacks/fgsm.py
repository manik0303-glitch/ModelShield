import torch
from attacks.loss_utils import compute_attack_loss


def fgsm_attack(model, images, labels, epsilon=8 / 255, multi_label=False):
    """
    Fast Gradient Sign Method (FGSM) attack.

    Supports both single-label (CIFAR-10) and multi-label (RAM++) pipelines.

    Args:
        model: Target PyTorch model.
        images: Input images tensor in range [0, 1].
        labels: Ground-truth target labels.
        epsilon: Perturbation bound.
        multi_label: If True, uses binary cross-entropy with logits; else cross_entropy.

    Returns:
        Perturbed images tensor in range [0, 1].
    """
    images = images.detach()
    labels = labels.detach()

    # Create a copy that can track gradients.
    images_adv = images.clone().detach().requires_grad_(True)

    # Clear any existing model gradients.
    model.zero_grad(set_to_none=True)

    # Calculate classification loss.
    outputs = model(images_adv)
    loss = compute_attack_loss(outputs, labels, multi_label=multi_label)

    # Calculate gradients with respect to the input image.
    loss.backward()

    # Perturb pixels in the direction that increases the loss.
    if images_adv.grad is not None:
        perturbation = epsilon * images_adv.grad.sign()
        adversarial_images = images_adv + perturbation
    else:
        adversarial_images = images_adv

    # Keep pixel values valid.
    adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)

    return adversarial_images.detach()
