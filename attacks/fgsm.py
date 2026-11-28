import torch

from attacks.loss_utils import InputGradientUnavailableError, compute_attack_loss


def fgsm_attack(model, images, labels, epsilon=8 / 255, multi_label=False):
    """Generate an FGSM perturbation in pixel space for single- or multi-label models."""
    images = images.detach()
    labels = labels.detach()
    images_adv = images.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(images_adv)
    if multi_label:
        loss = compute_attack_loss(outputs, labels, multi_label=True)
    else:
        loss = torch.nn.functional.cross_entropy(outputs, labels)
    if not loss.requires_grad:
        raise InputGradientUnavailableError("Cannot generate FGSM attack: input gradient is unavailable.")
    loss.backward()

    if images_adv.grad is None:
        raise InputGradientUnavailableError("Cannot generate FGSM attack: input gradient is unavailable.")

    adversarial_images = images_adv + epsilon * images_adv.grad.sign()
    return torch.clamp(adversarial_images, 0.0, 1.0).detach()
