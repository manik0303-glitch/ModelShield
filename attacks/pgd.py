import torch
import torch.nn.functional as F

def pgd_attack(
model,
images,
labels,
epsilon=8 / 255,
alpha=2 / 255,
steps=10,
random_start=True,
):

 images = images.detach()
 labels = labels.detach()

# Keep the original images for the perturbation constraint.
 original_images = images.clone().detach()

# Optionally start from a random perturbation.
 if random_start:
    adversarial_images = original_images + torch.empty_like(
        original_images
    ).uniform_(-epsilon, epsilon)

    adversarial_images = torch.clamp(
        adversarial_images, 0.0, 1.0
    )
 else:
    adversarial_images = original_images.clone()

# Iteratively increase the classification loss.
 for _ in range(steps):
    adversarial_images = (
        adversarial_images.detach().requires_grad_(True)
    )

    model.zero_grad(set_to_none=True)

    outputs = model(adversarial_images)
    loss = F.cross_entropy(outputs, labels)

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

