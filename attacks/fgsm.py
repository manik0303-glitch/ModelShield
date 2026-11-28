import torch

def fgsm_attack(model, images, labels, epsilon=8 / 255):


 images = images.detach()
 labels = labels.detach()

# Create a copy that can track gradients.
 images_adv = images.clone().detach().requires_grad_(True)

# Clear any existing model gradients.
 model.zero_grad(set_to_none=True)

# Calculate classification loss.
 outputs = model(images_adv)
 loss = torch.nn.functional.cross_entropy(outputs, labels)

# Calculate gradients with respect to the input image.
 loss.backward()

# Perturb pixels in the direction that increases the loss.
 perturbation = epsilon * images_adv.grad.sign()
 adversarial_images = images_adv + perturbation

# Keep pixel values valid.
 adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)

 return adversarial_images.detach()

