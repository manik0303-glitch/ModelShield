import torch
import torch.nn.functional as F

from model.load_model import load_model
from robustbench.data import load_cifar10
from attacks.fgsm import fgsm_attack

def main():
# Load the pretrained model.
 print("Loading model...")
model, device = load_model()

# Load a small batch of CIFAR-10 images.
print("Loading CIFAR-10 images...")
images, labels = load_cifar10(n_examples=32)

images = images.to(device)
labels = labels.to(device)

# Evaluate clean images.
model.eval()

with torch.no_grad():
    clean_outputs = model(images)
    clean_predictions = clean_outputs.argmax(dim=1)
    clean_accuracy = (
        clean_predictions == labels
    ).float().mean().item() * 100

# Generate adversarial images.
print("\nGenerating FGSM adversarial images...")
adversarial_images = fgsm_attack(
    model,
    images,
    labels,
    epsilon=8 / 255
)

# Evaluate adversarial images.
with torch.no_grad():
    adversarial_outputs = model(adversarial_images)
    adversarial_predictions = adversarial_outputs.argmax(dim=1)

    adversarial_accuracy = (
        adversarial_predictions == labels
    ).float().mean().item() * 100

# Measure attack success among initially correct predictions.
initially_correct = clean_predictions == labels
fooled = initially_correct & (adversarial_predictions != labels)

eligible = initially_correct.sum().item()
fooled_count = fooled.sum().item()

attack_success_rate = (
    100 * fooled_count / eligible if eligible > 0 else 0.0
)

# Verify perturbation constraints.
max_perturbation = (
    adversarial_images - images
).abs().max().item()

valid_range = (
    adversarial_images.min().item() >= 0.0
    and adversarial_images.max().item() <= 1.0
)

print("\n========== FGSM RESULTS ==========")
print(f"Clean accuracy: {clean_accuracy:.2f}%")
print(f"Adversarial accuracy: {adversarial_accuracy:.2f}%")
print(f"Attack success rate: {attack_success_rate:.2f}%")
print(f"Maximum pixel perturbation: {max_perturbation:.6f}")
print(f"Epsilon budget: {8 / 255:.6f}")
print(f"Pixels remain in [0, 1]: {valid_range}")
print("==================================")


if __name__ == "__main__":
 main()
