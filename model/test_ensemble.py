from model.load_model import load_model
from robustbench.data import load_cifar10
from attacks.ensemble import evaluate_attack_ensemble

def main():
 print("Loading RobustBench model...")
model, device = load_model()


print("Loading CIFAR-10 images...")
images, labels = load_cifar10(n_examples=32)

images = images.to(device)
labels = labels.to(device)

print("\nRunning the attack ensemble...")
print("This may take a little while on your CPU.")

results = evaluate_attack_ensemble(
    model,
    images,
    labels,
    epsilon=8 / 255,
    alpha=2 / 255,
    pgd_steps=10,
)

print("\n========== ENSEMBLE RESULTS ==========")
print(f"Images evaluated: {results['total_images']}")
print(f"Clean accuracy: {results['clean_accuracy']:.2f}%")
print(f"FGSM accuracy: {results['fgsm_accuracy']:.2f}%")
print(f"PGD accuracy: {results['pgd_accuracy']:.2f}%")

print(
    "FGSM attack success rate: "
    f"{results['fgsm_attack_success_rate']:.2f}%"
)
print(
    "PGD attack success rate: "
    f"{results['pgd_attack_success_rate']:.2f}%"
)
print(
    "Worst-case ensemble attack success rate: "
    f"{results['worst_case_attack_success_rate']:.2f}%"
)

print(
    f"Initially correct images: "
    f"{results['initially_correct_count']}"
)
print(
    f"Images fooled by either attack: "
    f"{results['worst_case_fooled_count']}"
)
print("=======================================")
 

if __name__ == "__main__":
 main()
