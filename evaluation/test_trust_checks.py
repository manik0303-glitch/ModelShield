import torch

from evaluation.trust_checks import (
check_perturbation_budget,
check_pixel_range,
check_prediction_outputs,
check_metric_consistency,
)

def main():
 print("Testing ModelShield trust checks...\n")


# Create sample images and valid adversarial images.
original = torch.full((4, 3, 32, 32), 0.5)
adversarial = original.clone()
adversarial[:, :, 0, 0] += 4 / 255

# Simulate model outputs for four images and ten classes.
outputs = torch.randn(4, 10)
labels = torch.tensor([0, 1, 2, 3])

checks = {
    "Perturbation budget": check_perturbation_budget(
        original, adversarial
    ),
    "Pixel range": check_pixel_range(adversarial),
    "Prediction outputs": check_prediction_outputs(
        outputs, labels
    ),
    "Metric consistency": check_metric_consistency(
        93.75, 59.38, 36.67
    ),
}

all_passed = True

for name, result in checks.items():
    status = "PASS" if result["passed"] else "FAIL"
    print(f"[{status}] {name}: {result['message']}")

    if not result["passed"]:
        all_passed = False

print("\nOverall trust check:", "PASS" if all_passed else "FAIL")


if __name__ == "__main__":
 main()
