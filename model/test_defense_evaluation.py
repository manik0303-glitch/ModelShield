import torch

from model.load_model import load_model
from robustbench.data import load_cifar10

from evaluation.defense_evaluation import evaluate_defense


def main():

    model, device = load_model()

    images, labels = load_cifar10(n_examples=8)

    images = images.to(device)
    labels = labels.to(device)

    results = evaluate_defense(
        model,
        images,
        labels,
        epsilon=8 / 255,
        pgd_steps=3,
        quantization_levels=16,
    )

    for defense_name, metrics in results.items():

        print(f"\n{defense_name}")

        for metric_name, value in metrics.items():

            print(f"{metric_name}: {value:.2f}%")

    print("\nDefense evaluation completed.")


if __name__ == "__main__":
    main()