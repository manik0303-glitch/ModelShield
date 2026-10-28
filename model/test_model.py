import torch
from model.load_model import load_model
from robustbench.data import load_cifar10


def main():
    print("Loading RobustBench model...")

    model, device = load_model()

    print("Model loaded successfully!")
    print("Device:", device)

    print("\nLoading 32 CIFAR-10 test images...")

    images, labels = load_cifar10(n_examples=32)

    images = images.to(device)
    labels = labels.to(device)

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)

    with torch.no_grad():
        outputs = model(images)

    predictions = outputs.argmax(dim=1)

    print("\nModel compatibility verified!")
    print("Output shape:", outputs.shape)
    print("First 10 predictions:", predictions[:10].tolist())
    print("First 10 true labels:", labels[:10].tolist())

    accuracy = (
        (predictions == labels).float().mean().item() * 100
    )

    print(f"Clean accuracy on this sample: {accuracy:.2f}%")


if __name__ == "__main__":
    main()