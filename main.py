from model.load_model import load_model


def main():
    model, weights, device = load_model()

    print("Model loaded successfully!")
    print("Model:", model.__class__.__name__)
    print("Device:", device)
    print("Number of classes:", len(weights.meta["categories"]))


if __name__ == "__main__":
    main()