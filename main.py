from model.load_model import load_ram_plus_model


def main():
    model, device, tag_list, _, _ = load_ram_plus_model()

    print("Model loaded successfully!")
    print("Model:", model.__class__.__name__)
    print("Device:", device)
    print("Tag vocabulary size:", len(tag_list))


if __name__ == "__main__":
    main()
