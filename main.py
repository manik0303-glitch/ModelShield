from model.load_model import load_model


def main():
    model, device, tag_list, class_threshold, transform = load_model()

    print("RAM++ Model loaded successfully!")
    print("Model:", model.__class__.__name__)
    print("Device:", device)
    print("Number of tags recognized:", len(tag_list))


if __name__ == "__main__":
    main()
