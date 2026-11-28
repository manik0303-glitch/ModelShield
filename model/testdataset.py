from model.dataset import load_dataset

CLASS_NAMES = [
"airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck",
]

def main():
 print("Loading CIFAR-10...")


train_loader, test_loader = load_dataset()

print("\nDataset loaded successfully!")
print("Training batches:", len(train_loader))
print("Testing batches:", len(test_loader))

images, labels = next(iter(test_loader))

print("Image tensor shape:", images.shape)
print("Labels:", labels[:8].tolist())
print("Class names:", [CLASS_NAMES[i] for i in labels[:8].tolist()])


if __name__ == "__main__":
 main()
