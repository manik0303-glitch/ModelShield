import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def load_dataset(batch_size=16, train_size=2000, test_size=500):

 transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

 train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

 test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Use smaller subsets for our initial CPU-based experiments.
 train_subset = Subset(
    train_dataset,
    range(min(train_size, len(train_dataset)))
)

 test_subset = Subset(
    test_dataset,
    range(min(test_size, len(test_dataset)))
)

 train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

 test_loader = DataLoader(
    test_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

 return train_loader, test_loader

