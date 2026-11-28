import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class TaggedImageDataset(Dataset):
    """Dataset for RAM++ image tagging using JSON annotations."""

    def __init__(
        self,
        annotation_path: str,
        image_dir: str,
        tag_list: List[str],
        transform=None,
    ):
        self.image_dir = Path(image_dir).resolve()
        self.transform = transform
        self.tag_list = [tag.strip() for tag in tag_list]
        self.tag_to_idx = {tag: index for index, tag in enumerate(self.tag_list)}

        annotation_file = Path(annotation_path)
        if not annotation_file.is_file():
            raise FileNotFoundError(
                f"RAM++ annotation file not found: {annotation_file}"
            )

        with annotation_file.open("r", encoding="utf-8") as file:
            self.annotations = json.load(file)
        if not isinstance(self.annotations, dict):
            raise ValueError("RAM++ annotations must be a JSON object.")
        if not self.annotations:
            raise ValueError("RAM++ annotation file contains no image annotations.")

        for filename, tags in self.annotations.items():
            if not isinstance(filename, str) or not filename:
                raise ValueError("Annotation image names must be non-empty strings.")
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) for tag in tags
            ):
                raise ValueError(
                    f"Tags for {filename!r} must be a list of strings."
                )

        self.image_filenames = list(self.annotations)

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        filename = self.image_filenames[idx]
        image_path = (self.image_dir / filename).resolve()
        try:
            image_path.relative_to(self.image_dir)
        except ValueError as exc:
            raise ValueError(
                f"Annotated image path for {filename!r} is outside the image directory."
            ) from exc

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Annotated RAM++ image not found: {image_path}"
            )

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            from torchvision.transforms.functional import to_tensor

            image_tensor = to_tensor(image.resize((384, 384)))

        tags = self.annotations[filename]
        unknown_tags = [
            tag.strip() for tag in tags if tag.strip() not in self.tag_to_idx
        ]
        if unknown_tags:
            raise ValueError(
                f"Unknown tags for {filename!r}: {', '.join(unknown_tags)}"
            )

        label_tensor = torch.zeros(len(self.tag_list), dtype=torch.float32)
        for tag in tags:
            label_tensor[self.tag_to_idx[tag.strip()]] = 1.0

        return image_tensor, label_tensor


def load_dataset(
    annotation_path: str = "data/annotations.json",
    image_dir: str = "data/images",
    tag_list: Optional[List[str]] = None,
    transform=None,
    batch_size: int = 4,
    test_size: int = 100,
) -> DataLoader:
    """Load a RAM++ multi-label dataset using the supplied tag vocabulary."""
    if tag_list is None:
        raise ValueError("tag_list is required for RAM++ dataset loading.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if test_size <= 0:
        raise ValueError("test_size must be greater than zero.")

    dataset = TaggedImageDataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        tag_list=tag_list,
        transform=transform,
    )
    sample_count = min(test_size, len(dataset))
    if sample_count < len(dataset):
        dataset = Subset(dataset, range(sample_count))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def load_cifar10_dataset(
    batch_size: int = 16,
    train_size: int = 2000,
    test_size: int = 500,
) -> Tuple[DataLoader, DataLoader]:
    """Load the legacy CIFAR-10 train and test subsets."""
    transform = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_subset = Subset(
        train_dataset,
        range(min(train_size, len(train_dataset))),
    )
    test_subset = Subset(
        test_dataset,
        range(min(test_size, len(test_dataset))),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader
