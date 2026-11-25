import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from typing import List, Optional, Union, Dict, Any


class TaggedImageDataset(Dataset):
    """
    Dataset for multi-label image tagging tasks (RAM++).

    Reads an image directory and a JSON annotation file mapping:
    {
        "filename.jpg": ["tag1", "tag2", ...],
        ...
    }
    """

    def __init__(
        self,
        annotation_path: str,
        image_dir: str,
        tag_list: Union[List[str], Any],
        transform=None,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.tag_list = list(tag_list)
        self.tag_to_idx = {tag.strip(): i for i, tag in enumerate(self.tag_list)}

        if os.path.exists(annotation_path):
            with open(annotation_path, "r", encoding="utf-8") as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}

        self.image_filenames = list(self.annotations.keys())

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, filename)

        if not os.path.exists(img_path):
            # Fallback placeholder image if physical file doesn't exist yet
            image = Image.new("RGB", (384, 384), color=(128, 128, 128))
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            from torchvision.transforms.functional import to_tensor
            image_tensor = to_tensor(image.resize((384, 384)))

        # Build multi-hot binary label vector
        tags = self.annotations.get(filename, [])
        label_tensor = torch.zeros(len(self.tag_list), dtype=torch.float32)
        for t in tags:
            tag_clean = t.strip()
            if tag_clean in self.tag_to_idx:
                label_tensor[self.tag_to_idx[tag_clean]] = 1.0

        return image_tensor, label_tensor


def load_dataset(
    annotation_path: str = "data/annotations.json",
    image_dir: str = "data/images",
    tag_list: Optional[List[str]] = None,
    transform=None,
    batch_size: int = 4,
    test_size: int = 100,
):
    """
    Creates and returns a DataLoader for tagged multi-label dataset.
    """
    if tag_list is None:
        tag_list = [f"tag_{i}" for i in range(10)]

    dataset = TaggedImageDataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        tag_list=tag_list,
        transform=transform,
    )

    n_samples = min(test_size, len(dataset))
    if n_samples > 0 and n_samples < len(dataset):
        dataset = Subset(dataset, range(n_samples))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return dataloader
