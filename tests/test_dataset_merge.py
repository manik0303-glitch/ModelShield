import json

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

import model.dataset as dataset_module


class FakeCIFAR10(Dataset):
    def __init__(self, root, train, download, transform):
        self.train = train
        self.transform = transform

    def __len__(self):
        return 6

    def __getitem__(self, index):
        image = Image.new("RGB", (32, 32), color=(index, index, index))
        return self.transform(image), index % 10


def test_load_dataset_is_ram_plus_and_keeps_cifar_loader_separate(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps({"sample.png": ["dog"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tag_list"):
        dataset_module.load_dataset(
            annotation_path=str(annotation_path),
            image_dir=str(tmp_path),
        )

    loader = dataset_module.load_dataset(
        annotation_path=str(annotation_path),
        image_dir=str(tmp_path),
        tag_list=["dog", "cat"],
        transform=lambda image: torch.zeros(3, 8, 8),
        batch_size=1,
    )
    images, labels = next(iter(loader))

    assert images.shape == (1, 3, 8, 8)
    assert labels.tolist() == [[1.0, 0.0]]


def test_tagged_dataset_reports_missing_annotations_and_images(tmp_path):
    missing_annotations = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="annotation"):
        dataset_module.TaggedImageDataset(
            annotation_path=str(missing_annotations),
            image_dir=str(tmp_path),
            tag_list=["dog"],
        )

    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps({"missing.png": ["dog"]}),
        encoding="utf-8",
    )
    dataset = dataset_module.TaggedImageDataset(
        annotation_path=str(annotation_path),
        image_dir=str(tmp_path),
        tag_list=["dog"],
    )

    with pytest.raises(FileNotFoundError, match="missing.png"):
        dataset[0]


def test_tagged_dataset_rejects_unknown_labels_instead_of_dropping_them(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(image_path)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps({"sample.png": ["unlisted-tag"]}),
        encoding="utf-8",
    )
    dataset = dataset_module.TaggedImageDataset(
        annotation_path=str(annotation_path),
        image_dir=str(tmp_path),
        tag_list=["dog"],
    )

    with pytest.raises(ValueError, match="Unknown tags.*unlisted-tag"):
        dataset[0]


def test_tagged_dataset_rejects_image_paths_outside_image_directory(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    outside_image = tmp_path / "outside.png"
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(outside_image)

    for filename in ("../outside.png", str(outside_image)):
        annotation_path = tmp_path / "annotations.json"
        annotation_path.write_text(
            json.dumps({filename: ["dog"]}),
            encoding="utf-8",
        )
        dataset = dataset_module.TaggedImageDataset(
            annotation_path=str(annotation_path),
            image_dir=str(image_dir),
            tag_list=["dog"],
        )

        with pytest.raises(ValueError, match="outside the image directory"):
            dataset[0]


def test_tagged_dataset_rejects_symlink_to_outside_image_directory(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    outside_image = tmp_path / "outside.png"
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(outside_image)
    image_link = image_dir / "linked.png"

    try:
        image_link.symlink_to(outside_image)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not available in this environment.")

    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps({"linked.png": ["dog"]}),
        encoding="utf-8",
    )
    dataset = dataset_module.TaggedImageDataset(
        annotation_path=str(annotation_path),
        image_dir=str(image_dir),
        tag_list=["dog"],
    )

    with pytest.raises(ValueError, match="outside the image directory"):
        dataset[0]


def test_cifar10_dataset_helper_preserves_two_loader_contract(monkeypatch):
    monkeypatch.setattr(dataset_module, "CIFAR10", FakeCIFAR10)

    train_loader, test_loader = dataset_module.load_cifar10_dataset(
        batch_size=2,
        train_size=3,
        test_size=4,
    )

    assert len(train_loader.dataset) == 3
    assert len(test_loader.dataset) == 4
    assert next(iter(train_loader))[0].shape == (2, 3, 224, 224)
