"""Gradient-masking evaluation for labeled photos and the seven-class ResNet-50."""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import functional as TF

from evaluation.gradient_masking import analyze_gradient_masking


DEFAULT_CHECKPOINT = Path(os.environ.get(
    "MODELSHIELD_OBJECT_CHECKPOINT", "C:/projects/ModelShield/models/resnet50_objects.pt"
))


def load_objects(checkpoint=DEFAULT_CHECKPOINT):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = saved["classes"]
    if len(classes) != 7 or set(classes) != {"cat", "dog", "flower", "phone", "truck", "umbrella", "watch"}:
        raise ValueError("Select the seven-class ModelShield checkpoint.")
    weights = ResNet50_Weights.IMAGENET1K_V2
    backbone = resnet50(weights=weights)
    backbone.fc = nn.Linear(backbone.fc.in_features, len(classes))
    backbone.fc.load_state_dict(saved["head"])
    preset = weights.transforms()
    from torchvision.transforms import Normalize
    model = nn.Sequential(Normalize(preset.mean, preset.std), backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.eval().requires_grad_(False).to(device), classes, preset


def image_pixels(source, preset):
    with Image.open(source) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        resized = TF.resize(rgb, preset.resize_size, preset.interpolation, antialias=preset.antialias)
        return TF.to_tensor(TF.center_crop(resized, preset.crop_size))


def diagnose(checkpoint, examples):
    if not 1 <= len(examples) <= 8:
        raise ValueError("Select one to eight labeled photos per run.")
    model, classes, preset = load_objects(checkpoint)
    device = next(model.parameters()).device
    images = torch.stack([image_pixels(source, preset) for source, _ in examples]).to(device)
    labels = torch.tensor([classes.index(label) for _, label in examples], device=device)
    report = analyze_gradient_masking(model, images, labels)
    report["metadata"].update(model="ResNet-50 (seven objects)", classes=classes,
                              input_space="224x224 RGB pixels in [0,1]")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample", nargs=2, action="append", required=True, metavar=("PHOTO", "LABEL"))
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(json.dumps(diagnose(args.checkpoint, args.sample), indent=2))
