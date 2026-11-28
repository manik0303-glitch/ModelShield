"""Print RAM++ tags for a local photo."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache"))
import torch
from PIL import Image, ImageOps
from ram import get_transform
from ram.models import ram_plus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / "pretrained"
    with Image.open(args.image) as photo:
        pixels = get_transform(image_size=384)(ImageOps.exif_transpose(photo).convert("RGB"))
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = ram_plus(pretrained=str(folder / "ram_plus_swin_large_14m.pth"),
                     text_encoder_type=str(folder), image_size=384, vit="swin_l").eval()
    with torch.inference_mode():
        tags, _, scores = model.generate_tag(pixels.unsqueeze(0), return_scores=True)
    print("Image tags:", tags[0])
    selected = set(tags[0].split(" | "))
    print("Tag scores:")
    for index in scores[0].argsort(descending=True).tolist():
        label = model.tag_list[index]
        if label in selected:
            print(f"  {label}: {scores[0, index].item():.2%}")
