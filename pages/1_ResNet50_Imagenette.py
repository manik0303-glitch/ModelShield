"""ResNet-50 + Imagenette evaluation page for ModelShield.

Run from the repository root with: streamlit run app.py
Then open this page from Streamlit's built-in page navigation.
"""
import io
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50

from attacks.ensemble import evaluate_attack_ensemble

DATA_ROOT = Path("data") / "imagenette2-160"
ARCHIVE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
# Imagenette folder names are WordNet IDs; ImageNet-1K indices below are exact.
WNID_TO_IMAGENET = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482,
    "n03000684": 491, "n03028079": 497, "n03394916": 566,
    "n03417042": 569, "n03425413": 571, "n03445777": 574,
    "n03888257": 701,
}


class PixelSpaceResNet(nn.Module):
    """Keep attack images in [0,1]; normalize inside the differentiable model."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


@st.cache_resource(show_spinner="Downloading/loading pretrained ResNet-50 weights...")
def get_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = PixelSpaceResNet(resnet50(weights=weights)).to(device).eval()
    return model, device, weights.meta["categories"]


def download_imagenette():
    """Download the small 160px Imagenette archive with safe extraction."""
    if (DATA_ROOT / "val").is_dir():
        return
    data_dir = DATA_ROOT.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "imagenette2-160.tgz"
    if not archive.exists():
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
    with tarfile.open(archive, "r:gz") as tf:
        root = data_dir.resolve()
        for member in tf.getmembers():
            target = (data_dir / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Unsafe path in dataset archive")
            if not (member.isfile() or member.isdir()):
                raise ValueError("Unexpected archive member type")
        tf.extractall(data_dir, filter="data")
    if not (DATA_ROOT / "val").is_dir():
        raise RuntimeError("Imagenette validation folder not found after extraction")


def get_dataset():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(DATA_ROOT / "val", transform=transform)
    class_ids = [WNID_TO_IMAGENET[wnid] for wnid in ds.classes]
    return ds, class_ids


st.title("ResNet-50 · Imagenette robustness evaluation")
st.caption("ImageNet-1K pretrained ResNet-50; Imagenette validation subset; 224 × 224 model inputs.")
st.info("This is a separate evaluation page. Existing CIFAR-10 and Red-Team Mission Lab workflows are unchanged.")
if not (DATA_ROOT / "val").is_dir():
    st.write("Imagenette is not downloaded yet (the 160px version is used to reduce download size).")
    if st.button("Download Imagenette", type="primary"):
        try:
            with st.spinner("Downloading and extracting dataset..."):
                download_imagenette()
            st.success("Dataset ready. You can now run evaluation.")
        except Exception as exc:
            st.error(f"Dataset download failed: {exc}")

if (DATA_ROOT / "val").is_dir():
    n = st.selectbox("Validation images", [8, 16, 32, 64], index=0)
    eps_pixels = st.slider("L-infinity epsilon (pixel levels / 255)", 1, 16, 4)
    steps = st.slider("PGD iterations", 1, 20, 5)
    if st.button("Evaluate clean, FGSM and PGD", type="primary"):
        try:
            with st.spinner("Evaluating ResNet-50; CPU may be slow..."):
                model, device, categories = get_resnet()
                ds, class_ids = get_dataset()
                # Deterministic, approximately balanced subset across the ten classes.
                by_class = {i: [] for i in range(len(ds.classes))}
                for idx, (_, class_idx) in enumerate(ds.samples):
                    by_class[class_idx].append(idx)
                selected = []
                for offset in range(max(len(v) for v in by_class.values())):
                    for class_idx in sorted(by_class):
                        if offset < len(by_class[class_idx]):
                            selected.append(by_class[class_idx][offset])
                        if len(selected) >= n:
                            break
                    if len(selected) >= n:
                        break
                loader = DataLoader(Subset(ds, selected), batch_size=1, shuffle=False)
                epsilon = eps_pixels / 255.0
                rows = []
                preview = None
                clean_count = fgsm_count = pgd_count = fooled_count = eligible = 0
                for batch_images, local_labels in loader:
                    images = batch_images.to(device)
                    labels = torch.tensor([class_ids[int(local_labels.item())]], device=device)
                    result = evaluate_attack_ensemble(
                        model, images, labels, epsilon=epsilon,
                        alpha=min(2 / 255, epsilon), pgd_steps=steps,
                    )
                    clean_ok = bool(result["clean_correct"][0].item())
                    fgsm_ok = bool(result["fgsm_correct"][0].item())
                    pgd_ok = bool(result["pgd_correct"][0].item())
                    clean_count += clean_ok
                    fgsm_count += fgsm_ok
                    pgd_count += pgd_ok
                    eligible += clean_ok
                    fooled_count += clean_ok and (not fgsm_ok or not pgd_ok)
                    rows.append({
                        "true_label": categories[int(labels.item())],
                        "clean_prediction": categories[int(result["clean_predictions"][0].item())],
                        "fgsm_prediction": categories[int(result["fgsm_predictions"][0].item())],
                        "pgd_prediction": categories[int(result["pgd_predictions"][0].item())],
                    })
                    if preview is None:
                        preview = (
                            images[0].detach().cpu(),
                            result["fgsm_images"][0].detach().cpu(),
                            result["pgd_images"][0].detach().cpu(),
                        )
            st.session_state["resnet_eval"] = {
                "rows": rows, "preview": preview, "count": len(rows),
                "clean": clean_count, "fgsm": fgsm_count, "pgd": pgd_count,
                "eligible": eligible, "fooled": fooled_count,
                "epsilon": epsilon, "steps": steps,
            }
        except Exception as exc:
            st.error(f"ResNet-50 evaluation failed: {exc}")

    if "resnet_eval" in st.session_state:
        r = st.session_state["resnet_eval"]
        total = r["count"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clean accuracy", f'{100*r["clean"]/total:.1f}%')
        c2.metric("FGSM accuracy", f'{100*r["fgsm"]/total:.1f}%')
        c3.metric("PGD accuracy", f'{100*r["pgd"]/total:.1f}%')
        c4.metric("Worst-case ASR", f'{100*r["fooled"]/r["eligible"]:.1f}%' if r["eligible"] else "N/A")
        st.caption(f'Worst-case ASR denominator: {r["eligible"]} initially correct images. Epsilon: {r["epsilon"]:.6f}; PGD steps: {r["steps"]}.')
        st.subheader("Example images")
        for col, image, name in zip(st.columns(3), r["preview"], ["Clean", "FGSM", "PGD"]):
            with col:
                st.image(image.permute(1, 2, 0).clamp(0, 1).numpy(), caption=name, use_column_width=True)
        df = pd.DataFrame(r["rows"])
        st.dataframe(df, use_container_width=True)
        st.download_button("Download predictions CSV", df.to_csv(index=False), "resnet50_imagenette_results.csv", "text/csv")
        st.caption("These results are for a small, selected Imagenette validation subset, not the full ImageNet benchmark.")
