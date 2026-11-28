from pathlib import Path
import io
from typing import Tuple, Any, Optional
import torch
import streamlit as st

try:
    import ram
    from ram.models import ram as ram_original, ram_plus
except ImportError:
    ram = None
    ram_original = None
    ram_plus = None

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "pretrained" / "ram_plus_swin_large_14m.pth"
DEFAULT_RAM_CHECKPOINT = Path(__file__).resolve().parent.parent / "pretrained" / "ram_swin_large_14m.pth"


@st.cache_resource
def load_ram_model(checkpoint_path: str | Path | None = None):
    """
    Loads and caches the official pretrained RAM++ model.
    Detects CUDA and falls back to CPU on out-of-memory errors.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if ram_plus is None:
        raise ImportError("RAM++ package is not installed. Please activate .venv-ram311 to run RAM++.")
    if not ckpt.is_file():
        raise FileNotFoundError(f"RAM++ checkpoint missing: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = ram_plus(
            pretrained=str(ckpt),
            vit="swin_l",
            image_size=384,
            stage="eval",
        )
        model = model.to(device).eval()
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda":
            raise
        del model
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        model = ram_plus(
            pretrained=str(ckpt),
            vit="swin_l",
            image_size=384,
            stage="eval",
        )
        model = model.to(device).eval()

    return model, ram.get_transform(image_size=384), device


@st.cache_resource
def load_ram_original_model(checkpoint_path: str | Path | None = None):
    """
    Loads and caches the official pretrained original RAM (Recognize Anything Model)
    using ``ram.models.ram()`` with Swin-Large backbone.
    Detects CUDA and falls back to CPU on out-of-memory errors.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_RAM_CHECKPOINT
    if ram_original is None:
        raise ImportError("RAM package is not installed. Please activate .venv-ram311 to run RAM.")
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Original RAM checkpoint missing: {ckpt}. "
            "Download ram_swin_large_14m.pth or upload a RAM checkpoint."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = ram_original(
            pretrained=str(ckpt),
            vit="swin_l",
            image_size=384,
            stage="eval",
        )
        model = model.to(device).eval()
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda":
            raise
        del model
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        model = ram_original(
            pretrained=str(ckpt),
            vit="swin_l",
            image_size=384,
            stage="eval",
        )
        model = model.to(device).eval()

    return model, ram.get_transform(image_size=384), device


def validate_ram_state_dict(state_dict: dict) -> Tuple[bool, str]:
    """
    Validates whether a state dictionary is structurally compatible with RAM++ (Swin-Large).
    """
    if not isinstance(state_dict, dict):
        return False, f"Checkpoint must be a dictionary of tensors, got {type(state_dict).__name__}."

    keys = list(state_dict.keys())
    if len(keys) < 20:
        return False, f"Checkpoint contains only {len(keys)} tensor keys; expected foundational Swin-L weights."

    # Look for vision backbone or tag decoder components
    has_visual = any("image_encoder" in k or "visual_encoder" in k or "patch_embed" in k or "layers" in k for k in keys)
    has_tag_or_head = any("tag_list" in k or "wordvec" in k or "head" in k or "tag_embed" in k or "norm" in k for k in keys)

    if not has_visual and not has_tag_or_head:
        return False, "Checkpoint tensors do not match RAM++ architecture (missing Swin visual encoder or tag heads)."

    return True, "Valid RAM++ compatible weights."


def load_ram_candidate_model(
    checkpoint_source: str | Path | io.BytesIO | Any,
    device: torch.device | str | None = None,
    architecture: str = "ram_plus",
):
    """
    Safely loads a custom RAM++ or original RAM candidate model from a file path
    or in-memory BytesIO buffer. Validates state dict compatibility and verifies
    forward inference.

    Args:
        checkpoint_source: Path or file-like buffer containing the checkpoint.
        device: Optional target device.
        architecture: ``"ram_plus"`` (RAM++) or ``"ram"`` (original RAM).
    """
    if isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch_norm = str(architecture or "ram_plus").strip().lower()
    if arch_norm not in ("ram_plus", "ram"):
        raise ValueError(f"Unsupported RAM architecture '{architecture}'. Expected 'ram_plus' or 'ram'.")

    if isinstance(checkpoint_source, (str, Path)):
        p = Path(checkpoint_source)
        if not p.is_file():
            raise FileNotFoundError(f"Candidate checkpoint not found at: {p}")
        raw = torch.load(str(p), map_location="cpu", weights_only=False)
    elif hasattr(checkpoint_source, "getvalue"):
        raw = torch.load(io.BytesIO(checkpoint_source.getvalue()), map_location="cpu", weights_only=False)
    elif hasattr(checkpoint_source, "read"):
        raw = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    else:
        raise TypeError(f"Unsupported checkpoint source: {type(checkpoint_source)}")

    state_dict = raw
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        state_dict = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]

    is_valid, reason = validate_ram_state_dict(state_dict)
    if not is_valid:
        raise ValueError(f"Incompatible candidate checkpoint: {reason}")

    if ram_plus is None or ram_original is None:
        raise ImportError("RAM++ package is not installed. Please activate .venv-ram311 to run RAM/RAM++.")

    # Instantiate the requested architecture (RAM++ uses ram_plus(), original RAM uses ram())
    constructor = ram_original if arch_norm == "ram" else ram_plus
    model = constructor(
        pretrained="",
        vit="swin_l",
        image_size=384,
        stage="eval",
    )
    
    # Clean prefixes if present
    cleaned_sd = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith("module."):
            clean_k = clean_k[7:]
        if clean_k.startswith("model."):
            clean_k = clean_k[6:]
        cleaned_sd[clean_k] = v

    model.load_state_dict(cleaned_sd, strict=False)
    model = model.to(target_device).eval()

    # Forward verification
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 384, 384, device=target_device)
        try:
            from ram import inference
            inference.inference_ram(dummy, model)
        except Exception as exc:
            raise RuntimeError(f"Candidate forward pass verification failed: {exc}")

    param_count = sum(p.numel() for p in model.parameters())
    arch_label = "RAM (ram)" if arch_norm == "ram" else "RAM++ (ram_plus)"
    info = {
        "total_parameters": param_count,
        "architecture": arch_label,
        "architecture_id": "ram_swin_large" if arch_norm == "ram" else "ram_plus_swin_large",
        "backbone": "Swin-Large",
        "device": str(target_device),
        "status": "Validated",
    }
    return model, ram.get_transform(image_size=384), target_device, info
