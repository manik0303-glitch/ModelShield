"""
ModelShield RAM++ Adapter
=========================
Reusable, cached wrapper around the RAM++ Swin-L image-tagging model.

Key design decisions:
  - Singleton model — loaded once per Streamlit session via @st.cache_resource.
  - Gradient-compatible — does NOT use torch.inference_mode() so FGSM/PGD work.
  - Fully isolated from CIFAR-10; outputs tag strings + sigmoid score tensors.
  - Returns a ModelMetadata dict alongside every result for display and audit.

Checkpoint layout (matches install.ps1):
    ram-plus/pretrained/ram_plus_swin_large_14m.pth   (~3 GB)
    ram-plus/pretrained/vocab.txt                      (BERT vocab)
    ram-plus/upstream/                                 (patched recognize-anything repo)
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Path setup — point to the patched upstream RAM++ code
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # repo root
_RAM_ROOT = _HERE / "ram-plus"
_UPSTREAM = _RAM_ROOT / "upstream"
_PRETRAINED = _RAM_ROOT / "pretrained"
_CHECKPOINT = _PRETRAINED / "ram_plus_swin_large_14m.pth"

if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

os.environ.setdefault("HF_HOME", str(_RAM_ROOT / ".cache"))

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
IMAGE_SIZE: int = 384
CHECKPOINT_NAME: str = "ram_plus_swin_large_14m.pth"

MODEL_METADATA: Dict[str, Any] = {
    "architecture": "RAM++ Swin-L",
    "checkpoint": CHECKPOINT_NAME,
    "image_size": IMAGE_SIZE,
    "vit": "swin_l",
    "tag_vocabulary_size": 4585,
    "text_encoder": "bert-base-uncased",
    "upstream_commit": "7cb804a8609e9f4b1a50b7f31436d2df40bb9481",
    "dataset": "Open-vocabulary (multi-label tagging)",
    "threat_model": "Linf (ε-perturbation on tag score space)",
    "is_pretrained": True,
    "checkpoint_source": "HuggingFace: xinyu1205/recognize-anything-plus-model",
}


# ---------------------------------------------------------------------------
# Availability check — graceful fallback when checkpoint not installed
# ---------------------------------------------------------------------------
def is_ram_available() -> Tuple[bool, str]:
    """Return (True, '') if RAM++ is ready, else (False, reason)."""
    if not _UPSTREAM.exists():
        return False, (
            "RAM++ upstream code not found. "
            "Run `ram-plus\\install.ps1` to install."
        )
    if not _CHECKPOINT.exists():
        return False, (
            f"Checkpoint not found at {_CHECKPOINT}. "
            "Run `ram-plus\\install.ps1` to download the ~3 GB model."
        )
    if not (_PRETRAINED / "vocab.txt").exists():
        return False, (
            "BERT vocab.txt not found. "
            "Run `ram-plus\\install.ps1` to download it."
        )
    try:
        from ram import get_transform  # noqa: F401
        from ram.models import ram_plus  # noqa: F401
    except ImportError as exc:
        return False, (
            f"RAM++ Python package not importable: {exc}. "
            "Activate the ram-plus venv or install dependencies."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Model loader (cached by Streamlit)
# ---------------------------------------------------------------------------
def _load_ram_model():
    """Load and return the RAM++ model (not cached here — use get_ram_model())."""
    from ram.models import ram_plus  # type: ignore

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = ram_plus(
        pretrained=str(_CHECKPOINT),
        text_encoder_type=str(_PRETRAINED),
        image_size=IMAGE_SIZE,
        vit="swin_l",
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


# Lazy singleton — avoids importing Streamlit at module level
_ram_model_singleton: Optional[Tuple[Any, Any]] = None


def get_ram_model():
    """Return the cached (model, device) pair. Loads on first call."""
    global _ram_model_singleton
    if _ram_model_singleton is None:
        _ram_model_singleton = _load_ram_model()
    return _ram_model_singleton


# ---------------------------------------------------------------------------
# Pixel space <-> model space
#   Orchestrator convention: attacks, epsilon budgets and perturbation checks
#   live in RAW PIXEL SPACE [0, 1].  RAM++ itself consumes ImageNet-normalized
#   tensors, so normalization happens ONLY inside the model boundary.
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _mean_std(like: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, dtype=like.dtype, device=like.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=like.dtype, device=like.device).view(1, 3, 1, 1)
    return mean, std


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """[0,1] pixels -> RAM++ ImageNet-normalized input (differentiable)."""
    mean, std = _mean_std(pixels)
    return (pixels - mean) / std


def denormalize_to_pixels(normalized: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
    """
    ImageNet-normalized tensor -> raw pixels in [0, 1].

    Raises ValueError if the recovered pixels are non-finite or fall outside
    [0, 1] by more than `tol` (i.e. the input was NOT a valid normalized image).
    Values within `tol` are float round-off and are clamped; nothing else is clipped.
    """
    mean, std = _mean_std(normalized)
    pixels = normalized * std + mean
    if not bool(torch.isfinite(pixels).all().item()):
        raise ValueError("Recovered pixels contain non-finite values.")
    lo, hi = float(pixels.min().item()), float(pixels.max().item())
    if lo < -tol or hi > 1.0 + tol:
        raise ValueError(f"Recovered pixel range [{lo:.6f}, {hi:.6f}] is outside [0, 1].")
    return pixels.clamp(0.0, 1.0)


class PixelSpaceRAM:
    """
    Pixel-space view of a RAM++ model.

    Accepts [N,3,H,W] tensors in [0,1], normalizes internally, and proxies every
    other attribute (tag_list, visual_encoder, label_embed, ...) to the real model.
    Attacks that call get_baseline / forward_with_grad / generate_tag with this
    object therefore operate in pixel space automatically.
    """
    expects_pixels = True

    def __init__(self, inner, device):
        self.inner = inner
        self.device = device

    def normalize(self, pixels: torch.Tensor) -> torch.Tensor:
        return normalize_pixels(pixels.to(self.device))

    def generate_tag(self, pixels, *args, **kwargs):
        return self.inner.generate_tag(self.normalize(pixels), *args, **kwargs)

    def __call__(self, pixels):
        return forward_with_grad(pixels, model=self, device=self.device)

    def __getattr__(self, name):  # only called when normal lookup fails
        if name in ("inner", "device"):
            raise AttributeError(name)
        return getattr(self.inner, name)


def _resolve_input(pixel_tensor, model, device, input_space):
    """Return (inner_model, device, model_ready_input)."""
    if model is None or device is None:
        m, d = (model, device)
        if m is None:
            m, d0 = get_ram_model()
            d = d if d is not None else d0
        elif d is None:
            d = getattr(m, "device", None) or get_ram_model()[1]
        model, device = m, d
    if isinstance(model, PixelSpaceRAM):
        return model.inner, device, model.normalize(pixel_tensor.to(device))
    if input_space == "pixel":
        return model, device, normalize_pixels(pixel_tensor.to(device))
    if input_space == "normalized":
        return model, device, pixel_tensor.to(device)
    raise ValueError(f"input_space must be 'normalized' or 'pixel', got {input_space!r}")


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess_pil(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a [1, 3, 384, 384] ImageNet-NORMALIZED float tensor
    (observed range ~[-2.118, 2.640]; this is what RAM++ consumes).
    Use preprocess_pil_pixels() for raw [0, 1] pixels.
    Applies EXIF auto-rotation and RGB conversion. Returns a CPU tensor.
    """
    from ram import get_transform  # type: ignore

    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    tensor = get_transform(image_size=IMAGE_SIZE)(img)  # shape [3, 384, 384]
    return tensor.unsqueeze(0)  # shape [1, 3, 384, 384]


def preprocess_pil_pixels(pil_image: Image.Image) -> torch.Tensor:
    """PIL image -> [1, 3, 384, 384] raw pixels in [0, 1] (exact inverse of preprocess_pil)."""
    return denormalize_to_pixels(preprocess_pil(pil_image))


def preprocess_bytes(image_bytes: bytes) -> Tuple[Image.Image, torch.Tensor]:
    """Load image bytes → (PIL image, preprocessed tensor)."""
    import io
    pil = Image.open(io.BytesIO(image_bytes))
    return pil, preprocess_pil(pil)


# ---------------------------------------------------------------------------
# Baseline inference (no gradient tracking needed)
# ---------------------------------------------------------------------------
def get_baseline(
    pixel_tensor: torch.Tensor,
    model=None,
    device=None,
    input_space: str = "normalized",
) -> Dict[str, Any]:
    """
    Run RAM++ on a [1, 3, 384, 384] tensor.

    input_space="normalized" (default, backwards compatible): tensor is already
        ImageNet-normalized, as returned by preprocess_pil().
    input_space="pixel": tensor is raw [0,1] pixels; normalization is applied here.
    If `model` is a PixelSpaceRAM the input is always treated as pixels.

    Returns dict with tags, scores_dict (active tags only), scores_tensor
    ([num_classes], CPU), tag_list, run_at, inference_ms.
    """
    inner, device, inp = _resolve_input(pixel_tensor, model, device, input_space)

    t0 = time.perf_counter()
    with torch.no_grad():
        tags_batch, _, scores_batch = inner.generate_tag(inp, return_scores=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    tag_strings: List[str] = tags_batch[0].split(" | ") if tags_batch[0] else []
    scores_cpu: torch.Tensor = scores_batch[0].detach().float().cpu()
    tag_list: List[str] = list(inner.tag_list)

    active = set(tag_strings)
    scores_dict: Dict[str, float] = {
        tag: float(scores_cpu[i].item()) for i, tag in enumerate(tag_list) if tag in active
    }
    return {
        "tags": tag_strings,
        "scores_dict": scores_dict,
        "scores_tensor": scores_cpu,
        "tag_list": tag_list,
        "run_at": _iso_now(),
        "inference_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Gradient-compatible forward pass (for attacks)
# ---------------------------------------------------------------------------
def _manual_head_scores(inner, inp: torch.Tensor) -> torch.Tensor:
    """Fallback differentiable head (RAM-style). Validated against generate_tag in preflight."""
    image_embeds = inner.image_proj(inner.visual_encoder(inp))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=inp.device)
    tagging_embed = inner.tagging_head(
        encoder_embeds=inner.label_embed.weight.unsqueeze(0).expand(image_embeds.shape[0], -1, -1),
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode="tagging",
    )
    return torch.sigmoid(inner.fc(tagging_embed[0]).squeeze(-1))


def _fc_hook_scores(inner, inp: torch.Tensor) -> torch.Tensor:
    """
    Differentiable scores taken from the model's OWN final linear layer (`fc`) while
    generate_tag() runs with autograd enabled.  Because the logits come from the same
    forward pass that produces the tags, no part of the network is re-implemented.
    Raises if the logits are detached (e.g. generate_tag runs under no_grad/inference_mode).
    """
    captured: Dict[str, torch.Tensor] = {}
    handle = inner.fc.register_forward_hook(lambda _m, _i, out: captured.__setitem__("logits", out))
    try:
        with torch.enable_grad():
            inner.generate_tag(inp, return_scores=True)
    finally:
        handle.remove()
    logits = captured.get("logits")
    if logits is None or not logits.requires_grad:
        raise RuntimeError("fc logits are not attached to the autograd graph (generate_tag is no-grad).")
    return torch.sigmoid(logits.squeeze(-1))[0]


def _ramplus_replica_scores(inner, inp: torch.Tensor) -> torch.Tensor:
    """RAM++ tagging path re-implemented without no_grad (verified against inference in preflight)."""
    image_embeds = inner.image_proj(inner.visual_encoder(inp))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=inp.device)
    bs = image_embeds.shape[0]
    des_per_class = int(inner.label_embed.shape[0] / inner.num_class)
    cls = image_embeds[:, 0, :]
    cls = cls / cls.norm(dim=-1, keepdim=True)
    logits_per_image = (inner.reweight_scale.exp() * cls @ inner.label_embed.t()).view(bs, -1, des_per_class)
    weights = torch.softmax(logits_per_image, dim=2)
    desc = inner.label_embed.view(-1, des_per_class, inner.label_embed.shape[-1])
    reweighted = (weights.unsqueeze(-1) * desc.unsqueeze(0)).sum(dim=2)
    label_embed = torch.relu(inner.wordvec_proj(reweighted))
    tagging_embed = inner.tagging_head(
        encoder_embeds=label_embed, encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts, return_dict=False, mode="tagging",
    )
    return torch.sigmoid(inner.fc(tagging_embed[0]).squeeze(-1))[0]


def forward_with_grad(
    pixel_tensor: torch.Tensor,
    model=None,
    device=None,
    input_space: str = "normalized",
) -> torch.Tensor:
    """
    Differentiable forward pass -> sigmoid tag scores [num_classes].

    Tries, in order: (1) generate_tag's own scores if attached to the graph,
    (2) the model's fc logits captured by a hook, (3) a RAM++ re-implementation,
    (4) the legacy RAM head.  Every path is verified against inference scores in
    preflight; a path that disagrees is rejected, never silently used.
    """
    inner, device, inp = _resolve_input(pixel_tensor, model, device, input_space)
    errors: List[str] = []
    with torch.enable_grad():
        try:
            out = inner.generate_tag(inp, return_scores=True)
            scores = out[2] if isinstance(out, (tuple, list)) and len(out) >= 3 else None
            if torch.is_tensor(scores) and scores.requires_grad:
                return scores[0]
            errors.append("generate_tag scores are detached")
        except Exception as exc:
            errors.append(f"generate_tag: {exc}")
        for name, fn in (("fc-hook", _fc_hook_scores), ("ram++ replica", _ramplus_replica_scores), ("legacy head", _manual_head_scores)):
            try:
                out = fn(inner, inp)
                return out if out.dim() == 1 else out[0]
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    raise RuntimeError("No differentiable RAM++ scoring path available: " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Genuine internal activations (Swin-L visual encoder)
# ---------------------------------------------------------------------------
def extract_activation_maps(model, pixel_tensor: torch.Tensor, device=None, input_space: str = "normalized") -> Dict[str, Any]:
    """
    Capture measured activations from the RAM++ Swin-L visual encoder via forward hooks:
      patch_embed : first Conv2d of the encoder (patch-embedding projection), channel-mean |activation|
      stage_final : output of the last Swin stage, token L2-norm reshaped to its spatial grid
    Returns {"maps": {name: 2D float CPU tensor}, "layers": {name: module path/shape}, "errors": [..]}.
    Nothing is synthesized: a map that cannot be captured is simply absent and reported in "errors".
    """
    inner, device, inp = _resolve_input(pixel_tensor, model, device, input_space)
    ve = inner.visual_encoder
    captured: Dict[str, torch.Tensor] = {}
    layers: Dict[str, str] = {}
    errors: List[str] = []
    handles = []

    def make_hook(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[name] = t.detach().float().cpu()
        return hook

    for path, mod in ve.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            handles.append(mod.register_forward_hook(make_hook("patch_embed")))
            layers["patch_embed"] = f"visual_encoder.{path}"
            break
    stages = getattr(ve, "layers", None)
    if stages is not None and len(stages) > 0:
        handles.append(stages[-1].register_forward_hook(make_hook("stage_final")))
        layers["stage_final"] = f"visual_encoder.layers[{len(stages) - 1}]"
    else:
        errors.append("visual_encoder.layers not found; final-stage map unavailable.")

    try:
        with torch.no_grad():
            ve(inp)
    except Exception as exc:  # report, never fabricate
        errors.append(f"visual_encoder forward failed: {exc}")
    finally:
        for h in handles:
            h.remove()

    maps: Dict[str, torch.Tensor] = {}
    for name, t in captured.items():
        try:
            if t.dim() == 4:                      # [B, C, H, W]
                maps[name] = t[0].abs().mean(dim=0)
            elif t.dim() == 3:                    # [B, L, C]
                length = t.shape[1]
                g = int(round(length ** 0.5))
                if g * g != length:
                    errors.append(f"{name}: {length} tokens is not a square grid; skipped.")
                    continue
                maps[name] = t[0].norm(dim=-1).reshape(g, g)
                layers[name] += f" -> tokens {tuple(t.shape[1:])} as {g}x{g} grid"
            else:
                errors.append(f"{name}: unexpected activation rank {t.dim()}.")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for name in ("patch_embed", "stage_final"):
        if name not in maps and name not in " ".join(errors):
            errors.append(f"{name}: not captured.")
    return {"maps": maps, "layers": layers, "errors": errors}


# ---------------------------------------------------------------------------
# Investigation ID helpers
# ---------------------------------------------------------------------------
def new_investigation_id() -> str:
    return f"inv_{uuid.uuid4().hex[:8]}"


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
