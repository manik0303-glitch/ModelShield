"""
Pixel-space helpers for the RAM++ Red-Team Orchestrator.

The shared RAM++ preprocessing (model.ram_adapter.preprocess_pil) returns
ImageNet-NORMALISED tensors (observed range about [-2.12, 2.45]), but every
integrity check, perturbation budget and image preview in the Orchestrator is
defined on RAW PIXELS in [0, 1].  All conversion between the two spaces lives
here so that no other module has to guess which space a tensor is in.
"""
from typing import Any, Dict

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PIXEL_TOL = 1e-5  # float32 round-trip tolerance for [0, 1] and budget checks


def _stat(ref: torch.Tensor, values) -> torch.Tensor:
    shape = (3, 1, 1) if ref.dim() == 3 else (1, 3, 1, 1)
    return ref.new_tensor(values).view(*shape)


def to_pixel_space(x: torch.Tensor) -> torch.Tensor:
    """Normalised model input -> raw pixels (inverse ImageNet normalisation)."""
    return x * _stat(x, IMAGENET_STD) + _stat(x, IMAGENET_MEAN)


def to_model_space(p: torch.Tensor) -> torch.Tensor:
    """Raw pixels in [0, 1] -> normalised model input."""
    return (p - _stat(p, IMAGENET_MEAN)) / _stat(p, IMAGENET_STD)


def pixel_linf(clean_norm: torch.Tensor, adv_norm: torch.Tensor) -> float:
    """Measured L-infinity distance in RAW PIXEL units between two model inputs."""
    d = (adv_norm.detach().float().cpu() - clean_norm.detach().float().cpu())
    return float((d * _stat(d, IMAGENET_STD)).abs().max().item())


def check_pixel_range_tolerant(pixels: torch.Tensor, tol: float = PIXEL_TOL) -> Dict[str, Any]:
    """Raw-pixel range check in [0, 1] with float32 round-trip tolerance."""
    p = pixels.detach().float().cpu()
    finite = bool(torch.isfinite(p).all().item())
    mn, mx = float(p.min().item()), float(p.max().item())
    passed = finite and mn >= -tol and mx <= 1.0 + tol
    return {
        "passed": passed,
        "min": round(mn, 6),
        "max": round(mx, 6),
        "message": f"Raw pixel range: [{mn:.6f}, {mx:.6f}] (finite={finite}).",
    }


def check_budget_pixel(clean_norm: torch.Tensor, adv_norm: torch.Tensor,
                       epsilon: float, tol: float = PIXEL_TOL) -> Dict[str, Any]:
    """L-infinity budget check measured in RAW PIXEL units."""
    if clean_norm.shape != adv_norm.shape:
        return {"passed": False, "message": "Shape mismatch between clean and adversarial tensors."}
    m = pixel_linf(clean_norm, adv_norm)
    return {
        "passed": m <= epsilon + tol,
        "max_perturbation": round(m, 6),
        "epsilon": round(float(epsilon), 6),
        "message": f"Measured pixel-space Linf = {m:.6f} (budget {epsilon:.6f}).",
    }
