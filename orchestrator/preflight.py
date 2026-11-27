"""
ModelShield Preflight Validator
Validates model, tensors, hyperparameters, and environment before launching attacks.
Distinguishes evaluation-integrity issues from model vulnerabilities.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import torch

from evaluation.trust_checks import (
    check_pixel_range,
    check_prediction_outputs,
)
from orchestrator.ram_space import PIXEL_TOL, to_pixel_space
from orchestrator.types import PreflightResult


def run_preflight_checks(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    pgd_steps: int = 10,
    pgd_alpha: float = 2 / 255,
    device: Optional[torch.device] = None,
) -> PreflightResult:
    """
    Execute non-destructive preflight validation before any attack generation.
    Returns PreflightResult with granular check results and failure diagnostics.
    """
    checks: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Check 1: Model availability and callable state
    if model is None or not callable(model):
        checks["model_callable"] = {
            "passed": False,
            "message": "Model instance is missing or not callable.",
        }
        errors.append("Model instance is not callable.")
    else:
        meta = getattr(model, "model_metadata", None)
        if meta:
            src = meta.get("checkpoint_source", "Unknown")
            arch = meta.get("architecture", model.__class__.__name__)
            is_pre = meta.get("is_pretrained", False)
            checks["model_callable"] = {
                "passed": True,
                "message": f"Model initialized: {arch} (Source: {src}, Pretrained: {is_pre}).",
            }
            checks["model_provenance"] = {
                "passed": True,
                "message": f"Architecture: {arch}; Source: {src}; Pretrained: {is_pre}.",
                "metadata": meta,
            }
        else:
            checks["model_callable"] = {
                "passed": True,
                "message": f"Model initialized: {model.__class__.__name__}.",
            }
            checks["model_provenance"] = {
                "passed": True,
                "message": f"Architecture: {model.__class__.__name__} (direct memory instance).",
                "metadata": {"architecture": model.__class__.__name__, "checkpoint_source": "Direct instance"},
            }

    # Check 2: Input tensor type and dimensionality
    if not isinstance(images, torch.Tensor):
        checks["image_tensor_type"] = {
            "passed": False,
            "message": f"Images must be torch.Tensor, got {type(images)}.",
        }
        errors.append("Images are not a torch.Tensor.")
    elif images.ndim != 4:
        checks["image_tensor_type"] = {
            "passed": False,
            "message": f"Images must be 4D (N, C, H, W), got ndim={images.ndim}.",
        }
        errors.append(f"Image tensor has incorrect dimensionality: {images.ndim}.")
    elif images.shape[1] not in (1, 3):
        checks["image_tensor_type"] = {
            "passed": False,
            "message": f"Image channel count must be 1 or 3, got C={images.shape[1]}.",
        }
        errors.append("Image channel count is not 1 or 3.")
    else:
        checks["image_tensor_type"] = {
            "passed": True,
            "message": f"Valid 4D image tensor: shape {tuple(images.shape)}.",
        }

    # Check 3: Label tensor type and alignment
    if not isinstance(labels, torch.Tensor):
        checks["label_tensor_type"] = {
            "passed": False,
            "message": f"Labels must be torch.Tensor, got {type(labels)}.",
        }
        errors.append("Labels are not a torch.Tensor.")
    elif labels.ndim != 1:
        checks["label_tensor_type"] = {
            "passed": False,
            "message": f"Labels must be 1D (N,), got ndim={labels.ndim}.",
        }
        errors.append(f"Labels tensor has incorrect dimensionality: {labels.ndim}.")
    elif isinstance(images, torch.Tensor) and images.ndim >= 1 and images.shape[0] != labels.shape[0]:
        checks["label_tensor_type"] = {
            "passed": False,
            "message": f"Batch size mismatch: images={images.shape[0]} vs labels={labels.shape[0]}.",
        }
        errors.append("Batch size mismatch between images and labels.")
    else:
        checks["label_tensor_type"] = {
            "passed": True,
            "message": f"Valid label tensor: {labels.shape[0]} samples.",
        }

    # Check 4: Pixel range validation [0.0, 1.0]
    if isinstance(images, torch.Tensor):
        pixel_check = check_pixel_range(images)
        checks["pixel_range"] = pixel_check
        if not pixel_check["passed"]:
            errors.append(f"Clean image pixels outside [0, 1]: {pixel_check['message']}.")

    # Check 5: Attack hyperparameter bounds
    param_errors = []
    if epsilon <= 0.0 or epsilon > 0.5:
        param_errors.append(f"Epsilon ({epsilon:.4f}) must be in (0, 0.5].")
    if pgd_steps < 1 or pgd_steps > 100:
        param_errors.append(f"PGD steps ({pgd_steps}) must be between 1 and 100.")
    if pgd_alpha <= 0.0 or pgd_alpha > epsilon:
        param_errors.append(f"PGD alpha ({pgd_alpha:.4f}) must be in (0, epsilon].")

    if param_errors:
        checks["hyperparameters"] = {
            "passed": False,
            "message": "; ".join(param_errors),
        }
        errors.extend(param_errors)
    else:
        checks["hyperparameters"] = {
            "passed": True,
            "message": f"Hyperparameters valid: eps={epsilon:.4f}, steps={pgd_steps}, alpha={pgd_alpha:.4f}.",
        }

    # Check 6: Forward inference probe & finite output validation
    if model is not None and callable(model) and isinstance(images, torch.Tensor) and isinstance(labels, torch.Tensor) and len(errors) == 0:
        target_device = device or next(model.parameters(), torch.device("cpu"))
        try:
            model.eval()
            sample_in = images[:min(4, len(images))].to(target_device)
            sample_lbl = labels[:min(4, len(labels))].to(target_device)
            with torch.no_grad():
                sample_out = model(sample_in)

            output_check = check_prediction_outputs(sample_out, sample_lbl)
            checks["prediction_outputs"] = output_check
            if not output_check["passed"]:
                errors.append(f"Forward output check failed: {output_check['message']}.")
        except Exception as exc:
            checks["prediction_outputs"] = {
                "passed": False,
                "message": f"Inference execution raised exception: {str(exc)}.",
            }
            errors.append(f"Model forward pass failed: {str(exc)}.")

    passed = len(errors) == 0
    return PreflightResult(
        passed=passed,
        checks=checks,
        errors=errors,
        timestamp=timestamp,
    )



def validate_ram_baseline(model, normalized, baseline, device=None):
    """Validate clean RAM++ inference without modifying the image or model."""
    checks = {}
    def record(name, passed, message):
        checks[name] = {"passed": bool(passed), "message": message}
    shape_ok = isinstance(normalized, torch.Tensor) and tuple(normalized.shape) == (1, 3, 384, 384) and normalized.is_floating_point()
    record("input_shape", shape_ok, "Expected one floating-point RGB 384x384 tensor.")
    if shape_ok:
        # The installed transform uses ImageNet normalization. Validate the
        # inverse representation, without clipping or modifying the input.
        pixels = to_pixel_space(normalized)
        finite = bool(torch.isfinite(pixels).all().item())
        valid = finite and bool(((pixels >= -PIXEL_TOL) & (pixels <= 1 + PIXEL_TOL)).all().item())
        record("inverse_pixel_range", valid, f"Recovered pixel range: [{pixels.min().item():.6f}, {pixels.max().item():.6f}].")
    scores = baseline.get("scores_tensor")
    vocabulary = list(baseline.get("tag_list", []))
    active = list(baseline.get("tags", []))
    score_ok = isinstance(scores, torch.Tensor) and scores.ndim == 1 and scores.numel() == len(vocabulary) and len(vocabulary) > 0
    if score_ok:
        score_ok = bool(torch.isfinite(scores).all().item() and ((scores >= 0) & (scores <= 1)).all().item())
    record("tag_scores", score_ok, "Baseline scores must be finite probabilities aligned to the vocabulary.")
    record("active_tags", set(active).issubset(set(vocabulary)), "Active tags must belong to the vocabulary.")
    record("model_available", callable(getattr(model, "generate_tag", None)), "RAM++ tagging interface must be available.")
    errors = [v["message"] for v in checks.values() if not v["passed"]]
    return PreflightResult(passed=not errors, checks=checks, errors=errors, timestamp=datetime.utcnow().isoformat()+"Z")


def run_ram_preflight_checks(
    model,
    normalized_images,
    baseline_scores,
    baseline_tags,
    tag_list,
    epsilon: float = 8 / 255,
    pgd_steps: int = 10,
    pgd_alpha: float = 2 / 255,
    device=None,
    differentiable_probe=None,
):
    """
    RAM++ preflight for the Orchestrator (replaces the CIFAR-era checks, which
    assume raw [0, 1] pixels and single-label logits).

    Non-destructive: nothing is clipped, rescaled or modified.  The optional
    `differentiable_probe` callable (returns a check dict) is only invoked when
    every cheap structural check has already passed, because it costs a full
    forward pass.
    """
    checks: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    def record(name, passed, message, **extra):
        checks[name] = {"passed": bool(passed), "message": message, **extra}
        if not passed:
            errors.append(message)

    # 1) Model interface and provenance
    has_iface = model is not None and callable(getattr(model, "generate_tag", None)) \
        and hasattr(model, "visual_encoder")
    if has_iface:
        meta: Dict[str, Any] = {}
        try:
            from model.ram_adapter import MODEL_METADATA
            meta = dict(MODEL_METADATA)
        except Exception:
            pass
        arch = meta.get("architecture", model.__class__.__name__)
        record("model_interface", not getattr(model, "training", False),
               f"{arch}: tagging interface available; "
               + ("eval mode." if not getattr(model, "training", False) else "model is in TRAIN mode."))
        checks["model_provenance"] = {
            "passed": True,
            "message": f"Architecture: {arch}; Source: {meta.get('checkpoint_source', 'unknown')}; "
                       f"Pretrained: {meta.get('is_pretrained', 'unknown')}.",
            "metadata": meta,
        }
    else:
        record("model_interface", False,
               "RAM++ model must expose generate_tag() and visual_encoder.")

    # 2) Input tensor (normalised model input)
    shape_ok = (isinstance(normalized_images, torch.Tensor)
                and tuple(normalized_images.shape) == (1, 3, 384, 384)
                and normalized_images.is_floating_point()
                and bool(torch.isfinite(normalized_images).all().item()))
    record("input_tensor", shape_ok,
           "Input is one finite floating-point RGB 384x384 tensor (shape [1,3,384,384])." if shape_ok
           else "Expected one finite floating-point RGB tensor of shape [1,3,384,384].")

    # 3) Inverse normalisation must recover valid raw pixels
    if shape_ok:
        px = to_pixel_space(normalized_images.detach().float().cpu())
        lo, hi = float(px.min().item()), float(px.max().item())
        record("inverse_pixel_range", lo >= -PIXEL_TOL and hi <= 1 + PIXEL_TOL,
               f"Inverse-normalised pixel range: [{lo:.6f}, {hi:.6f}] "
               f"(model-space range [{normalized_images.min().item():.4f}, {normalized_images.max().item():.4f}]).")

    # 4) Baseline scores and vocabulary
    vocab = list(tag_list) if tag_list is not None else []
    scores = None
    if isinstance(baseline_scores, torch.Tensor):
        scores = baseline_scores.detach().float().cpu().reshape(-1)
    score_ok = (scores is not None and len(vocab) > 0 and scores.numel() == len(vocab)
                and bool(torch.isfinite(scores).all().item())
                and bool(((scores >= 0) & (scores <= 1)).all().item()))
    record("tag_scores", score_ok,
           f"Baseline scores are finite probabilities aligned to the {len(vocab)}-tag vocabulary." if score_ok
           else "Baseline scores must be finite probabilities aligned one-to-one with the tag vocabulary.")

    # 5) Active tags
    active = list(baseline_tags or [])
    vocab_set = set(vocab)
    if not active:
        record("active_tags", False,
               "No baseline tags were recognised: tag-preservation metrics are undefined, "
               "so the mission cannot produce a meaningful robustness comparison.")
    else:
        unknown = [t for t in active if t not in vocab_set]
        record("active_tags", not unknown,
               f"{len(active)} baseline tags, all inside the RAM++ vocabulary." if not unknown
               else f"{len(unknown)} baseline tags are not in the vocabulary (e.g. {unknown[:3]}).")

    # 6) Attack hyper-parameters (pixel-space units)
    problems = []
    if epsilon <= 0.0 or epsilon > 0.5:
        problems.append(f"Epsilon ({epsilon:.4f}) must be in (0, 0.5].")
    if pgd_steps < 1 or pgd_steps > 100:
        problems.append(f"PGD steps ({pgd_steps}) must be between 1 and 100.")
    if pgd_alpha <= 0.0 or pgd_alpha > epsilon:
        problems.append(f"PGD alpha ({pgd_alpha:.4f}) must be in (0, epsilon].")
    record("hyperparameters", not problems,
           "; ".join(problems) if problems
           else f"Hyperparameters valid (pixel units): eps={epsilon:.4f}, steps={pgd_steps}, alpha={pgd_alpha:.4f}.")

    # 7) Differentiable attack path (expensive: only if everything above passed)
    if not errors and differentiable_probe is not None:
        try:
            probe = differentiable_probe()
        except Exception as exc:
            probe = {"passed": False, "message": f"Differentiable-path probe raised {type(exc).__name__}: {exc}"}
        checks["differentiable_path"] = probe
        if not probe.get("passed", False):
            errors.append(probe.get("message", "Differentiable attack path unavailable."))

    return PreflightResult(
        passed=not errors,
        checks=checks,
        errors=errors,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
