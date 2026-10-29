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

