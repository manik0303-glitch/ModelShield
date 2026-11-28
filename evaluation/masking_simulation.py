"""
evaluation/masking_simulation.py
---------------------------------
Controlled gradient-masking simulation runner.

Introduces a deliberate non-differentiable transformation (quantization)
in a temporary pipeline and runs standard + BPDA attacks to produce
evidence for the existing gradient-masking detector.

Supports both single-label classification and multi-label (RAM++) pipelines.
"""

import math
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.bpda_pgd import bpda_pgd_attack, straight_through_quantize
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from evaluation.gradient_masking import analyze_gradient_masking


class _HardQuantizedPipeline(nn.Module):
    """
    Temporary, read-only wrapper that prepends hard quantization to the
    base model's forward pass. The base model is NOT mutated.
    """

    def __init__(self, base_model: nn.Module, levels: int):
        super().__init__()
        self.base_model = base_model
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = torch.round(x.clamp(0.0, 1.0) * (self.levels - 1)) / (self.levels - 1)
        return self.base_model(x_q)


def _safe_asr(success_count: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(100.0 * success_count / denominator, 2)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        if not math.isfinite(v):
            return None
        return round(v, 6)
    except Exception:
        return None


def _smoke_check_pipeline(
    pipeline: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    levels: int,
    epsilon: float,
    multi_label: bool = False,
) -> Dict[str, Any]:
    """
    Validates the quantized pipeline before any attack runs.
    """
    errors: List[str] = []

    # Shape / dtype / device / finite logits
    with torch.no_grad():
        try:
            out = pipeline(images)
        except Exception as exc:
            return {"passed": False, "errors": [f"Forward pass failed: {exc}"]}

    if out.shape[0] != images.shape[0] or out.ndim != 2:
        errors.append(f"Unexpected logit shape: {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        errors.append("Non-finite logits from quantized pipeline.")

    # Output range after quantization
    with torch.no_grad():
        x_q = torch.round(images.clamp(0, 1) * (levels - 1)) / (levels - 1)
    if x_q.min().item() < -1e-6 or x_q.max().item() > 1.0 + 1e-6:
        errors.append("Quantized tensor out of [0,1] range.")
    if not torch.isfinite(x_q).all():
        errors.append("Non-finite values in quantized tensor.")

    # Hard quantization must change at least some grid-misaligned inputs
    mid = 0.5 / (levels - 1)
    test_val = torch.full_like(images[:1], mid)
    q_val = torch.round(test_val * (levels - 1)) / (levels - 1)
    if (test_val - q_val).abs().max().item() < 1e-9:
        errors.append("Quantization did not change grid-misaligned inputs (levels may be 1).")

    # Ordinary gradient through hard quantization must be near zero
    x_for_grad = images[:2].clone().detach().requires_grad_(True)
    try:
        out_g = pipeline(x_for_grad)
        if multi_label:
            loss_g = F.binary_cross_entropy_with_logits(out_g, labels[:2].float())
        else:
            loss_g = F.cross_entropy(out_g, labels[:2].long())
        grads = torch.autograd.grad(loss_g, x_for_grad, allow_unused=True)[0]
    except Exception as exc:
        grads = None
    if grads is not None and grads.abs().max().item() > 1e-4:
        errors.append(
            "Ordinary gradient through hard quantization is unexpectedly large "
            f"(max |g| = {grads.abs().max().item():.2e}). "
            "Check pipeline: quantization may not be in the gradient path."
        )

    # BPDA backward should produce finite usable gradients
    x_bpda = images[:2].clone().detach().requires_grad_(True)
    try:
        x_bq = straight_through_quantize(x_bpda, levels)
        out_b = pipeline.base_model(x_bq)
        if multi_label:
            loss_b = F.binary_cross_entropy_with_logits(out_b, labels[:2].float())
        else:
            loss_b = F.cross_entropy(out_b, labels[:2].long())
        bpda_grads = torch.autograd.grad(loss_b, x_bpda, allow_unused=True)[0]
    except Exception as exc:
        bpda_grads = None
        errors.append(f"BPDA gradient computation failed: {exc}")

    if bpda_grads is None:
        errors.append("BPDA backward produced None gradients (disconnected graph).")
    elif not torch.isfinite(bpda_grads).all():
        errors.append("BPDA gradients contain non-finite values.")
    elif bpda_grads.abs().max().item() < 1e-9:
        errors.append("BPDA gradients are all zero (straight-through may be broken).")

    return {"passed": len(errors) == 0, "errors": errors}


def _measure_ordinary_gradient_stats(
    pipeline: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    multi_label: bool = False,
) -> Dict[str, Any]:
    x_in = images.clone().detach().requires_grad_(True)
    try:
        out = pipeline(x_in)
        if multi_label:
            loss = F.binary_cross_entropy_with_logits(out, labels.float())
        else:
            loss = F.cross_entropy(out, labels.long())
        grads = torch.autograd.grad(loss, x_in, allow_unused=True)[0]
    except Exception:
        return {"norm": None, "near_zero_fraction": None, "finite": False}

    if grads is None:
        return {"norm": None, "near_zero_fraction": 1.0, "finite": True}

    return {
        "norm": _safe_float(grads.norm(p=2).item()),
        "near_zero_fraction": _safe_float((grads.abs() < 1e-7).float().mean().item()),
        "finite": bool(torch.isfinite(grads).all().item()),
    }


def _measure_bpda_gradient_stats(
    base_model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    levels: int,
    multi_label: bool = False,
) -> Dict[str, Any]:
    x_in = images.clone().detach().requires_grad_(True)
    try:
        x_q = straight_through_quantize(x_in, levels)
        out = base_model(x_q)
        if multi_label:
            loss = F.binary_cross_entropy_with_logits(out, labels.float())
        else:
            loss = F.cross_entropy(out, labels.long())
        grads = torch.autograd.grad(loss, x_in, allow_unused=True)[0]
    except Exception:
        return {"norm": None, "near_zero_fraction": None, "finite": False}

    if grads is None:
        return {"norm": None, "near_zero_fraction": 1.0, "finite": True}

    return {
        "norm": _safe_float(grads.norm(p=2).item()),
        "near_zero_fraction": _safe_float((grads.abs() < 1e-7).float().mean().item()),
        "finite": bool(torch.isfinite(grads).all().item()),
    }


def _run_simulation_validator(
    smoke_ok: bool,
    smoke_errors: List[str],
    ord_grad_stats: Dict,
    bpda_grad_stats: Dict,
    sim_clean_acc: Optional[float],
    eligible_count: int,
    fgsm_asr: Optional[float],
    pgd_asr: Optional[float],
    bpda_asr: Optional[float],
    bpda_advantage_margin: float = 5.0,
    min_eligible: int = 2,
) -> Dict[str, Any]:
    checks = []
    all_passed = True

    c1_pass = smoke_ok
    checks.append({
        "name": "hard_quantization_active",
        "passed": c1_pass,
        "details": "Pipeline smoke checks passed." if c1_pass else f"Smoke check failures: {'; '.join(smoke_errors)}",
    })
    if not c1_pass:
        all_passed = False

    ord_norm = ord_grad_stats.get("norm")
    ord_nz = ord_grad_stats.get("near_zero_fraction")
    c2_pass = (
        ord_grad_stats.get("finite", False)
        and (ord_norm is None or ord_norm < 1e-4)
        and (ord_nz is None or ord_nz > 0.9)
    )
    checks.append({
        "name": "ordinary_gradient_obscured",
        "passed": c2_pass,
        "details": (
            f"Ordinary gradient L2 norm = {ord_norm}, near-zero fraction = {ord_nz}. "
            + ("Gradients appear appropriately obscured." if c2_pass
               else "Gradients are not sufficiently small — masking condition may not hold.")
        ),
    })
    if not c2_pass:
        all_passed = False

    bpda_finite = bpda_grad_stats.get("finite", False)
    bpda_nz = bpda_grad_stats.get("near_zero_fraction", 1.0)
    c3_pass = bpda_finite and (bpda_nz is None or bpda_nz < 0.9)
    checks.append({
        "name": "bpda_gradient_usable",
        "passed": c3_pass,
        "details": (
            f"BPDA gradient finite={bpda_finite}, near-zero fraction={bpda_nz}. "
            + ("BPDA gradients are usable." if c3_pass else "BPDA gradients are unavailable or zero.")
        ),
    })
    if not c3_pass:
        all_passed = False

    c4_pass = (sim_clean_acc is not None and sim_clean_acc > 0.0 and eligible_count >= min_eligible)
    checks.append({
        "name": "clean_accuracy_acceptable",
        "passed": c4_pass,
        "details": (
            f"Quantized pipeline clean accuracy = {sim_clean_acc}%, eligible samples = {eligible_count}. "
            + ("Sufficient samples for evaluation." if c4_pass
               else "Too few clean-correct examples for a reliable evaluation.")
        ),
    })
    if not c4_pass:
        all_passed = False

    if bpda_asr is None or pgd_asr is None:
        c5_pass = False
        c5_detail = "One or more ASRs are undefined; advantage cannot be assessed."
    else:
        advantage = bpda_asr - pgd_asr
        c5_pass = advantage >= bpda_advantage_margin
        c5_detail = (
            f"BPDA ASR = {bpda_asr:.2f}%, Standard PGD ASR = {pgd_asr:.2f}%, "
            f"advantage = {advantage:+.2f}pp (required >= {bpda_advantage_margin}pp). "
            + ("Advantage criterion met." if c5_pass else "Advantage criterion not met.")
        )
    checks.append({
        "name": "bpda_attack_advantage",
        "passed": c5_pass,
        "details": c5_detail,
    })
    if not c5_pass:
        all_passed = False

    if not smoke_ok or not bpda_grad_stats.get("finite", False):
        status = "ERROR"
        summary = "Simulation infrastructure failed; results cannot be trusted."
    elif all_passed:
        status = "VALIDATED"
        summary = (
            "The configured quantization experiment produced the expected "
            "gradient-obscuring behavior under all configured checks."
        )
    elif c4_pass and c2_pass and not c5_pass:
        status = "CONDITION_NOT_DEMONSTRATED"
        summary = (
            "The quantization condition was active, but the BPDA attack did not "
            "demonstrate a sufficient advantage over standard PGD in this run. "
            "Review the measurements; no detector result has been forced."
        )
    else:
        status = "INCONCLUSIVE"
        summary = (
            "One or more required simulation checks did not pass. "
            "The controlled condition may not have been established."
        )

    return {
        "status": status,
        "checks": checks,
        "summary": summary,
    }


def run_masking_simulation(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    quantization_levels: int = 16,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    pgd_steps: int = 10,
    sample_cap: int = 16,
    random_seed: int = 123,
    bpda_advantage_margin: float = 5.0,
    min_eligible: int = 2,
    multi_label: bool = False,
    threshold: Union[float, torch.Tensor] = 0.68,
    return_tensors: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, torch.Tensor]]]:
    """
    Orchestrates the full controlled gradient-masking simulation.
    Supports both single-label and multi-label models.
    """
    device = images.device
    model.eval()

    n = min(images.shape[0], sample_cap)
    imgs = images[:n].clone().detach()
    lbls = labels[:n].clone().detach()

    configuration = {
        "quantization_levels": quantization_levels,
        "epsilon": round(float(epsilon), 6),
        "alpha": round(float(alpha), 6),
        "pgd_steps": pgd_steps,
        "sample_count_requested": n,
        "random_seed": random_seed,
        "bpda_advantage_margin_pp": bpda_advantage_margin,
        "min_eligible_samples": min_eligible,
        "multi_label": multi_label,
    }

    def _inconclusive(reason: str) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, torch.Tensor]]]:
        payload = {
            "mode": "simulated_gradient_masking",
            "simulation_validation": {
                "status": "INCONCLUSIVE",
                "checks": [],
                "summary": reason,
            },
            "detector_result": {"status": "INCONCLUSIVE", "summary": reason, "signals": {}},
            "measurements": {
                "sample_count": n,
                "eligible_clean_correct_count": 0,
                "original_clean_accuracy": None,
                "simulated_clean_accuracy": None,
                "clean_accuracy_drop_percentage_points": None,
                "attacks": {
                    "fgsm": {"asr": None, "success_count": 0, "denominator": 0},
                    "pgd": {"asr": None, "success_count": 0, "denominator": 0},
                    "bpda_pgd": {"asr": None, "success_count": 0, "denominator": 0},
                },
                "gradient_checks": {
                    "ordinary_gradient_norm": None,
                    "ordinary_near_zero_fraction": None,
                    "bpda_gradient_norm": None,
                    "bpda_gradient_finite": False,
                },
            },
            "configuration": configuration,
            "limitations": [
                "This is a controlled quantization-based experiment.",
                reason,
            ],
        }
        if return_tensors:
            empty_bundle = {
                "clean_images": imgs.detach().cpu(),
                "pgd_images": imgs.detach().cpu(),
                "bpda_images": imgs.detach().cpu(),
            }
            return payload, empty_bundle
        return payload

    # Phase A: Capture original clean accuracy
    with torch.no_grad():
        orig_out = model(imgs)
        if multi_label:
            thresh = threshold if not isinstance(threshold, torch.Tensor) else threshold.to(orig_out.device)
            orig_preds = (orig_out.sigmoid() > thresh).long()
            # For open-vocabulary multi-label without ground truth annotations,
            # we consider a sample eligible if at least one tag is recognized with confidence
            orig_clean_correct = (orig_preds.sum(dim=1) > 0)
        else:
            orig_preds = orig_out.argmax(dim=1)
            orig_clean_correct = orig_preds.eq(lbls.long())
    orig_clean_acc = _safe_float(100.0 * orig_clean_correct.float().mean().item())

    # Phase B: Build and validate simulated pipeline
    sim_pipeline = _HardQuantizedPipeline(model, levels=quantization_levels)
    smoke_result = _smoke_check_pipeline(
        sim_pipeline, imgs, lbls, quantization_levels, epsilon, multi_label=multi_label
    )

    if not smoke_result["passed"]:
        return _inconclusive(
            "Simulation pipeline smoke checks failed: " + "; ".join(smoke_result["errors"])
        )

    # Phase C: Measure simulated clean accuracy
    with torch.no_grad():
        sim_out = sim_pipeline(imgs)
        if multi_label:
            thresh = threshold if not isinstance(threshold, torch.Tensor) else threshold.to(sim_out.device)
            sim_preds = (sim_out.sigmoid() > thresh).long()
            sim_clean_correct = (sim_preds.sum(dim=1) > 0)
        else:
            sim_preds = sim_out.argmax(dim=1)
            sim_clean_correct = sim_preds.eq(lbls.long())

    sim_clean_acc = _safe_float(100.0 * sim_clean_correct.float().mean().item())
    clean_acc_drop = (
        _safe_float(orig_clean_acc - sim_clean_acc)
        if orig_clean_acc is not None and sim_clean_acc is not None else None
    )
    eligible_count = int(sim_clean_correct.sum().item())

    if eligible_count < min_eligible:
        return _inconclusive(
            f"Only {eligible_count} samples correctly classified by quantized pipeline "
            f"(minimum required: {min_eligible}). "
            "Adjust quantization_levels or increase sample_cap."
        )

    # Gradient comparison
    ord_grad_stats = _measure_ordinary_gradient_stats(sim_pipeline, imgs, lbls, multi_label=multi_label)
    bpda_grad_stats = _measure_bpda_gradient_stats(model, imgs, lbls, quantization_levels, multi_label=multi_label)

    # Phase D: Run attacks
    torch.manual_seed(random_seed)

    # FGSM
    fgsm_adv = fgsm_attack(sim_pipeline, imgs, lbls, epsilon=epsilon, multi_label=multi_label)
    with torch.no_grad():
        f_out = sim_pipeline(fgsm_adv)
        if multi_label:
            thresh = threshold if not isinstance(threshold, torch.Tensor) else threshold.to(f_out.device)
            fgsm_preds = (f_out.sigmoid() > thresh).long()
            # In multi-label, fooled means at least one detected tag was wiped or flipped
            fgsm_correct = (fgsm_preds == sim_preds).all(dim=1)
        else:
            fgsm_preds = f_out.argmax(dim=1)
            fgsm_correct = fgsm_preds.eq(lbls.long())
    fgsm_fooled = sim_clean_correct & ~fgsm_correct
    fgsm_success = int(fgsm_fooled.sum().item())
    fgsm_asr = _safe_asr(fgsm_success, eligible_count)

    # Standard PGD (gradient masked through quantization)
    pgd_adv = pgd_attack(
        sim_pipeline, imgs, lbls,
        epsilon=epsilon, alpha=alpha, steps=pgd_steps, random_start=False, multi_label=multi_label,
    )
    with torch.no_grad():
        p_out = sim_pipeline(pgd_adv)
        if multi_label:
            thresh = threshold if not isinstance(threshold, torch.Tensor) else threshold.to(p_out.device)
            pgd_preds = (p_out.sigmoid() > thresh).long()
            pgd_correct = (pgd_preds == sim_preds).all(dim=1)
        else:
            pgd_preds = p_out.argmax(dim=1)
            pgd_correct = pgd_preds.eq(lbls.long())
    pgd_fooled = sim_clean_correct & ~pgd_correct
    pgd_success = int(pgd_fooled.sum().item())
    pgd_asr = _safe_asr(pgd_success, eligible_count)

    # BPDA-assisted PGD (adaptive bypass)
    bpda_adv = bpda_pgd_attack(
        model=model,
        images=imgs,
        labels=lbls,
        quantization_levels=quantization_levels,
        epsilon=epsilon,
        alpha=alpha,
        steps=pgd_steps,
        random_start=False,
        random_seed=random_seed,
        multi_label=multi_label,
    )
    with torch.no_grad():
        b_out = sim_pipeline(bpda_adv)
        if multi_label:
            thresh = threshold if not isinstance(threshold, torch.Tensor) else threshold.to(b_out.device)
            bpda_preds = (b_out.sigmoid() > thresh).long()
            bpda_correct = (bpda_preds == sim_preds).all(dim=1)
        else:
            bpda_preds = b_out.argmax(dim=1)
            bpda_correct = bpda_preds.eq(lbls.long())
    bpda_fooled = sim_clean_correct & ~bpda_correct
    bpda_success = int(bpda_fooled.sum().item())
    bpda_asr = _safe_asr(bpda_success, eligible_count)

    # Trust attestation check
    from evaluation.trust_checks import get_trust_attestation_summary
    attestation = get_trust_attestation_summary(
        original_images=imgs,
        adversarial_images=bpda_adv,
        outputs=b_out,
        labels=lbls,
        epsilon=epsilon,
        multi_label=multi_label,
    )

    # Per-sample audit telemetry
    sample_audits = []
    for idx in range(min(n, 8)):
        clean_match = bool(sim_clean_correct[idx].item()) if idx < len(sim_clean_correct) else False
        pgd_fool = bool(pgd_fooled[idx].item()) if idx < len(pgd_fooled) else False
        bpda_fool = bool(bpda_fooled[idx].item()) if idx < len(bpda_fooled) else False
        
        # Max Linf perturbation on this specific sample
        max_linf = float((bpda_adv[idx] - imgs[idx]).abs().max().item())

        sample_audits.append({
            "sample_index": idx,
            "clean_preds_bin": sim_preds[idx].detach().cpu().tolist() if multi_label else int(sim_preds[idx].item()),
            "pgd_preds_bin": pgd_preds[idx].detach().cpu().tolist() if multi_label else int(pgd_preds[idx].item()),
            "bpda_preds_bin": bpda_preds[idx].detach().cpu().tolist() if multi_label else int(bpda_preds[idx].item()),
            "target_labels_bin": lbls[idx].detach().cpu().tolist() if multi_label else int(lbls[idx].item()),
            "clean_correct": clean_match,
            "pgd_fooled": pgd_fool,
            "bpda_fooled": bpda_fool,
            "max_linf": round(max_linf, 5),
        })

    measurements = {
        "sample_count": n,
        "eligible_clean_correct_count": eligible_count,
        "original_clean_accuracy": orig_clean_acc,
        "simulated_clean_accuracy": sim_clean_acc,
        "clean_accuracy_drop_percentage_points": clean_acc_drop,
        "attacks": {
            "fgsm": {
                "asr": fgsm_asr,
                "success_count": fgsm_success,
                "denominator": eligible_count,
            },
            "pgd": {
                "asr": pgd_asr,
                "success_count": pgd_success,
                "denominator": eligible_count,
            },
            "bpda_pgd": {
                "asr": bpda_asr,
                "success_count": bpda_success,
                "denominator": eligible_count,
            },
        },
        "gradient_checks": {
            "ordinary_gradient_norm": ord_grad_stats.get("norm"),
            "ordinary_near_zero_fraction": ord_grad_stats.get("near_zero_fraction"),
            "bpda_gradient_norm": bpda_grad_stats.get("norm"),
            "bpda_gradient_finite": bpda_grad_stats.get("finite", False),
        },
        "trust_attestation": attestation,
        "sample_audits": sample_audits,
    }

    # Internal reference to tensors for interactive UI rendering (not in serializable measurements dict)
    tensors_bundle = {
        "clean_images": imgs.detach().cpu(),
        "pgd_images": pgd_adv.detach().cpu(),
        "bpda_images": bpda_adv.detach().cpu(),
    }

    # Phase E: Run detector
    try:
        detector_result = analyze_gradient_masking(
            model=sim_pipeline,
            images=imgs,
            labels=lbls,
            epsilon=epsilon,
            alpha=alpha,
            pgd_steps=pgd_steps,
            pgd_restarts=1,
            baseline_pgd_asr=pgd_asr,
            baseline_fgsm_asr=fgsm_asr,
            run_random_search=False,
            max_diagnostic_samples=n,
            multi_label=multi_label,
            threshold=threshold,
            fast_mode=True,
        )
    except Exception as exc:
        detector_result = {
            "status": "INCONCLUSIVE",
            "summary": f"Detector raised an exception: {exc}",
            "signals": {},
            "warning_reasons": [],
            "actionable_guidance": [],
        }

    # Phase F: Validator
    sim_validation = _run_simulation_validator(
        smoke_ok=smoke_result["passed"],
        smoke_errors=smoke_result.get("errors", []),
        ord_grad_stats=ord_grad_stats,
        bpda_grad_stats=bpda_grad_stats,
        sim_clean_acc=sim_clean_acc,
        eligible_count=eligible_count,
        fgsm_asr=fgsm_asr,
        pgd_asr=pgd_asr,
        bpda_asr=bpda_asr,
        bpda_advantage_margin=bpda_advantage_margin,
        min_eligible=min_eligible,
    )

    result_payload = {
        "mode": "simulated_gradient_masking",
        "simulation_validation": sim_validation,
        "detector_result": detector_result,
        "measurements": measurements,
        "configuration": configuration,
        "limitations": [
            "This is a controlled quantization-based experiment, not proof of detection for all masking mechanisms.",
            "BPDA uses a straight-through gradient estimator (not the true derivative of rounding).",
            "Results apply to this sample subset, epsilon budget, and quantization configuration only.",
        ],
    }

    if return_tensors:
        return result_payload, tensors_bundle
    return result_payload
