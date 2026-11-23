"""
Live Model Evaluation and Snapshot Generation Engine for ModelShield.

Executes genuine, deterministic evaluation of PyTorch models against the CIFAR-10 dataset,
computes FGSM and PGD adversarial metrics and trust-check diagnostics,
and constructs fully validated EvaluationSnapshot objects for the Model Security Regression Gate.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from robustbench.data import load_cifar10

from attacks.ensemble import evaluate_attack_ensemble
from evaluation.regression_gate import (
    EvaluationMetadata,
    EvaluationSnapshot,
    RegressionGateResult,
    RegressionThresholds,
    evaluate_regression_gate,
    validate_snapshot_integrity,
)
from evaluation.trust_checks import (
    check_metric_consistency,
    check_perturbation_budget,
    check_pixel_range,
    check_prediction_outputs,
)
from evaluation.trust_flag import generate_trust_flag


def load_cifar10_samples(
    n_samples: int = 16,
    data_dir: str = "./data",
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads genuine CIFAR-10 test examples using RobustBench / local torchvision dataset.
    """
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = load_cifar10(n_examples=n_samples, data_dir=data_dir)
    return images.to(target_device), labels.to(target_device)


def evaluate_model_live(
    model: nn.Module,
    model_name: str,
    architecture: str = "UnknownArchitecture",
    dataset_name: str = "CIFAR-10",
    n_samples: int = 16,
    epsilon: float = 8.0 / 255.0,
    alpha: Optional[float] = None,
    pgd_steps: int = 10,
    random_start: bool = True,
    device: Optional[torch.device] = None,
    preloaded_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[EvaluationSnapshot, Dict[str, Any]]:
    """
    Evaluates a live model on genuine CIFAR-10 data under explicit attack parameters,
    computes trust checks, and builds an EvaluationSnapshot.

    Parameters:
        model: Evaluated PyTorch nn.Module.
        model_name: Descriptive identifier for the model.
        architecture: Descriptive name of the model architecture.
        dataset_name: Dataset identifier (defaults to "CIFAR-10").
        n_samples: Number of evaluation test samples.
        epsilon: Maximum L_inf perturbation radius.
        alpha: PGD step size (defaults to min(2/255, epsilon)).
        pgd_steps: Number of PGD iterations.
        random_start: Whether to use random initialization for PGD.
        device: Computation device.
        preloaded_data: Optional (images, labels) tuple to avoid re-loading dataset.
        progress_callback: Optional callback(status_message, progress_fraction).

    Returns:
        (EvaluationSnapshot, raw_evaluation_results)
    """
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device).eval()

    step_size = alpha if alpha is not None else min(2.0 / 255.0, epsilon)

    if progress_callback:
        progress_callback(f"Loading {n_samples} samples from {dataset_name}...", 0.1)

    if preloaded_data is not None:
        images, labels = preloaded_data
        images = images[:n_samples].to(target_device)
        labels = labels[:n_samples].to(target_device)
    else:
        images, labels = load_cifar10_samples(n_samples=n_samples, device=target_device)

    if len(labels) == 0:
        raise ValueError(f"Insufficient dataset samples: loaded {len(labels)} items.")

    if progress_callback:
        progress_callback(f"Running FGSM & PGD attacks on {model_name}...", 0.3)

    raw_results = evaluate_attack_ensemble(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        alpha=step_size,
        pgd_steps=pgd_steps,
    )

    if progress_callback:
        progress_callback(f"Executing ModelShield trust checks on {model_name}...", 0.7)

    with torch.no_grad():
        clean_outputs = model(images)

    fgsm_adv = raw_results["fgsm_images"]
    pgd_adv = raw_results["pgd_images"]

    checks = {
        "Perturbation budget (FGSM)": check_perturbation_budget(images, fgsm_adv, epsilon),
        "Perturbation budget (PGD)": check_perturbation_budget(images, pgd_adv, epsilon),
        "Pixel range (FGSM)": check_pixel_range(fgsm_adv),
        "Pixel range (PGD)": check_pixel_range(pgd_adv),
        "Prediction outputs": check_prediction_outputs(clean_outputs, labels),
        "Metric consistency": check_metric_consistency(
            raw_results["clean_accuracy"],
            raw_results["fgsm_accuracy"],
            raw_results["worst_case_attack_success_rate"],
        ),
    }

    trust_flag = generate_trust_flag(checks)

    # Clean serializable trust flag
    clean_trust_flag = {
        "status": str(trust_flag.get("status", "UNKNOWN")),
        "summary": str(trust_flag.get("summary", "")),
        "passed": int(trust_flag.get("passed", 0)),
        "failed": int(trust_flag.get("failed", 0)),
        "total": int(trust_flag.get("total", 0)),
    }

    metadata = EvaluationMetadata(
        model_name=str(model_name),
        dataset_name=str(dataset_name),
        num_samples=int(len(labels)),
        epsilon=float(epsilon),
        alpha=float(step_size),
        pgd_steps=int(pgd_steps),
        random_start=bool(random_start),
        timestamp=datetime.now(timezone.utc).isoformat(),
        extra={
            "architecture": str(architecture),
            "device": str(target_device),
        },
    )

    metrics = {
        "clean_accuracy": float(raw_results["clean_accuracy"]),
        "fgsm_accuracy": float(raw_results["fgsm_accuracy"]),
        "pgd_accuracy": float(raw_results["pgd_accuracy"]),
        "fgsm_attack_success_rate": float(raw_results["fgsm_attack_success_rate"]),
        "pgd_attack_success_rate": float(raw_results["pgd_attack_success_rate"]),
        "worst_case_attack_success_rate": float(raw_results["worst_case_attack_success_rate"]),
    }

    snapshot = EvaluationSnapshot(
        metadata=metadata,
        metrics=metrics,
        trust_flag=clean_trust_flag,
        raw_checks=checks,
    )

    # Validate snapshot integrity before returning
    is_valid, errs, val_snap = validate_snapshot_integrity(snapshot, model_name)
    if not is_valid or val_snap is None:
        raise ValueError(f"Generated snapshot for '{model_name}' failed integrity validation: {'; '.join(errs)}")

    if progress_callback:
        progress_callback(f"Evaluation complete for {model_name}.", 1.0)

    return val_snap, raw_results


def is_snapshot_compatible_with_config(
    snapshot: Optional[EvaluationSnapshot],
    dataset_name: str,
    n_samples: int,
    epsilon: float,
    pgd_steps: int,
    alpha: Optional[float] = None,
    eps_tolerance: float = 1e-5,
) -> bool:
    """
    Determines whether an existing evaluation snapshot matches a requested configuration
    and can therefore be reused without repeating model inference.
    """
    if snapshot is None or snapshot.metadata is None:
        return False

    meta = snapshot.metadata
    if meta.dataset_name.strip().upper() != dataset_name.strip().upper():
        return False

    if meta.num_samples != n_samples:
        return False

    if abs(meta.epsilon - epsilon) > eps_tolerance:
        return False

    if meta.pgd_steps != pgd_steps:
        return False

    if alpha is not None and meta.alpha is not None:
        if abs(meta.alpha - alpha) > eps_tolerance:
            return False

    return True


def run_live_regression_pipeline(
    baseline_model: Optional[nn.Module],
    candidate_model: nn.Module,
    baseline_name: str,
    candidate_name: str,
    candidate_architecture: str,
    baseline_architecture: str = "WideResNet-28-10 (Carmon2019 / RobustBench)",
    existing_baseline_snapshot: Optional[EvaluationSnapshot] = None,
    dataset_name: str = "CIFAR-10",
    n_samples: int = 16,
    epsilon: float = 8.0 / 255.0,
    alpha: Optional[float] = None,
    pgd_steps: int = 10,
    thresholds: Optional[RegressionThresholds] = None,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[EvaluationSnapshot, EvaluationSnapshot, RegressionGateResult]:
    """
    Orchestrates the complete live evaluation pipeline:
    1. Obtains or evaluates the Baseline snapshot.
    2. Genuinely evaluates the Candidate model under identical conditions.
    3. Runs the Model Security Regression Gate and returns all artifacts.
    """
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step_size = alpha if alpha is not None else min(2.0 / 255.0, epsilon)

    # Preload the exact dataset once so both models are evaluated on identical images & labels
    if progress_callback:
        progress_callback(f"Loading {n_samples} CIFAR-10 test examples...", 0.05)
    data_tuple = load_cifar10_samples(n_samples=n_samples, device=target_device)

    # 1. Baseline Snapshot
    baseline_snap = existing_baseline_snapshot
    can_reuse = is_snapshot_compatible_with_config(
        snapshot=baseline_snap,
        dataset_name=dataset_name,
        n_samples=n_samples,
        epsilon=epsilon,
        pgd_steps=pgd_steps,
        alpha=step_size,
    )

    if not can_reuse:
        if baseline_model is None:
            raise ValueError(
                "Baseline evaluation required, but no baseline model or compatible snapshot was provided."
            )
        if progress_callback:
            progress_callback(f"Evaluating Baseline model '{baseline_name}'...", 0.2)
        baseline_snap, _ = evaluate_model_live(
            model=baseline_model,
            model_name=baseline_name,
            architecture=baseline_architecture,
            dataset_name=dataset_name,
            n_samples=n_samples,
            epsilon=epsilon,
            alpha=step_size,
            pgd_steps=pgd_steps,
            device=target_device,
            preloaded_data=data_tuple,
            progress_callback=lambda msg, frac: progress_callback(f"[Baseline] {msg}", 0.1 + 0.35 * frac) if progress_callback else None,
        )
    else:
        if progress_callback:
            progress_callback("Reusing compatible Baseline snapshot.", 0.45)

    # 2. Candidate Snapshot
    if progress_callback:
        progress_callback(f"Evaluating Candidate model '{candidate_name}'...", 0.5)

    candidate_snap, _ = evaluate_model_live(
        model=candidate_model,
        model_name=candidate_name,
        architecture=candidate_architecture,
        dataset_name=dataset_name,
        n_samples=n_samples,
        epsilon=epsilon,
        alpha=step_size,
        pgd_steps=pgd_steps,
        device=target_device,
        preloaded_data=data_tuple,
        progress_callback=lambda msg, frac: progress_callback(f"[Candidate] {msg}", 0.5 + 0.4 * frac) if progress_callback else None,
    )

    # 3. Regression Gate Evaluation
    if progress_callback:
        progress_callback("Running Model Security Regression Gate...", 0.95)

    gate_thresholds = thresholds or RegressionThresholds()
    gate_result = evaluate_regression_gate(
        baseline_snapshot=baseline_snap,
        candidate_snapshot=candidate_snap,
        thresholds=gate_thresholds,
    )

    if progress_callback:
        progress_callback("Live regression gate pipeline complete.", 1.0)

    return baseline_snap, candidate_snap, gate_result

