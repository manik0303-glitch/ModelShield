import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def compute_gradient_health(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn=F.cross_entropy,
) -> Dict[str, Any]:
    """
    Computes per-sample input gradient statistics to diagnose vanishing,
    exploding, non-finite, or zeroed gradients.
    """
    was_training = model.training
    model.eval()

    images_clone = images.clone().detach().requires_grad_(True)
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(images_clone)
        loss = loss_fn(outputs, labels)
        grads = torch.autograd.grad(loss, images_clone, allow_unused=True)[0]
    except Exception as exc:
        if was_training:
            model.train()
        return {
            "status": "ERROR",
            "median_l2_norm": None,
            "median_mean_abs": None,
            "median_near_zero_fraction": None,
            "non_finite": True,
            "details": f"Failed to compute input gradients: {exc}",
        }
    finally:
        if was_training:
            model.train()

    if grads is None:
        return {
            "status": "SUSPICIOUS",
            "median_l2_norm": 0.0,
            "median_mean_abs": 0.0,
            "median_near_zero_fraction": 1.0,
            "non_finite": False,
            "details": "Input gradients are disconnected/None (non-differentiable model path).",
        }

    # Per-sample metrics
    batch_size = images.shape[0]
    l2_norms = []
    mean_abs_list = []
    near_zero_fractions = []
    has_non_finite = False

    for i in range(batch_size):
        g = grads[i]
        if not torch.isfinite(g).all().item():
            has_non_finite = True
            continue

        l2 = torch.norm(g, p=2).item()
        mean_abs = g.abs().mean().item()
        near_zero_frac = (g.abs() < 1e-7).float().mean().item()

        l2_norms.append(l2)
        mean_abs_list.append(mean_abs)
        near_zero_fractions.append(near_zero_frac)

    if has_non_finite or len(l2_norms) == 0:
        return {
            "status": "ERROR",
            "median_l2_norm": None,
            "median_mean_abs": None,
            "median_near_zero_fraction": None,
            "non_finite": True,
            "details": "Non-finite (NaN or Inf) values detected in input gradients.",
        }

    l2_norms.sort()
    mean_abs_list.sort()
    near_zero_fractions.sort()

    mid = len(l2_norms) // 2
    median_l2 = float(l2_norms[mid])
    median_mean_abs = float(mean_abs_list[mid])
    median_near_zero = float(near_zero_fractions[mid])

    # Literature heuristics (Athalye et al.): vanishing norm or >90% zero elements
    is_suspicious = (median_l2 < 1e-5) or (median_near_zero > 0.90)

    if is_suspicious:
        details = (
            f"Suspiciously small or sparse input gradients: "
            f"median L2 norm = {median_l2:.2e}, "
            f"near-zero element fraction = {median_near_zero * 100:.1f}%."
        )
        status = "SUSPICIOUS"
    else:
        details = (
            f"Healthy input gradients: median L2 norm = {median_l2:.4f}, "
            f"mean abs = {median_mean_abs:.6f}."
        )
        status = "OK"

    return {
        "status": status,
        "median_l2_norm": round(median_l2, 6),
        "median_mean_abs": round(median_mean_abs, 8),
        "median_near_zero_fraction": round(median_near_zero, 4),
        "non_finite": False,
        "details": details,
    }


def evaluate_loss_landscape_ray(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_correct_mask: torch.Tensor,
    epsilon: float = 8 / 255,
    alphas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates loss along the 1D direction of the gradient ray:
    x(alpha) = clamp(x + alpha * sign(grad_x L), 0, 1).
    Uses 0 backward passes (only forward passes).
    """
    if alphas is None:
        alphas = [0.0, 0.5 * epsilon, 1.0 * epsilon]

    eligible_count = int(clean_correct_mask.sum().item())
    if eligible_count == 0:
        return {
            "status": "NOT_AVAILABLE",
            "alphas": [float(round(a, 6)) for a in alphas],
            "losses": [],
            "slope": None,
            "details": "No clean-correct samples available for loss ray evaluation.",
        }

    sub_images = images[clean_correct_mask]
    sub_labels = labels[clean_correct_mask]

    # Compute sign of input gradient
    x_in = sub_images.clone().detach().requires_grad_(True)
    try:
        model.zero_grad(set_to_none=True)
        out = model(x_in)
        loss = F.cross_entropy(out, sub_labels)
        grad = torch.autograd.grad(loss, x_in, allow_unused=True)[0]
    except Exception as exc:
        return {
            "status": "ERROR",
            "alphas": [float(round(a, 6)) for a in alphas],
            "losses": [],
            "slope": None,
            "details": f"Failed gradient calculation for loss ray: {exc}",
        }

    if grad is None:
        return {
            "status": "SUSPICIOUS",
            "alphas": [float(round(a, 6)) for a in alphas],
            "losses": [],
            "slope": None,
            "details": "Model path is non-differentiable; directional loss cannot be computed.",
        }

    grad_sign = grad.sign()
    losses = []

    model.eval()
    with torch.no_grad():
        for a in alphas:
            perturbed = torch.clamp(sub_images + a * grad_sign, 0.0, 1.0)
            outputs = model(perturbed)
            sample_loss = F.cross_entropy(outputs, sub_labels).item()
            losses.append(sample_loss)

    l0 = losses[0]
    l_end = losses[-1]
    slope = l_end - l0

    # If loss along gradient direction actually decreases or is completely flat despite non-trivial base loss
    if slope < -1e-4:
        status = "SUSPICIOUS"
        details = (
            f"Loss decreased along the gradient direction (delta = {slope:.4f}). "
            "The gradient vector does not point in a direction of increasing loss (shattered/misleading gradient)."
        )
    elif abs(slope) < 1e-6 and l0 > 0.05:
        status = "SUSPICIOUS"
        details = (
            f"Loss landscape is completely flat along the gradient ray (slope = {slope:.2e}). "
            "Vanishing gradient or local plateau detected."
        )
    else:
        status = "OK"
        details = (
            f"Loss increases consistently along the gradient direction (delta = {slope:+.4f}). "
            "Directional monotonicity confirmed."
        )

    return {
        "status": status,
        "alphas": [float(round(a, 6)) for a in alphas],
        "losses": [float(round(l, 6)) for l in losses],
        "slope": float(round(slope, 6)),
        "details": details,
    }


def evaluate_pgd_restarts(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_correct_mask: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    restarts: int = 2,
    baseline_pgd_asr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates PGD across independent random restarts.
    Reuses baseline_pgd_asr if supplied as Restart #1 to save CPU time.
    """
    eligible = int(clean_correct_mask.sum().item())
    if eligible == 0:
        return {
            "status": "NOT_AVAILABLE",
            "restart_success_rates": [],
            "spread_percentage_points": 0.0,
            "details": "No clean-correct samples available to calculate ASR across restarts.",
        }

    rates = []
    if baseline_pgd_asr is not None:
        rates.append(baseline_pgd_asr)
        runs_needed = max(0, restarts - 1)
    else:
        runs_needed = restarts

    model.eval()
    for _ in range(runs_needed):
        adv = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=True,
        )
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        adv_correct = preds.eq(labels)
        fooled = clean_correct_mask & ~adv_correct
        asr = 100.0 * float(fooled.sum().item()) / eligible
        rates.append(round(asr, 2))

    spread = max(rates) - min(rates) if len(rates) > 1 else 0.0
    is_suspicious = spread >= 20.0

    if is_suspicious:
        status = "SUSPICIOUS"
        details = (
            f"High sensitivity to random restarts: spread of {spread:.2f} percentage points "
            f"across {len(rates)} restarts. Attack landscape may be shattered or rugged."
        )
    else:
        status = "OK"
        details = (
            f"Restart sensitivity is stable: spread is {spread:.2f} percentage points "
            f"across {len(rates)} restarts."
        )

    return {
        "status": status,
        "restart_success_rates": rates,
        "spread_percentage_points": round(spread, 2),
        "details": details,
    }


def evaluate_attack_strength_consistency(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_correct_mask: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    max_steps: int = 20,
    snapshot_steps: Optional[List[int]] = None,
    test_epsilons: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates monotonicity with increasing optimization effort:
    1. Step Trajectory Snapshotting: Evaluates intermediate steps (e.g. 5, 10, 20)
       during a single PGD trajectory (saving redundant backward passes).
    2. Epsilon Sweep: Evaluates PGD at scaled perturbation budgets (e.g. 0.5*eps, 1.0*eps, 2.0*eps).
    """
    if snapshot_steps is None:
        snapshot_steps = [5, 10, 20]
    if test_epsilons is None:
        test_epsilons = [0.5 * epsilon, 1.0 * epsilon, 2.0 * epsilon]

    eligible = int(clean_correct_mask.sum().item())
    if eligible == 0:
        return {
            "status": "NOT_AVAILABLE",
            "step_sweep": {"steps": snapshot_steps, "success_rates": []},
            "epsilon_sweep": {"epsilons": [round(e, 5) for e in test_epsilons], "success_rates": []},
            "non_monotonic_drops": [],
            "details": "No clean-correct samples available for strength consistency check.",
        }

    drops = []
    model.eval()

    # Step Trajectory Snapshotting (single run up to max_steps)
    step_rates = {}
    target_steps = sorted([s for s in snapshot_steps if s <= max_steps])
    if not target_steps:
        target_steps = [max_steps]

    orig_images = images.clone().detach()
    adv_images = orig_images + torch.empty_like(orig_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0.0, 1.0)

    for step_idx in range(1, max_steps + 1):
        adv_images = adv_images.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        grads = torch.autograd.grad(loss, adv_images, allow_unused=True)[0]

        if grads is None:
            break

        adv_images = adv_images.detach() + alpha * grads.sign()
        pert = torch.clamp(adv_images - orig_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(orig_images + pert, 0.0, 1.0)

        if step_idx in target_steps:
            with torch.no_grad():
                preds = model(adv_images).argmax(dim=1)
            adv_correct = preds.eq(labels)
            fooled = clean_correct_mask & ~adv_correct
            asr = 100.0 * float(fooled.sum().item()) / eligible
            step_rates[step_idx] = round(asr, 2)

    step_results = [step_rates.get(s, 0.0) for s in target_steps]
    for i in range(1, len(step_results)):
        drop = step_results[i - 1] - step_results[i]
        if drop > 5.0:
            drops.append(
                f"Step drop: ASR fell by {drop:.1f}% when increasing PGD steps from "
                f"{target_steps[i - 1]} to {target_steps[i]}."
            )

    # Epsilon Sweep
    eps_results = []
    for eps_val in test_epsilons:
        adv_eps = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=eps_val,
            alpha=min(2 / 255, eps_val),
            steps=10,
            random_start=True,
        )
        with torch.no_grad():
            preds = model(adv_eps).argmax(dim=1)
        adv_correct = preds.eq(labels)
        fooled = clean_correct_mask & ~adv_correct
        asr = 100.0 * float(fooled.sum().item()) / eligible
        eps_results.append(round(asr, 2))

    for i in range(1, len(eps_results)):
        drop = eps_results[i - 1] - eps_results[i]
        if drop > 5.0:
            drops.append(
                f"Epsilon drop: ASR fell by {drop:.1f}% when increasing budget from "
                f"{test_epsilons[i - 1]:.4f} to {test_epsilons[i]:.4f}."
            )

    is_suspicious = len(drops) > 0
    if is_suspicious:
        status = "SUSPICIOUS"
        details = (
            f"Non-monotonic attack strength observed: {len(drops)} drop(s) detected. "
            "Increasing optimization budget did not yield equal or higher attack success."
        )
    else:
        status = "OK"
        details = "Attack success rate maintained or improved monotonically with steps and epsilon."

    return {
        "status": status,
        "step_sweep": {
            "steps": target_steps,
            "success_rates": step_results,
        },
        "epsilon_sweep": {
            "epsilons": [round(float(e), 5) for e in test_epsilons],
            "success_rates": eps_results,
        },
        "non_monotonic_drops": drops,
        "details": details,
    }


def evaluate_fgsm_vs_pgd(
    fgsm_asr: float,
    pgd_asr: float,
    tolerance: float = 5.0,
) -> Dict[str, Any]:
    """
    Compares FGSM ASR against iterative PGD ASR.
    One-step attacks significantly outperforming iterative attacks is a classic masking signal.
    """
    diff = fgsm_asr - pgd_asr
    is_suspicious = diff > tolerance

    if is_suspicious:
        status = "SUSPICIOUS"
        details = (
            f"One-step FGSM outperformed iterative PGD by {diff:.2f} percentage points "
            f"(FGSM: {fgsm_asr:.2f}%, PGD: {pgd_asr:.2f}%). "
            "Iterative gradient descent failed to find better adversaries than single-step gradient."
        )
    else:
        status = "OK"
        details = (
            f"Iterative PGD ({pgd_asr:.2f}%) matches or outperforms one-step FGSM ({fgsm_asr:.2f}%)."
        )

    return {
        "status": status,
        "fgsm_success_rate": round(fgsm_asr, 2),
        "pgd_success_rate": round(pgd_asr, 2),
        "details": details,
    }


def evaluate_random_search_baseline(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_correct_mask: torch.Tensor,
    epsilon: float = 8 / 255,
    pgd_asr: float = 0.0,
    num_samples: int = 25,
) -> Dict[str, Any]:
    """
    Lightweight Zero-Order (Gradient-Free) Random Perturbation Check.
    If pure random uniform noise achieves higher ASR than gradient-based PGD,
    gradients are demonstrably misleading or obfuscated.
    """
    eligible = int(clean_correct_mask.sum().item())
    if eligible == 0:
        return {
            "status": "NOT_AVAILABLE",
            "random_search_success_rate": 0.0,
            "pgd_success_rate": round(pgd_asr, 2),
            "details": "No clean-correct samples available for random search.",
        }

    sub_images = images[clean_correct_mask]
    sub_labels = labels[clean_correct_mask]
    n_sub = sub_images.shape[0]

    model.eval()
    fooled_any = torch.zeros(n_sub, dtype=torch.bool, device=images.device)

    with torch.no_grad():
        for _ in range(num_samples):
            noise = torch.empty_like(sub_images).uniform_(-epsilon, epsilon)
            perturbed = torch.clamp(sub_images + noise, 0.0, 1.0)
            preds = model(perturbed).argmax(dim=1)
            fooled_any |= (preds != sub_labels)

    random_asr = 100.0 * float(fooled_any.sum().item()) / eligible
    diff = random_asr - pgd_asr
    is_suspicious = diff > 5.0

    if is_suspicious:
        status = "SUSPICIOUS"
        details = (
            f"Gradient-free random noise achieved higher attack success ({random_asr:.2f}%) "
            f"than gradient-guided PGD ({pgd_asr:.2f}%). "
            "White-box gradients are less informative than random exploration."
        )
    else:
        status = "OK"
        details = (
            f"Gradient-guided PGD ({pgd_asr:.2f}%) effectively outperforms "
            f"random uniform perturbation ({random_asr:.2f}%)."
        )

    return {
        "status": status,
        "random_search_success_rate": round(random_asr, 2),
        "pgd_success_rate": round(pgd_asr, 2),
        "details": details,
    }


def analyze_gradient_masking(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    pgd_steps: int = 10,
    pgd_restarts: int = 2,
    baseline_pgd_asr: Optional[float] = None,
    baseline_fgsm_asr: Optional[float] = None,
    run_random_search: bool = True,
    max_diagnostic_samples: int = 16,
) -> Dict[str, Any]:
    """
    Master orchestrator evaluating heuristic indicators of gradient masking.
    Returns a JSON-serializable dictionary with measurements, findings, limitations,
    and a final status.
    """
    model.eval()
    device = images.device

    # Cap diagnostic sample size to guarantee interactive speed on CPU
    if images.shape[0] > max_diagnostic_samples:
        diag_images = images[:max_diagnostic_samples]
        diag_labels = labels[:max_diagnostic_samples]
    else:
        diag_images = images
        diag_labels = labels

    # 1. Clean accuracy and clean-correct mask
    with torch.no_grad():
        clean_outputs = model(diag_images)
        clean_predictions = clean_outputs.argmax(dim=1)
    clean_correct = clean_predictions.eq(diag_labels)
    clean_correct_count = int(clean_correct.sum().item())
    total_samples = len(diag_labels)
    clean_accuracy = 100.0 * clean_correct_count / total_samples if total_samples > 0 else 0.0

    # Guard: Zero clean-correct edge case
    if clean_correct_count == 0:
        return {
            "status": "INCONCLUSIVE",
            "summary": "Inconclusive: 0 evaluated samples were correctly classified on clean inputs. Attack success rate cannot be computed.",
            "warning_reasons": ["No clean-correct samples available for diagnostic evaluation."],
            "actionable_guidance": ["Evaluate on a larger sample size or verify that dataset labels match model classes."],
            "signals": {},
            "metadata": {
                "sample_count": total_samples,
                "clean_correct_count": 0,
                "clean_accuracy": 0.0,
                "epsilon": round(float(epsilon), 6),
                "pgd_steps": pgd_steps,
                "device": str(device),
            },
            "limitations": [
                "Diagnostic requires at least one correctly classified clean image to compute Attack Success Rate (ASR).",
            ],
        }

    # If baseline ASRs were not provided, compute them on the diagnostic batch
    if baseline_fgsm_asr is None:
        adv_fgsm = fgsm_attack(model, diag_images, diag_labels, epsilon=epsilon)
        with torch.no_grad():
            f_preds = model(adv_fgsm).argmax(dim=1)
        fgsm_fooled = clean_correct & ~f_preds.eq(diag_labels)
        baseline_fgsm_asr = 100.0 * float(fgsm_fooled.sum().item()) / clean_correct_count

    if baseline_pgd_asr is None:
        adv_pgd = pgd_attack(
            model=model,
            images=diag_images,
            labels=diag_labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=pgd_steps,
            random_start=True,
        )
        with torch.no_grad():
            p_preds = model(adv_pgd).argmax(dim=1)
        pgd_fooled = clean_correct & ~p_preds.eq(diag_labels)
        baseline_pgd_asr = 100.0 * float(pgd_fooled.sum().item()) / clean_correct_count

    # Signal 1: Input Gradient Health
    sig_grad = compute_gradient_health(model, diag_images, diag_labels)

    # Signal 2: 1D Loss Landscape Ray
    sig_ray = evaluate_loss_landscape_ray(
        model=model,
        images=diag_images,
        labels=diag_labels,
        clean_correct_mask=clean_correct,
        epsilon=epsilon,
    )

    # Signal 3: PGD Restart Sensitivity
    sig_restarts = evaluate_pgd_restarts(
        model=model,
        images=diag_images,
        labels=diag_labels,
        clean_correct_mask=clean_correct,
        epsilon=epsilon,
        alpha=alpha,
        steps=pgd_steps,
        restarts=pgd_restarts,
        baseline_pgd_asr=baseline_pgd_asr,
    )

    # Signal 4: Attack Strength Consistency
    sig_strength = evaluate_attack_strength_consistency(
        model=model,
        images=diag_images,
        labels=diag_labels,
        clean_correct_mask=clean_correct,
        epsilon=epsilon,
        alpha=alpha,
        max_steps=max(20, pgd_steps),
    )

    # Signal 5: FGSM vs PGD Discrepancy
    sig_fgsm_pgd = evaluate_fgsm_vs_pgd(
        fgsm_asr=baseline_fgsm_asr,
        pgd_asr=baseline_pgd_asr,
    )

    # Signal 6: Random Search Baseline
    if run_random_search:
        sig_rand = evaluate_random_search_baseline(
            model=model,
            images=diag_images,
            labels=diag_labels,
            clean_correct_mask=clean_correct,
            epsilon=epsilon,
            pgd_asr=baseline_pgd_asr,
        )
    else:
        sig_rand = {
            "status": "NOT_RUN",
            "random_search_success_rate": 0.0,
            "pgd_success_rate": round(baseline_pgd_asr, 2),
            "details": "Random search baseline was disabled by configuration.",
        }

    signals = {
        "gradient_health": sig_grad,
        "loss_landscape_ray": sig_ray,
        "pgd_restart_sensitivity": sig_restarts,
        "attack_strength_consistency": sig_strength,
        "fgsm_vs_pgd": sig_fgsm_pgd,
        "gradient_free_baseline": sig_rand,
    }

    # Aggregate warnings and guidance
    warning_reasons = []
    actionable_guidance = []

    if sig_grad["status"] in ("SUSPICIOUS", "ERROR"):
        warning_reasons.append(f"Gradient Health: {sig_grad['details']}")
        actionable_guidance.append(
            "Gradients are near-zero or disconnected. If using non-differentiable preprocessing, "
            "apply BPDA (Backward Pass Differentiable Approximation)."
        )

    if sig_ray["status"] == "SUSPICIOUS":
        warning_reasons.append(f"Loss Landscape: {sig_ray['details']}")
        actionable_guidance.append(
            "Loss decreases along the gradient direction. The local gradient vector is misleading; "
            "evaluate with adaptive step sizes or zeroth-order optimization."
        )

    if sig_restarts["status"] == "SUSPICIOUS":
        warning_reasons.append(f"PGD Restarts: {sig_restarts['details']}")
        actionable_guidance.append(
            "High variance across random restarts suggests local minima traps. Increase random restart count (e.g. 10-20 restarts)."
        )

    if sig_strength["status"] == "SUSPICIOUS":
        warning_reasons.append(f"Attack Strength: {sig_strength['details']}")
        actionable_guidance.append(
            "Attack success drops as perturbation effort grows. Check optimizer step size decay or boundary clamping issues."
        )

    if sig_fgsm_pgd["status"] == "SUSPICIOUS":
        warning_reasons.append(f"FGSM vs PGD: {sig_fgsm_pgd['details']}")
        actionable_guidance.append(
            "One-step attack outperformed iterative attack. Review PGD step size (alpha) and investigate potential shattered gradients."
        )

    if sig_rand["status"] == "SUSPICIOUS":
        warning_reasons.append(f"Random Exploration: {sig_rand['details']}")
        actionable_guidance.append(
            "Pure random perturbations outperformed gradient descent. Perform robust evaluation using decision-based attacks (e.g., Square Attack or RayS)."
        )

    # Determine overall status
    if any(s.get("status") == "ERROR" for s in signals.values()):
        status = "WARNING"
        summary = (
            f"Potential gradient masking or numerical instability: {len(warning_reasons)} warning(s) flagged. "
            "Gradient-based evaluations may not provide an accurate assessment of model robustness."
        )
    elif len(warning_reasons) > 0:
        status = "WARNING"
        summary = (
            f"Potential gradient masking: {len(warning_reasons)} suspicious signal(s) observed. "
            "The evaluated gradient-based attacks may be underestimating the model's true vulnerability."
        )
    else:
        status = "NO_STRONG_SIGNAL"
        summary = (
            "No strong gradient-masking signal detected across the evaluated diagnostics. "
            "This does not prove the model is robust or that masking is entirely absent."
        )

    return {
        "status": status,
        "summary": summary,
        "warning_reasons": warning_reasons,
        "actionable_guidance": list(dict.fromkeys(actionable_guidance)),
        "signals": signals,
        "metadata": {
            "sample_count": total_samples,
            "clean_correct_count": clean_correct_count,
            "clean_accuracy": round(clean_accuracy, 2),
            "epsilon": round(float(epsilon), 6),
            "pgd_steps": pgd_steps,
            "device": str(device),
        },
        "limitations": [
            "This is a heuristic diagnostic and does not prove the presence or absence of gradient masking.",
            "Results apply strictly to the evaluated sample batch, threat model (Linf), and attack configurations.",
        ],
    }

