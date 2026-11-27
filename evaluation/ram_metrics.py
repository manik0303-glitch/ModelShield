"""
ModelShield RAM++ Robustness Metrics
=====================================
Multi-label tag-based evaluation metrics that replace CIFAR-10 accuracy
for all RAM++ adversarial evaluations.

Metric conventions:
  - All percentage values are in [0.0, 100.0].
  - "baseline" = tags/scores from the clean (unperturbed) image.
  - "attacked" = tags/scores from the adversarially perturbed image.
  - Only active tags (above RAM++'s internal threshold) are considered
    for tag-set metrics; full sigmoid score tensors are used for score metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import torch


# ---------------------------------------------------------------------------
# Tag-set metrics
# ---------------------------------------------------------------------------

def tag_preservation_rate(
    baseline_tags: List[str],
    attacked_tags: List[str],
) -> float:
    """
    Fraction of baseline active tags that survive in the attacked output.
    Returns 100.0 if baseline has no active tags (trivially preserved).

    Higher is better (attack failed to remove tags).
    """
    baseline_set: Set[str] = set(baseline_tags)
    attacked_set: Set[str] = set(attacked_tags)
    if not baseline_set:
        return 100.0
    preserved = baseline_set & attacked_set
    return 100.0 * len(preserved) / len(baseline_set)


def tag_disappearance_count(
    baseline_tags: List[str],
    attacked_tags: List[str],
) -> int:
    """Number of baseline tags lost after the attack."""
    baseline_set = set(baseline_tags)
    attacked_set = set(attacked_tags)
    return len(baseline_set - attacked_set)


def tag_introduction_count(
    baseline_tags: List[str],
    attacked_tags: List[str],
) -> int:
    """Number of new (hallucinated) tags introduced by the attack."""
    baseline_set = set(baseline_tags)
    attacked_set = set(attacked_tags)
    return len(attacked_set - baseline_set)


def tag_introduction_rate(
    baseline_tags: List[str],
    attacked_tags: List[str],
) -> float:
    """
    Fraction of attacked tags that are NEW (not in baseline).
    Returns 0.0 if attacked_tags is empty.

    Lower is better (attack didn't hallucinate new tags).
    """
    attacked_set = set(attacked_tags)
    if not attacked_set:
        return 0.0
    baseline_set = set(baseline_tags)
    new_tags = attacked_set - baseline_set
    return 100.0 * len(new_tags) / len(attacked_set)


def jaccard_similarity(
    baseline_tags: List[str],
    attacked_tags: List[str],
) -> float:
    """
    Jaccard similarity between baseline and attacked tag sets.
    Returns 1.0 if both are empty.
    """
    a, b = set(baseline_tags), set(attacked_tags)
    if not a and not b:
        return 100.0
    union = a | b
    intersection = a & b
    return 100.0 * len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Score-level metrics
# ---------------------------------------------------------------------------

def mean_score_delta(
    baseline_scores: torch.Tensor,
    attacked_scores: torch.Tensor,
    active_indices: Optional[List[int]] = None,
) -> float:
    """
    Mean absolute change in sigmoid scores across all (or active) tags.

    Args:
        baseline_scores: [num_classes] float tensor
        attacked_scores: [num_classes] float tensor
        active_indices:  if provided, compute only over these tag indices

    Returns:
        Mean absolute score delta in [0.0, 1.0].
    """
    baseline_scores = baseline_scores.float().cpu()
    attacked_scores = attacked_scores.float().cpu()
    delta = (attacked_scores - baseline_scores).abs()
    if active_indices is not None and len(active_indices) > 0:
        delta = delta[torch.tensor(active_indices)]
    return float(delta.mean().item())


def per_tag_score_deltas(
    baseline_scores: torch.Tensor,
    attacked_scores: torch.Tensor,
    baseline_tags: List[str],
    tag_list: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Per-tag score change for all ACTIVE baseline tags.

    Returns dict: {tag_name: {"baseline": float, "attacked": float, "delta": float}}
    """
    tag_index = {t: i for i, t in enumerate(tag_list)}
    result = {}
    baseline_cpu = baseline_scores.float().cpu()
    attacked_cpu = attacked_scores.float().cpu()
    for tag in baseline_tags:
        idx = tag_index.get(tag)
        if idx is None:
            continue
        b_score = float(baseline_cpu[idx].item())
        a_score = float(attacked_cpu[idx].item())
        result[tag] = {
            "baseline": round(b_score, 4),
            "attacked": round(a_score, 4),
            "delta": round(a_score - b_score, 4),
        }
    return result


def max_score_suppression(
    baseline_scores: torch.Tensor,
    attacked_scores: torch.Tensor,
    baseline_tags: List[str],
    tag_list: List[str],
) -> Tuple[str, float]:
    """
    Find the baseline tag that experienced the largest score drop.
    Returns (tag_name, score_drop) where score_drop > 0 means suppression.
    """
    tag_index = {t: i for i, t in enumerate(tag_list)}
    baseline_cpu = baseline_scores.float().cpu()
    attacked_cpu = attacked_scores.float().cpu()
    worst_tag, worst_drop = "", 0.0
    for tag in baseline_tags:
        idx = tag_index.get(tag)
        if idx is None:
            continue
        drop = float((baseline_cpu[idx] - attacked_cpu[idx]).item())
        if drop > worst_drop:
            worst_drop = drop
            worst_tag = tag
    return worst_tag, round(worst_drop, 4)


# ---------------------------------------------------------------------------
# Perturbation validity checks (replaces CIFAR-10 trust checks)
# ---------------------------------------------------------------------------

def check_perturbation_budget_ram(
    original_tensor: torch.Tensor,
    adversarial_tensor: torch.Tensor,
    epsilon: float,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Check that Linf perturbation stays within epsilon budget."""
    if original_tensor.shape != adversarial_tensor.shape:
        return {"passed": False, "message": "Shape mismatch between original and adversarial tensors."}
    max_pert = (adversarial_tensor - original_tensor).abs().max().item()
    passed = max_pert <= epsilon + tolerance
    return {
        "passed": passed,
        "max_perturbation": round(max_pert, 6),
        "epsilon": round(epsilon, 6),
        "message": f"Max Linf perturbation: {max_pert:.6f} (budget: {epsilon:.6f})",
    }


def check_pixel_range_ram(tensor: torch.Tensor) -> Dict[str, Any]:
    """Check that pixel values stay in [0, 1]."""
    mn, mx = float(tensor.min().item()), float(tensor.max().item())
    passed = mn >= 0.0 and mx <= 1.0
    return {
        "passed": passed,
        "min": round(mn, 6),
        "max": round(mx, 6),
        "message": f"Pixel range: [{mn:.4f}, {mx:.4f}]",
    }


def check_scores_finite_ram(scores: torch.Tensor) -> Dict[str, Any]:
    """Check that RAM++ sigmoid scores are all finite and in [0, 1]."""
    finite = bool(torch.isfinite(scores).all().item())
    in_range = bool((scores >= 0.0).all().item() and (scores <= 1.0).all().item())
    passed = finite and in_range
    return {
        "passed": passed,
        "finite": finite,
        "in_range_0_1": in_range,
        "message": "Scores are finite and in [0, 1]." if passed else "Score tensor failed validity check.",
    }


# ---------------------------------------------------------------------------
# Full per-image attack outcome summary
# ---------------------------------------------------------------------------

def compute_attack_outcome(
    baseline_tags: List[str],
    attacked_tags: List[str],
    baseline_scores: torch.Tensor,
    attacked_scores: torch.Tensor,
    tag_list: List[str],
    epsilon: float,
    original_tensor: Optional[torch.Tensor] = None,
    adversarial_tensor: Optional[torch.Tensor] = None,
    attack_name: str = "attack",
) -> Dict[str, Any]:
    """
    Compute the full set of RAM++ robustness metrics for a single image attack.

    Returns a structured dict suitable for display, audit logs, and regression snapshots.
    """
    pres = tag_preservation_rate(baseline_tags, attacked_tags)
    disapp = tag_disappearance_count(baseline_tags, attacked_tags)
    intro = tag_introduction_count(baseline_tags, attacked_tags)
    intro_rate = tag_introduction_rate(baseline_tags, attacked_tags)
    jac = jaccard_similarity(baseline_tags, attacked_tags)
    score_delta = mean_score_delta(baseline_scores, attacked_scores)

    # Active tag indices for score-delta calculation
    tag_index = {t: i for i, t in enumerate(tag_list)}
    active_indices = [tag_index[t] for t in baseline_tags if t in tag_index]
    active_score_delta = mean_score_delta(baseline_scores, attacked_scores, active_indices)

    worst_tag, worst_drop = max_score_suppression(
        baseline_scores, attacked_scores, baseline_tags, tag_list
    )
    per_tag = per_tag_score_deltas(
        baseline_scores, attacked_scores, baseline_tags, tag_list
    )

    # Attack success: any baseline tag was lost
    attack_succeeded = disapp > 0

    checks: Dict[str, Any] = {}
    if original_tensor is not None and adversarial_tensor is not None:
        checks["perturbation_budget"] = check_perturbation_budget_ram(
            original_tensor, adversarial_tensor, epsilon
        )
        checks["pixel_range"] = check_pixel_range_ram(adversarial_tensor)
    checks["scores_finite"] = check_scores_finite_ram(attacked_scores)

    return {
        "attack_name": attack_name,
        "epsilon": round(epsilon, 6),
        # Tag-set metrics
        "baseline_tags": baseline_tags,
        "attacked_tags": attacked_tags,
        "tag_preservation_rate": round(pres, 2),
        "tag_disappearance_count": disapp,
        "tag_introduction_count": intro,
        "tag_introduction_rate": round(intro_rate, 2),
        "jaccard_similarity": round(jac, 2),
        "attack_succeeded": attack_succeeded,
        # Score metrics
        "mean_score_delta_all": round(score_delta, 6),
        "mean_score_delta_active": round(active_score_delta, 6),
        "worst_suppressed_tag": worst_tag,
        "worst_score_drop": worst_drop,
        "per_tag_score_deltas": per_tag,
        # Integrity checks
        "trust_checks": checks,
    }


# ---------------------------------------------------------------------------
# Regression gate metric schema for RAM++
# ---------------------------------------------------------------------------

RAM_METRIC_DIRECTIONS = {
    # Higher is better
    "tag_preservation_rate": "HIGHER_IS_BETTER",
    "jaccard_similarity": "HIGHER_IS_BETTER",
    # Lower is better
    "tag_disappearance_count": "LOWER_IS_BETTER",
    "tag_introduction_count": "LOWER_IS_BETTER",
    "tag_introduction_rate": "LOWER_IS_BETTER",
    "mean_score_delta_active": "LOWER_IS_BETTER",
    "mean_score_delta_all": "LOWER_IS_BETTER",
    "fgsm_attack_success_rate": "LOWER_IS_BETTER",
    "pgd_attack_success_rate": "LOWER_IS_BETTER",
}
