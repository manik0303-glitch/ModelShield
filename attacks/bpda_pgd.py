"""
attacks/bpda_pgd.py
-------------------
BPDA-assisted PGD (Backward Pass Differentiable Approximation).

BPDA (Athalye et al., ICML 2018) approximates the gradient through a
non-differentiable transformation by replacing the true backward pass
with a differentiable substitute (here: identity / straight-through).

The forward pass uses the EXACT same transformation as standard attacks.
Only the backward gradient computation uses the approximation.

Threat model: Linf, same epsilon/alpha/steps convention as pgd_attack.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class _StraightThroughQuantize(torch.autograd.Function):
    """
    Custom autograd function implementing quantization with a
    straight-through gradient estimator (BPDA approximation).

    Forward:  x -> round(x * (L-1)) / (L-1)   [hard quantization]
    Backward: passes upstream gradient unchanged (identity approximation)

    LIMITATION: This is NOT the true derivative of rounding.
    It is the BPDA straight-through approximation. Results should be
    interpreted as an attack against an approximate gradient, not a
    certified lower bound.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, levels: int) -> torch.Tensor:
        ctx.save_for_backward(x)
        x_clamped = x.clamp(0.0, 1.0)
        quantized = torch.round(x_clamped * (levels - 1)) / (levels - 1)
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through: pass gradient through unchanged
        # Return gradient for x; None for levels (not a tensor)
        return grad_output, None


def straight_through_quantize(x: torch.Tensor, levels: int) -> torch.Tensor:
    """
    Quantization with straight-through backward approximation (BPDA).
    Forward: identical to hard quantization.
    Backward: identity (straight-through estimator).
    """
    if levels < 2:
        raise ValueError("quantization levels must be at least 2")
    return _StraightThroughQuantize.apply(x, levels)


def bpda_pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    quantization_levels: int = 16,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
    random_seed: Optional[int] = None,
) -> torch.Tensor:
    """
    PGD with BPDA straight-through approximation through hard quantization.

    The forward path applies REAL quantization (same as standard attacks).
    The backward path uses the straight-through estimator instead of the
    true (zero/undefined) gradient of rounding.

    Returns adversarial images in [0, 1] that do NOT mutate `images`.
    """
    was_training = model.training
    model.eval()

    images = images.detach()
    labels = labels.detach()
    original_images = images.clone().detach()

    if random_seed is not None:
        torch.manual_seed(random_seed)

    if random_start:
        adv = original_images + torch.empty_like(original_images).uniform_(
            -epsilon, epsilon
        )
        adv = adv.clamp(0.0, 1.0)
    else:
        adv = original_images.clone()

    best_adv = adv.clone()
    best_loss = torch.full((images.shape[0],), -float("inf"), device=images.device)

    try:
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)

            # Forward: apply BPDA quantization (real quant + ST backward)
            adv_quantized = straight_through_quantize(adv, levels=quantization_levels)

            model.zero_grad(set_to_none=True)
            outputs = model(adv_quantized)
            loss_per_sample = F.cross_entropy(outputs, labels, reduction="none")
            loss = loss_per_sample.mean()

            grads = torch.autograd.grad(loss, adv, allow_unused=True)[0]

            if grads is None:
                # BPDA failed to produce gradients; return best found so far
                break

            adv = adv.detach() + alpha * grads.sign()
            perturbation = (adv - original_images).clamp(-epsilon, epsilon)
            adv = (original_images + perturbation).clamp(0.0, 1.0)

            # Track best (highest-loss) adversarial candidates
            with torch.no_grad():
                adv_q = straight_through_quantize(adv, levels=quantization_levels)
                step_outputs = model(adv_q)
                step_loss = F.cross_entropy(step_outputs, labels, reduction="none")
                improved = step_loss > best_loss
                best_adv[improved] = adv[improved]
                best_loss[improved] = step_loss[improved]

    finally:
        if was_training:
            model.train()

    return best_adv.detach()

