"""
Orchestrator-local RAM++ tag-suppression attacks (FGSM / PGD).

Why this exists: the attack must differentiate the SAME function that is
evaluated.  `resolve_scoring_fn` therefore only accepts a differentiable
scoring path whose clean output reproduces the measured clean RAM++ scores.
If no candidate qualifies, preflight fails explicitly instead of attacking a
surrogate that disagrees with the evaluated model.

The perturbation lives in RAW PIXEL space ([0, 1]); the budget is therefore an
exact pixel-space L-infinity bound.  Loss: BCE with target 1 on the baseline
active tags, ascended (pushes the active tag scores down).
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from orchestrator.ram_space import to_model_space

ScoringFn = Callable[[torch.Tensor], torch.Tensor]


def _factory_generate_tag(model, device) -> ScoringFn:
    def fn(x: torch.Tensor) -> torch.Tensor:
        _, _, scores = model.generate_tag(x, return_scores=True)
        return scores[0]
    return fn


def _factory_fc_hook(model, device) -> ScoringFn:
    from model.ram_adapter import _fc_hook_scores

    def fn(x: torch.Tensor) -> torch.Tensor:
        return _fc_hook_scores(model, x.to(device))
    return fn


def _factory_replica(model, device) -> ScoringFn:
    from model.ram_adapter import _ramplus_replica_scores

    def fn(x: torch.Tensor) -> torch.Tensor:
        return _ramplus_replica_scores(model, x.to(device))
    return fn


def _factory_adapter(model, device) -> ScoringFn:
    from model.ram_adapter import forward_with_grad

    def fn(x: torch.Tensor) -> torch.Tensor:
        return forward_with_grad(x, model=model, device=device)
    return fn


def resolve_scoring_fn(model, device, x_norm: torch.Tensor, ref_scores: torch.Tensor,
                       tol: float = 1e-3) -> Tuple[Optional[ScoringFn], Dict[str, Any]]:
    """Pick the first differentiable scoring path that reproduces `ref_scores`."""
    ref = ref_scores.detach().float().cpu().reshape(-1)
    report: Dict[str, Any] = {"candidates": [], "tolerance": tol, "passed": False, "name": None}
    for name, factory in (
        ("model.generate_tag(return_scores=True)", _factory_generate_tag),
        ("model.fc logits captured during generate_tag", _factory_fc_hook),
        ("RAM++ tagging path re-implemented with autograd", _factory_replica),
        ("ram_adapter.forward_with_grad", _factory_adapter),
    ):
        entry: Dict[str, Any] = {"name": name, "passed": False}
        fn = None
        try:
            fn = factory(model, device)
            x = x_norm.detach().to(device).float().clone().requires_grad_(True)
            with torch.enable_grad():
                s = fn(x)
                # requires_grad alone is not enough (parameters also require grad): the INPUT
                # must receive a non-zero, finite gradient.
                g = torch.autograd.grad(s.sum(), x, allow_unused=True)[0] if s.requires_grad else None
            s_cpu = s.detach().float().cpu().reshape(-1)
            entry["input_grad_linf"] = None if g is None else float(g.abs().max().item())
            entry["differentiable"] = bool(g is not None and bool(torch.isfinite(g).all().item())
                                           and entry["input_grad_linf"] > 0.0)
            entry["shape_ok"] = tuple(s_cpu.shape) == tuple(ref.shape)
            entry["finite"] = bool(torch.isfinite(s_cpu).all().item())
            if entry["shape_ok"]:
                entry["max_abs_diff"] = float((s_cpu - ref).abs().max().item())
            entry["passed"] = bool(
                entry["differentiable"] and entry["shape_ok"] and entry["finite"]
                and entry.get("max_abs_diff", float("inf")) <= tol
            )
        except Exception as exc:  # candidate unusable; record why
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["candidates"].append(entry)
        if entry["passed"]:
            report.update(passed=True, name=name, max_abs_diff=entry["max_abs_diff"])
            return fn, report
    report["message"] = "; ".join(
        f"{c['name']}: " + (c.get("error") or
                            f"input_gradient_ok={c.get('differentiable')} (|grad|max={c.get('input_grad_linf')}), shape_ok={c.get('shape_ok')}, "
                            f"max|diff|={c.get('max_abs_diff')}")
        for c in report["candidates"]
    )
    return None, report


def tag_suppression_attack(scoring_fn: ScoringFn, pixel_clean: torch.Tensor,
                           active_idx: List[int], epsilon: float, alpha: float,
                           steps: int, device) -> Dict[str, Any]:
    """
    Sign-gradient ascent on -log(score) of the baseline active tags.
    FGSM = steps 1 with alpha == epsilon.  PGD = `steps` steps of size `alpha`,
    projected onto the pixel-space epsilon ball and [0, 1] after every step
    (deterministic start at the clean image).

    Returns the adversarial RAW-PIXEL tensor [1, 3, H, W] on CPU plus a trace.
    """
    if not active_idx:
        raise ValueError("No baseline active tags: nothing to suppress.")
    x0 = pixel_clean.detach().to(device).float()
    idx = torch.as_tensor(active_idx, dtype=torch.long, device=device)
    x_adv = x0.clone()
    loss_trace: List[float] = []
    grad_linf_trace: List[float] = []
    for _ in range(int(steps)):
        x_var = x_adv.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            scores = scoring_fn(to_model_space(x_var))
            loss = -torch.log(scores[idx].clamp_min(1e-6)).mean()
            (grad,) = torch.autograd.grad(loss, x_var)
        loss_trace.append(float(loss.detach().item()))
        grad_linf_trace.append(float(grad.abs().max().item()))
        x_adv = x_adv.detach() + float(alpha) * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x0 - epsilon), x0 + epsilon).clamp(0.0, 1.0)
    return {
        "adversarial_pixels": x_adv.detach().cpu(),
        "loss_trace": loss_trace,
        "grad_linf_trace": grad_linf_trace,
        "gradient_nonzero": any(g > 0.0 for g in grad_linf_trace),
    }
