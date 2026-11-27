"""
ModelShield Adaptive Red-Team Orchestrator Engine
Coordinates preflight checks, baseline evaluation, adaptive attack escalation,
defense evaluation, budget accounting, and evidence collection.

Supports both CIFAR-10 classification mode (legacy) and RAM++ multi-label
image-tagging mode (default for per-image investigations).
"""
import base64
import io
import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import math

import numpy as np
from PIL import Image
import torch

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from defenses.defense_methods import quantization_defense
from evaluation.defense_evaluation import calculate_accuracy, calculate_attack_success
from evaluation.trust_checks import (
    check_metric_consistency,
    check_perturbation_budget,
    check_pixel_range,
    check_prediction_outputs,
)
from evaluation.trust_flag import generate_trust_flag
from orchestrator.policy import AdaptiveDecisionPolicy
from orchestrator.preflight import run_preflight_checks
from orchestrator.types import (
    DecisionAction,
    DecisionRecord,
    EvaluationPhase,
    FinalAssessment,
    PreflightResult,
    StepEvaluationResult,
)

# ── RAM++ imports ─────────────────────────────────────────────────────────────
# The RAM++ attacks now live in orchestrator/ram_attacks.py (verified differentiable
# path, exact pixel-space budget), so attacks/ram_fgsm.py is no longer required here.
# A missing module must NOT silently demote a RAM++ mission to the CIFAR-10 path.
try:
    from evaluation.ram_metrics import compute_attack_outcome, check_scores_finite_ram
    from orchestrator.preflight import run_ram_preflight_checks
    from orchestrator.ram_attacks import resolve_scoring_fn, tag_suppression_attack
    from orchestrator.ram_space import (
        check_budget_pixel,
        check_pixel_range_tolerant,
        to_model_space,
        to_pixel_space,
    )
    _RAM_MODULES_AVAILABLE = True
    _RAM_IMPORT_ERROR = ""
except ImportError as _ram_exc:  # pragma: no cover
    _RAM_MODULES_AVAILABLE = False
    _RAM_IMPORT_ERROR = str(_ram_exc)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def tensor_to_base64_png(tensor: torch.Tensor, size=(128, 128), smooth: bool = False) -> str:
    """Convert a [3, H, W] RAW-PIXEL float tensor in [0, 1] to base64 PNG data URI.

    Pass raw pixels (see ram_space.to_pixel_space); normalised RAM++ inputs would be clamped to garbage.
    """
    try:
        t = tensor.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)
        arr = (t.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr).resize(size, Image.BILINEAR if smooth else Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def diff_heatmap_to_base64_png(clean: torch.Tensor, adv: torch.Tensor, epsilon: float, size=(128, 128)) -> str:
    """Generate colorized perturbation heatmap base64 PNG using inferno colormap."""
    try:
        c = clean.detach().cpu().squeeze()
        a = adv.detach().cpu().squeeze()
        diff = (a - c).abs().mean(dim=0).clamp(0.0, epsilon) / max(epsilon, 1e-6)
        diff_np = diff.numpy()
        try:
            import matplotlib.cm as cm
            rgba = cm.inferno(diff_np)
            arr = (rgba[:, :, :3] * 255).astype(np.uint8)
        except Exception:
            val = np.clip(diff_np, 0.0, 1.0)
            r = np.clip(val * 2.0, 0.0, 1.0)
            g = np.clip(val * 2.0 - 0.5, 0.0, 1.0)
            b = np.clip(val * 3.0 - 2.0, 0.0, 1.0)
            arr = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

        img = Image.fromarray(arr).resize(size, Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def get_diff_matrix_2d(clean: torch.Tensor, adv: torch.Tensor, pool: int = 1) -> List[List[float]]:
    """Return channel-averaged perturbation matrix for Plotly heatmap.

    pool > 1 average-pools the matrix (e.g. 384x384 -> 48x48 with pool=8) to keep the audit JSON small.
    """
    try:
        c = clean.detach().cpu().squeeze()
        a = adv.detach().cpu().squeeze()
        diff = (a - c).abs().mean(dim=0)
        if pool > 1:
            diff = torch.nn.functional.avg_pool2d(diff[None, None], pool)[0, 0]
        diff = diff.numpy()
        return [[round(float(v), 5) for v in row] for row in diff]
    except Exception:
        return []


def extract_conv1_features(model: torch.nn.Module, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Extract intermediate feature map from the first Conv2d layer."""
    activation = {}

    def hook(m, inp, out):
        activation["features"] = out.detach().cpu()

    hook_handle = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hook_handle = module.register_forward_hook(hook)
            break

    try:
        with torch.no_grad():
            t = image_tensor
            if t.dim() == 3:
                t = t.unsqueeze(0)
            _ = model(t)
    except Exception:
        return None
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if "features" in activation:
        f = activation["features"].squeeze(0).mean(dim=0)
        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            f_norm = (f - f_min) / (f_max - f_min)
        else:
            f_norm = torch.zeros_like(f)
        return f_norm
    return None


def feature_map_to_base64_png(fmap: Optional[torch.Tensor], colormap: str = "viridis", size=(128, 128)) -> str:
    """Convert a 2D float tensor [H, W] in [0, 1] to base64 PNG."""
    if fmap is None:
        return ""
    try:
        arr = fmap.detach().cpu().clamp(0.0, 1.0).numpy()
        try:
            import matplotlib.cm as cm
            rgba = getattr(cm, colormap, cm.viridis)(arr)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        except Exception:
            val = np.clip(arr, 0.0, 1.0)
            rgb = (np.stack([val, val, val], axis=-1) * 255).astype(np.uint8)

        img = Image.fromarray(rgb).resize(size, Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# RAM++ helpers: genuine internal activations, display formatting, JSON safety
# ─────────────────────────────────────────────────────────────────────────────
class _ActivationProbe:
    """
    Captures real activations from the SAME forward pass that produces the tags:
      * patch_embed : output of the first Conv2d inside model.visual_encoder
                      (Swin patch-embedding convolution), mean |activation| over channels.
      * stage4      : visual-encoder output tokens, L2 norm per spatial token, reshaped to a grid.
    Nothing is synthesised; if a hook does not fire the map is simply None.
    """

    def __init__(self, model):
        self.model = model
        self._handles: list = []
        self._conv = None
        self._enc = None

    def __enter__(self):
        enc = getattr(self.model, "visual_encoder", None)
        if enc is not None:
            conv = next((m for m in enc.modules() if isinstance(m, torch.nn.Conv2d)), None)
            if conv is not None:
                self._handles.append(conv.register_forward_hook(self._on_conv))
            self._handles.append(enc.register_forward_hook(self._on_enc))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def _on_conv(self, module, inp, out):
        if self._conv is None and isinstance(out, torch.Tensor):
            self._conv = out.detach().float().cpu()

    def _on_enc(self, module, inp, out):
        if self._enc is None:
            o = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(o, torch.Tensor):
                self._enc = o.detach().float().cpu()

    def maps(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {"patch_embed": None, "patch_embed_shape": None, "stage4": None, "stage4_grid": None}
        c = self._conv
        if c is not None and c.dim() == 4:
            res["patch_embed"] = c[0].abs().mean(dim=0)
            res["patch_embed_shape"] = [int(c.shape[1]), int(c.shape[2]), int(c.shape[3])]
        e = self._enc
        if e is not None and e.dim() == 3:
            n = int(e.shape[1])
            side = math.isqrt(n)
            tokens = None
            if side * side == n:
                tokens = e[0]
            else:
                side = math.isqrt(n - 1) if n > 1 else 0
                if side > 0 and side * side == n - 1:
                    tokens = e[0, 1:]  # drop the leading global (CLS-like) token
            if tokens is not None:
                res["stage4"] = tokens.norm(dim=-1).reshape(side, side)
                res["stage4_grid"] = [side, side]
        return res


def _norm_map(m: torch.Tensor, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Min-max scale to [0, 1], optionally using a reference map's range (keeps clean/adv comparable)."""
    r = ref if ref is not None else m
    lo, hi = float(r.min()), float(r.max())
    if hi <= lo:
        return torch.zeros_like(m)
    return ((m - lo) / (hi - lo)).clamp(0.0, 1.0)


def _json_default(o: Any):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ─────────────────────────────────────────────────────────────────────────────
# RAM++ helper: format tag list for display
# ─────────────────────────────────────────────────────────────────────────────
def _format_tags(tags: List[str], max_n: int = 8) -> str:
    if not tags:
        return "(none)"
    shown = tags[:max_n]
    suffix = f" … +{len(tags) - max_n} more" if len(tags) > max_n else ""
    return ", ".join(shown) + suffix


class RedTeamOrchestrator:
    """
    Coordinates an automated, adaptive adversarial security evaluation.

    Supports two modes:
      - RAM++ mode (ram_mode=True, default when investigation is loaded):
          Evaluates multi-label image-tagging robustness using BCE-based attacks.
          `images`  = pixel tensor [1, 3, 384, 384] (the shared investigation image)
          `labels`  = baseline tag scores tensor [1, num_tags] from RAM++ baseline
          `tag_list` = list of all RAM++ tag strings (len = num_tags)
          `baseline_tags` = list of active tag strings at baseline threshold
      - CIFAR-10 mode (ram_mode=False, legacy):
          `images`  = [N, 3, 32, 32] float tensors
          `labels`  = [N] integer class labels
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        images: torch.Tensor,
        labels: torch.Tensor,
        config: Optional[Dict[str, Any]] = None,
        max_budget: int = 5,
        model_name: str = "RAM++ Swin-L",
        dataset_name: str = "Single Image Investigation",
        results_dir: str = "results/runs",
        # RAM++ mode extras
        ram_mode: bool = False,
        tag_list: Optional[List[str]] = None,
        baseline_tags: Optional[List[str]] = None,
        baseline_scores: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.device = device
        self.images = images
        self.labels = labels
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_budget = max_budget
        self.remaining_budget = max_budget
        self.results_dir = results_dir

        # RAM++ mode
        if ram_mode and not _RAM_MODULES_AVAILABLE:
            raise RuntimeError(f"RAM++ mode requested but RAM++ orchestrator modules failed to import: {_RAM_IMPORT_ERROR}")
        self.ram_mode = bool(ram_mode)
        self.tag_list = list(tag_list) if tag_list is not None else []
        self._tag_index: Dict[str, int] = {str(t): i for i, t in enumerate(self.tag_list)}
        self._scoring_fn = None
        self.scoring_report: Optional[Dict[str, Any]] = None
        self._stage_scores: Dict[str, torch.Tensor] = {}
        self._stage_tags: Dict[str, List[str]] = {}
        self._stage_maps: Dict[str, Dict[str, Any]] = {}
        self.baseline_tags = baseline_tags or []
        self.baseline_scores = baseline_scores  # [1, num_tags] tensor

        # Configuration defaults
        self.config = config or {}
        attacks_cfg = self.config.get("attacks", {})
        fgsm_cfg = attacks_cfg.get("fgsm", {})
        pgd_cfg = attacks_cfg.get("pgd", {})

        self.epsilon = float(fgsm_cfg.get("epsilon", 8 / 255))
        self.pgd_steps = int(pgd_cfg.get("steps", 10))
        self.pgd_alpha = float(pgd_cfg.get("step_size", 2 / 255))
        self.quantization_levels = int(self.config.get("quantization_levels", 16))

        # Lifecycle tracking
        self.run_id = f"ms-run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.current_phase = EvaluationPhase.INITIALIZED
        self.step_history: List[StepEvaluationResult] = []
        self.decision_history: List[DecisionRecord] = []
        self.completed_actions: List[str] = []
        self.metrics_by_action: Dict[str, Dict[str, Any]] = {}
        self.trust_flags_by_action: Dict[str, Dict[str, Any]] = {}
        self.anomalies_detected: List[str] = []
        self.preflight_result: Optional[PreflightResult] = None
        self.final_assessment: Optional[FinalAssessment] = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        # Subsystems
        self.policy = AdaptiveDecisionPolicy()

        # Cache for intermediate evaluation tensors
        self.clean_predictions: Optional[torch.Tensor] = None
        self.fgsm_adv_images: Optional[torch.Tensor] = None
        self.pgd_adv_images: Optional[torch.Tensor] = None
        self.sample_inspection: Dict[str, Any] = {}
        self.detailed_samples: List[Dict[str, Any]] = []
        self.class_metrics: Dict[str, Any] = {}
        self.flip_matrix: Dict[str, Dict[str, int]] = {}
        self.discoveries: List[Dict[str, Any]] = []

        # RAM++ cache
        self.ram_baseline_outcome: Optional[Dict[str, Any]] = None
        self.fgsm_outcome: Optional[Dict[str, Any]] = None
        self.pgd_outcome: Optional[Dict[str, Any]] = None
        self.fgsm_adv_scores: Optional[torch.Tensor] = None
        self.pgd_adv_scores: Optional[torch.Tensor] = None

        os.makedirs(self.results_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def run(self, progress_callback: Optional[Callable[[str, int, int], None]] = None) -> FinalAssessment:
        """Execute full adaptive red-team workflow from preflight to final assessment."""
        self.start_time = time.time()
        self.current_phase = EvaluationPhase.PREFLIGHT
        step_counter = 1

        while True:
            preflight_passed = self.preflight_result.passed if self.preflight_result else None
            preflight_errors = self.preflight_result.errors if self.preflight_result else []

            decision = self.policy.decide_next_action(
                step_number=step_counter,
                current_phase=self.current_phase,
                preflight_passed=preflight_passed,
                preflight_errors=preflight_errors,
                completed_actions=self.completed_actions,
                metrics_by_action=self.metrics_by_action,
                trust_flags_by_action=self.trust_flags_by_action,
                remaining_budget=self.remaining_budget,
                anomalies_detected=self.anomalies_detected,
            )
            self.decision_history.append(decision)

            if progress_callback:
                progress_callback(
                    f"Step {step_counter}: {decision.action.value}",
                    len(self.completed_actions),
                    self.max_budget,
                )

            if decision.action == DecisionAction.HALT_INTEGRITY_FAILURE:
                self.current_phase = EvaluationPhase.FAILED
                break
            elif decision.action == DecisionAction.STOP_BUDGET_EXHAUSTED:
                self.current_phase = EvaluationPhase.BUDGET_EXHAUSTED
                break
            elif decision.action == DecisionAction.STOP_SUFFICIENT_EVIDENCE:
                self.current_phase = EvaluationPhase.COMPLETED
                break

            self.remaining_budget = max(0, self.remaining_budget - decision.budget_cost)

            step_res = self._execute_action(decision.action, step_counter)
            self.step_history.append(step_res)
            self.completed_actions.append(decision.action.value)
            self.metrics_by_action[decision.action.value] = step_res.metrics
            self.trust_flags_by_action[decision.action.value] = step_res.trust_flag

            if step_res.status == "FAIL":
                self.anomalies_detected.append(f"Step {step_counter} failed: {step_res.error_message}")

            step_counter += 1

        self.end_time = time.time()
        self.final_assessment = self._generate_final_assessment()
        self._persist_run()
        return self.final_assessment

    # ─────────────────────────────────────────────────────────────────────────
    # Action dispatcher
    # ─────────────────────────────────────────────────────────────────────────
    def _execute_action(self, action: DecisionAction, step_number: int) -> StepEvaluationResult:
        step_start = time.time()
        timestamp = datetime.utcnow().isoformat() + "Z"

        # ── PREFLIGHT ──────────────────────────────────────────────────────
        if action == DecisionAction.RUN_PREFLIGHT:
            self.current_phase = EvaluationPhase.PREFLIGHT

            if self.ram_mode:
                # RAM++ inputs are ImageNet-normalised; the CIFAR-era preflight expects raw [0, 1]
                # pixels and single-label logits, so RAM++ gets its own non-destructive preflight.
                self.model.eval()
                self.preflight_result = run_ram_preflight_checks(
                    model=self.model,
                    normalized_images=self.images,
                    baseline_scores=self.baseline_scores,
                    baseline_tags=self.baseline_tags,
                    tag_list=self.tag_list,
                    epsilon=self.epsilon,
                    pgd_steps=self.pgd_steps,
                    pgd_alpha=self.pgd_alpha,
                    device=self.device,
                    differentiable_probe=self._probe_differentiable_path,
                )
            else:
                self.preflight_result = run_preflight_checks(
                    model=self.model,
                    images=self.images,
                    labels=self.labels,
                    epsilon=self.epsilon,
                    pgd_steps=self.pgd_steps,
                    pgd_alpha=self.pgd_alpha,
                    device=self.device,
                )

            status = "SUCCESS" if self.preflight_result.passed else "FAIL"
            err_msg = "; ".join(self.preflight_result.errors) if not self.preflight_result.passed else None
            trust_flag = {
                "status": "PASS" if self.preflight_result.passed else "FAIL",
                "summary": "Preflight checks passed." if self.preflight_result.passed else f"Preflight failed: {err_msg}",
            }

            return StepEvaluationResult(
                step_number=step_number,
                action_type=action.value,
                phase=EvaluationPhase.PREFLIGHT,
                metrics={"checks_total": len(self.preflight_result.checks), "passed": self.preflight_result.passed},
                trust_checks=self.preflight_result.checks,
                trust_flag=trust_flag,
                execution_time_seconds=time.time() - step_start,
                status=status,
                error_message=err_msg,
                timestamp=timestamp,
            )

        # ── BASELINE ──────────────────────────────────────────────────────
        elif action == DecisionAction.RUN_BASELINE:
            self.current_phase = EvaluationPhase.BASELINE

            if self.ram_mode:
                return self._run_ram_baseline(step_number, step_start, timestamp, action)
            else:
                return self._run_cifar_baseline(step_number, step_start, timestamp, action)

        # ── FGSM ──────────────────────────────────────────────────────────
        elif action == DecisionAction.RUN_FGSM:
            self.current_phase = EvaluationPhase.ATTACK_EVALUATION

            if self.ram_mode:
                return self._run_ram_fgsm(step_number, step_start, timestamp, action)
            else:
                return self._run_cifar_fgsm(step_number, step_start, timestamp, action)

        # ── PGD ───────────────────────────────────────────────────────────
        elif action == DecisionAction.ESCALATE_PGD:
            self.current_phase = EvaluationPhase.ATTACK_EVALUATION

            if self.ram_mode:
                return self._run_ram_pgd(step_number, step_start, timestamp, action)
            else:
                return self._run_cifar_pgd(step_number, step_start, timestamp, action)

        # ── DEFENSE ───────────────────────────────────────────────────────
        elif action == DecisionAction.TEST_DEFENSE:
            self.current_phase = EvaluationPhase.DEFENSE_VERIFICATION

            if self.ram_mode:
                return self._run_ram_defense(step_number, step_start, timestamp, action)
            else:
                return self._run_cifar_defense(step_number, step_start, timestamp, action)

        # ── REPEAT VERIFICATION ───────────────────────────────────────────
        elif action == DecisionAction.REPEAT_VERIFICATION:
            self.current_phase = EvaluationPhase.ANOMALY_VERIFICATION
            images_dev = self.images.to(self.device)

            if self.ram_mode:
                # Genuine re-verification: re-run clean RAM++ inference and compare with the shared baseline.
                try:
                    ev = self._ram_evaluate(self.images, with_maps=False)
                    ref = self._ref_scores()
                    sc = ev["scores_tensor"].detach().float().cpu().reshape(-1)
                    max_diff = float((sc - ref).abs().max().item()) if sc.numel() == ref.numel() else None
                    same = set(ev["tags"]) == set(self.baseline_tags)
                    trust_checks = {
                        "prediction_outputs": {
                            "passed": bool(same and max_diff is not None and max_diff <= 1e-3),
                            "message": (f"Re-verified RAM++ baseline: {len(ev['tags'])} active tags, identical tag set={same}, "
                                        f"max |score diff|={max_diff}."),
                        },
                        "pixel_range": check_pixel_range_tolerant(to_pixel_space(self.images.detach().cpu())),
                    }
                except Exception as exc:
                    trust_checks = {"prediction_outputs": {"passed": False, "message": f"{type(exc).__name__}: {exc}"}}
                trust_flag = generate_trust_flag(trust_checks)
                return StepEvaluationResult(
                    step_number=step_number,
                    action_type=action.value,
                    phase=EvaluationPhase.ANOMALY_VERIFICATION,
                    metrics={"re_verified": True},
                    trust_checks=trust_checks,
                    trust_flag=trust_flag,
                    execution_time_seconds=time.time() - step_start,
                    status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
                    timestamp=timestamp,
                )
            else:
                labels_dev = self.labels.to(self.device)
                with torch.no_grad():
                    outputs = self.model(images_dev)
                    preds = outputs.argmax(dim=1)

                total = labels_dev.size(0)
                correct = (preds == labels_dev).sum().item()
                clean_acc = 100.0 * correct / total if total > 0 else 0.0

                trust_checks = {
                    "prediction_outputs": check_prediction_outputs(outputs.cpu(), self.labels),
                    "pixel_range": check_pixel_range(self.images),
                }
                trust_flag = generate_trust_flag(trust_checks)

                return StepEvaluationResult(
                    step_number=step_number,
                    action_type=action.value,
                    phase=EvaluationPhase.ANOMALY_VERIFICATION,
                    metrics={"verified_clean_accuracy": clean_acc, "samples": total},
                    trust_checks=trust_checks,
                    trust_flag=trust_flag,
                    execution_time_seconds=time.time() - step_start,
                    status="SUCCESS",
                    timestamp=timestamp,
                )

        raise ValueError(f"Unsupported action execution: {action}")

    # ─────────────────────────────────────────────────────────────────────────
    # RAM++ action implementations
    #   All integrity checks, budgets and previews are in RAW PIXEL space; the
    #   normalised model input is only used to call RAM++ (see ram_space.py).
    #   Every number reported comes from a forward pass executed in this run.
    # ─────────────────────────────────────────────────────────────────────────
    def _ref_scores(self) -> torch.Tensor:
        """Shared-investigation baseline scores as a 1-D CPU tensor."""
        if self.baseline_scores is not None:
            return self.baseline_scores.detach().float().cpu().reshape(-1)
        if self.ram_baseline_outcome is not None:
            return self.ram_baseline_outcome["scores_tensor"].detach().float().cpu().reshape(-1)
        raise RuntimeError("No RAM++ baseline scores available.")

    def _active_idx(self) -> List[int]:
        return [self._tag_index[t] for t in self.baseline_tags if t in self._tag_index]

    def _probe_differentiable_path(self) -> Dict[str, Any]:
        """Preflight probe: find a differentiable scoring path that reproduces clean scores."""
        fn, report = resolve_scoring_fn(self.model, self.device, self.images, self._ref_scores())
        self._scoring_fn = fn
        self.scoring_report = report
        if report["passed"]:
            return {
                "passed": True,
                "message": (f"Attack gradient path verified: {report['name']} reproduces the clean "
                            f"RAM++ scores (max |diff| = {report['max_abs_diff']:.2e})."),
                "details": report,
            }
        return {
            "passed": False,
            "message": ("No differentiable scoring path reproduces the clean RAM++ scores, so gradient "
                        "attacks would target a different function than the one evaluated. "
                        "Details: " + report.get("message", "n/a")),
            "details": report,
        }

    def _ram_evaluate(self, norm_tensor: torch.Tensor, with_maps: bool = True) -> Dict[str, Any]:
        """Run RAM++ inference on a normalised [1,3,384,384] tensor; optionally capture activations."""
        from model.ram_adapter import get_baseline
        x = norm_tensor.detach()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        probe = _ActivationProbe(self.model) if with_maps else None
        if probe is not None:
            probe.__enter__()
        try:
            ev = get_baseline(x, model=self.model, device=self.device)
        finally:
            if probe is not None:
                probe.__exit__(None, None, None)
        ev["maps"] = probe.maps() if probe is not None else {}
        return ev

    def _record_stage(self, key: str, ev: Dict[str, Any]) -> None:
        self._stage_scores[key] = ev["scores_tensor"].detach().float().cpu().reshape(-1)
        self._stage_tags[key] = list(ev["tags"])
        self._stage_maps[key] = ev.get("maps", {}) or {}

    def _top_scored(self, tags: List[str], scores: torch.Tensor, n: int = 5) -> List[Dict[str, Any]]:
        rows = [(t, float(scores[self._tag_index[t]].item())) for t in tags if t in self._tag_index]
        rows.sort(key=lambda r: -r[1])
        return [{"class": t, "prob": round(v * 100.0, 2)} for t, v in rows[:n]]

    def _ram_fail(self, step_number, step_start, timestamp, action, phase, name, exc) -> StepEvaluationResult:
        msg = f"{type(exc).__name__}: {exc}"
        return StepEvaluationResult(
            step_number=step_number, action_type=action.value, phase=phase,
            metrics={"error": msg},
            trust_checks={name: {"passed": False, "message": msg}},
            trust_flag={"status": "FAIL", "summary": msg},
            execution_time_seconds=time.time() - step_start,
            status="FAIL", error_message=msg, timestamp=timestamp,
        )

    def _store_maps(self, s: Dict[str, Any], prefix: str, stage: str) -> None:
        """Convert recorded activations of `stage` into PNGs on sample `s` (clean-referenced scaling)."""
        maps = self._stage_maps.get(stage, {}) or {}
        clean = self._stage_maps.get("clean", {}) or {}
        meta = s.setdefault("feature_meta", {})
        for kind, key, cmap in (("patch", "patch_embed", "viridis"), ("stage4", "stage4", "viridis")):
            m = maps.get(key)
            if m is None:
                continue
            ref = clean.get(key)
            s[f"{prefix}_{kind}_feature_b64"] = feature_map_to_base64_png(_norm_map(m, ref), cmap, size=(160, 160))
            if prefix != "clean" and ref is not None:
                d = (m - ref).abs()
                s[f"{prefix}_{kind}_distortion_b64"] = feature_map_to_base64_png(
                    _norm_map(d), "magma", size=(160, 160)) if float(d.max()) > 0 else ""
                meta[f"{prefix}_{kind}_mean_rel_shift"] = round(float(d.mean() / (ref.abs().mean() + 1e-12)), 5)
        meta["patch_embed_shape"] = maps.get("patch_embed_shape") or meta.get("patch_embed_shape")
        meta["stage4_grid"] = maps.get("stage4_grid") or meta.get("stage4_grid")
        # legacy keys read by older views
        if f"{prefix}_patch_feature_b64" in s:
            s[f"{prefix}_feature_b64"] = s[f"{prefix}_patch_feature_b64"]
        if prefix == "pgd" and "pgd_patch_distortion_b64" in s:
            s["feature_distortion_b64"] = s["pgd_patch_distortion_b64"]

    def _rebuild_tag_views(self) -> None:
        """Per-tag score tables (measured) used by the weakness and score-distribution views."""
        clean = self._ref_scores()
        base = list(self.baseline_tags)
        order = sorted((t for t in base if t in self._tag_index), key=lambda t: -float(clean[self._tag_index[t]]))
        stage_keys = [("fgsm", "fgsm"), ("pgd", "pgd"), ("def_fgsm", "def_fgsm"), ("def_pgd", "def_pgd")]
        rows: Dict[str, Dict[str, Any]] = {}
        for t in order[:20]:
            i = self._tag_index[t]
            row: Dict[str, Any] = {"clean_acc": round(float(clean[i]) * 100.0, 2), "clean_active": True}
            for stage, prefix in stage_keys:
                if stage in self._stage_scores:
                    row[f"{prefix}_acc"] = round(float(self._stage_scores[stage][i]) * 100.0, 2)
                    row[f"{prefix}_active"] = t in set(self._stage_tags[stage])
            rows[t] = row
        self.class_metrics = rows

        comp: List[Dict[str, Any]] = []
        for t in order[:8]:
            i = self._tag_index[t]
            entry = {"tag": t, "is_new": False, "clean": round(float(clean[i]) * 100.0, 2)}
            for stage, _ in stage_keys:
                if stage in self._stage_scores:
                    entry[stage] = round(float(self._stage_scores[stage][i]) * 100.0, 2)
            comp.append(entry)
        introduced: Dict[str, float] = {}
        for stage in ("fgsm", "pgd"):
            for t in self._stage_tags.get(stage, []):
                if t not in set(base) and t in self._tag_index:
                    introduced[t] = max(introduced.get(t, 0.0), float(self._stage_scores[stage][self._tag_index[t]]))
        for t, _ in sorted(introduced.items(), key=lambda kv: -kv[1])[:3]:
            i = self._tag_index[t]
            entry = {"tag": t, "is_new": True, "clean": round(float(clean[i]) * 100.0, 2)}
            for stage, _k in stage_keys:
                if stage in self._stage_scores:
                    entry[stage] = round(float(self._stage_scores[stage][i]) * 100.0, 2)
            comp.append(entry)
        self.sample_inspection["tag_comparison"] = comp
        self.flip_matrix = {}  # class-flip matrices do not exist for multi-label tagging

    def _run_ram_baseline(self, step_number, step_start, timestamp, action) -> StepEvaluationResult:
        """Clean RAM++ inference on the shared image; verifies it reproduces the shared baseline."""
        self.model.eval()
        try:
            ev = self._ram_evaluate(self.images, with_maps=True)
        except Exception as exc:
            return self._ram_fail(step_number, step_start, timestamp, action,
                                  EvaluationPhase.BASELINE, "ram_baseline", exc)

        ref = self._ref_scores()
        scores = ev["scores_tensor"].detach().float().cpu().reshape(-1)
        same_len = scores.numel() == ref.numel()
        max_diff = float((scores - ref).abs().max().item()) if same_len else None
        same_tags = set(ev["tags"]) == set(self.baseline_tags)
        consistent = bool(same_len and same_tags and max_diff <= 1e-3)

        self.ram_baseline_outcome = ev
        self._record_stage("clean", ev)
        active = list(self.baseline_tags)
        tag_count = len(active)
        top_tags = [r["class"] for r in self._top_scored(active, ref, 10)]
        top5 = self._top_scored(active, ref, 5)

        self.sample_inspection["baseline_tags"] = active
        self.sample_inspection["top_tags"] = top_tags
        self.sample_inspection["tag_count"] = tag_count

        clean_px = to_pixel_space(self.images[0].detach().cpu())
        s: Dict[str, Any] = {
            "sample_id": 1, "true_label": 0, "true_class": "investigation_image",
            "clean_pred": 0, "clean_class": _format_tags([r["class"] for r in top5], max_n=5),
            "clean_tag_count": tag_count,
            "clean_confidence": top5[0]["prob"] if top5 else 0.0,
            "clean_top5": top5,
            "clean_img_b64": tensor_to_base64_png(clean_px, size=(160, 160), smooth=True),
            "feature_meta": {},
            "fgsm_pred": None, "fgsm_class": None, "fgsm_confidence": 0.0, "fgsm_top5": [],
            "fgsm_l_inf": 0.0, "fgsm_img_b64": "", "fgsm_feature_b64": "", "fgsm_heatmap_b64": "", "fgsm_diff_matrix": [],
            "pgd_pred": None, "pgd_class": None, "pgd_confidence": 0.0, "pgd_top5": [],
            "pgd_l_inf": 0.0, "pgd_img_b64": "", "pgd_feature_b64": "", "feature_distortion_b64": "",
            "pgd_heatmap_b64": "", "pgd_diff_matrix": [],
            "fooled": False,
            "trajectory": [{
                "stage": "Clean Baseline", "step": 0, "pred": _format_tags(active, 5),
                "confidence": 100.0, "true_class_prob": 100.0, "tags_kept": tag_count, "tags_total": tag_count,
                "mean_baseline_score": round(float(sum(ref[self._tag_index[t]] for t in active if t in self._tag_index))
                                             / max(1, tag_count) * 100.0, 2),
            }],
        }
        self.detailed_samples = [s]
        self._store_maps(s, "clean", "clean")
        self._rebuild_tag_views()

        trust_checks = {
            "ram_baseline_consistency": {
                "passed": consistent,
                "message": (f"Mission re-run reproduces the shared baseline: identical tag set={same_tags}, "
                            f"max |score diff|={max_diff if max_diff is None else format(max_diff, '.2e')}."),
            },
            "pixel_range": check_pixel_range_tolerant(clean_px),
            "scores_finite": check_scores_finite_ram(scores),
        }
        trust_flag = generate_trust_flag(trust_checks)
        return StepEvaluationResult(
            step_number=step_number, action_type=action.value, phase=EvaluationPhase.BASELINE,
            metrics={
                "tag_count": tag_count,
                "top_tags": top_tags,
                "total_samples": 1,
                "inference_ms": ev.get("inference_ms"),
                "baseline_max_score_diff": max_diff,
                # Policy-compatibility alias: baseline tags measured against themselves.
                # This is definitional (100%), NOT a measured classification accuracy.
                "clean_accuracy": 100.0,
            },
            trust_checks=trust_checks, trust_flag=trust_flag,
            execution_time_seconds=time.time() - step_start,
            status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
            error_message=None if trust_flag["status"] != "FAIL" else trust_flag.get("summary"),
            timestamp=timestamp,
        )

    def _run_ram_fgsm(self, step_number, step_start, timestamp, action) -> StepEvaluationResult:
        return self._run_ram_attack("fgsm", step_number, step_start, timestamp, action)

    def _run_ram_pgd(self, step_number, step_start, timestamp, action) -> StepEvaluationResult:
        return self._run_ram_attack("pgd", step_number, step_start, timestamp, action)

    def _run_ram_attack(self, kind, step_number, step_start, timestamp, action) -> StepEvaluationResult:
        """Tag-suppression attack (pixel-space Linf budget); every metric is re-measured on the final image."""
        label = "FGSM" if kind == "fgsm" else "PGD"
        phase = EvaluationPhase.ATTACK_EVALUATION
        if self._scoring_fn is None:
            return self._ram_fail(step_number, step_start, timestamp, action, phase, f"ram_{kind}",
                                  RuntimeError("No verified differentiable scoring path (preflight did not resolve one)."))
        eps = self.epsilon
        steps = 1 if kind == "fgsm" else self.pgd_steps
        alpha = eps if kind == "fgsm" else self.pgd_alpha
        base = self._ref_scores()
        pixel_clean = to_pixel_space(self.images.detach().cpu())
        try:
            atk = tag_suppression_attack(self._scoring_fn, pixel_clean, self._active_idx(), eps, alpha, steps, self.device)
            adv_px = atk["adversarial_pixels"]
            adv_norm = to_model_space(adv_px)
            ev = self._ram_evaluate(adv_norm, with_maps=True)
        except Exception as exc:
            return self._ram_fail(step_number, step_start, timestamp, action, phase, f"ram_{kind}", exc)

        adv_tags = list(ev["tags"])
        outcome = compute_attack_outcome(
            baseline_tags=self.baseline_tags, attacked_tags=adv_tags,
            baseline_scores=base, attacked_scores=ev["scores_tensor"],
            tag_list=self.tag_list, epsilon=eps, attack_name=label,
        )
        n_base = len(self.baseline_tags)
        kept = len(set(self.baseline_tags) & set(adv_tags))
        pres = (kept / n_base) if n_base else 1.0                  # exact fraction in [0, 1]
        disap, intro = outcome["tag_disappearance_count"], outcome["tag_introduction_count"]
        asr = round((1.0 - pres) * 100.0, 2)                       # % of baseline tags lost
        attack_succeeded = disap > 0

        setattr(self, f"{kind}_adv_images", adv_norm.detach().cpu())
        self._record_stage(kind, ev)

        budget_check = check_budget_pixel(self.images.detach().cpu(), adv_norm, eps)
        pixel_check = check_pixel_range_tolerant(adv_px)
        roundtrip_check = check_pixel_range_tolerant(to_pixel_space(adv_norm))
        trust_checks = {
            "perturbation_budget": budget_check,
            "pixel_range": pixel_check,
            "normalisation_roundtrip": roundtrip_check,
            "scores_finite": outcome["trust_checks"].get("scores_finite", {"passed": True, "message": "n/a"}),
            "gradient_signal": {
                "passed": bool(atk["gradient_nonzero"]),
                "message": ("Input gradient was non-zero (max |grad| over steps = "
                            f"{max(atk['grad_linf_trace']):.3e})." if atk["gradient_nonzero"]
                            else "Input gradient was exactly zero at every step: the attack could not move the image "
                                 "(disconnected graph or masked gradient); no robustness conclusion is valid."),
            },
            "metric_consistency": {
                # independent cross-check against ram_metrics (which rounds to 2 decimals of a percent)
                "passed": abs(pres - outcome["tag_preservation_rate"] / 100.0) < 1e-4,
                "message": f"Tag preservation {pres:.2%} = {kept}/{n_base} baseline tags retained "
                           f"(ram_metrics reports {outcome['tag_preservation_rate']:.2f}%).",
            },
        }
        trust_flag = generate_trust_flag(trust_checks)

        s = self.detailed_samples[0] if self.detailed_samples else None
        if s is not None:
            clean_px = pixel_clean[0]
            adv_px0 = adv_px[0]
            top = self._top_scored(adv_tags, ev["scores_tensor"], 5)
            s[f"{kind}_pred"] = 0
            s[f"{kind}_class"] = _format_tags([r["class"] for r in top], 5) if top else "(no tags)"
            s[f"{kind}_tag_count"] = len(adv_tags)
            s[f"{kind}_tags_kept"], s[f"{kind}_tags_lost"], s[f"{kind}_tags_new"] = kept, disap, intro
            s[f"{kind}_confidence"] = round(pres * 100.0, 2)       # tag preservation %
            s[f"{kind}_top5"] = top
            s[f"{kind}_l_inf"] = round(float(budget_check.get("max_perturbation", 0.0)), 5)
            s[f"{kind}_img_b64"] = tensor_to_base64_png(adv_px0, size=(160, 160), smooth=True)
            s[f"{kind}_heatmap_b64"] = diff_heatmap_to_base64_png(clean_px, adv_px0, eps, size=(160, 160))
            s[f"{kind}_diff_matrix"] = get_diff_matrix_2d(clean_px, adv_px0, pool=8)
            s["fooled"] = bool(s.get("fooled")) or attack_succeeded
            self._store_maps(s, kind, kind)
            s["trajectory"].append({
                "stage": "FGSM 1-Step" if kind == "fgsm" else f"PGD-{self.pgd_steps} Final",
                "step": steps, "pred": _format_tags([r["class"] for r in top], 5) if top else "(no tags)",
                "confidence": round(pres * 100.0, 2), "true_class_prob": round(pres * 100.0, 2),
                "tags_kept": kept, "tags_total": n_base,
                "mean_baseline_score": round(float(sum(ev["scores_tensor"][self._tag_index[t]] for t in self.baseline_tags
                                                       if t in self._tag_index)) / max(1, n_base) * 100.0, 2),
            })

        self.sample_inspection[f"{kind}_tags"] = adv_tags
        setattr(self, f"{kind}_outcome", {
            "adv_tags": adv_tags, "pres_rate": pres, "disap_count": disap,
            "intro_count": intro, "attack_succeeded": attack_succeeded,
        })
        self._rebuild_tag_views()

        metrics = {
            "tag_preservation_rate": pres,
            "tags_kept": kept, "baseline_tag_count": n_base, "attacked_tag_count": len(adv_tags),
            "tags_disappeared": disap, "tags_introduced": intro,
            "tag_introduction_rate": outcome["tag_introduction_rate"],
            "jaccard_similarity": outcome["jaccard_similarity"],
            "mean_score_delta_active": outcome["mean_score_delta_active"],
            "worst_suppressed_tag": outcome["worst_suppressed_tag"],
            "worst_score_drop": outcome["worst_score_drop"],
            # policy-compatibility aliases (tag semantics, see README of this module)
            "attack_success_rate": asr,                       # % of baseline tags lost
            "adversarial_accuracy": round(pres * 100.0, 2),   # tag preservation %
            "epsilon": eps,
            "max_perturbation": budget_check.get("max_perturbation", 0.0),
            "attack_loss_start": atk["loss_trace"][0], "attack_loss_end": atk["loss_trace"][-1],
            "scoring_path": (self.scoring_report or {}).get("name"),
            "total_samples": 1,
        }
        if kind == "pgd":
            metrics.update({"pgd_steps": self.pgd_steps, "pgd_alpha": self.pgd_alpha})
        return StepEvaluationResult(
            step_number=step_number, action_type=action.value, phase=phase, metrics=metrics,
            trust_checks=trust_checks, trust_flag=trust_flag,
            execution_time_seconds=time.time() - step_start,
            status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
            error_message=None if trust_flag["status"] != "FAIL" else trust_flag.get("summary"),
            timestamp=timestamp,
        )

    def _run_ram_defense(self, step_number, step_start, timestamp, action) -> StepEvaluationResult:
        """Input quantisation applied in RAW PIXEL space, then re-evaluated by RAM++."""
        levels = self.quantization_levels
        base_tags = list(self.baseline_tags)
        n_base = len(base_tags)
        trust_checks: Dict[str, Dict[str, Any]] = {}
        pres: Dict[str, Optional[float]] = {"clean": None, "fgsm": None, "pgd": None}

        inputs = {"clean": to_pixel_space(self.images.detach().cpu())}
        if self.fgsm_adv_images is not None:
            inputs["fgsm"] = to_pixel_space(self.fgsm_adv_images)
        if self.pgd_adv_images is not None:
            inputs["pgd"] = to_pixel_space(self.pgd_adv_images)

        for name, px in inputs.items():
            key = "def_clean" if name == "clean" else f"def_{name}"
            try:
                q = quantization_defense(px.clone(), levels=levels).clamp(0.0, 1.0)
                ev = self._ram_evaluate(to_model_space(q), with_maps=False)
                self._record_stage(key, ev)
                kept = len(set(base_tags) & set(ev["tags"]))
                pres[name] = kept / n_base if n_base else 1.0
                trust_checks[f"defense_{name}"] = {
                    "passed": bool(torch.isfinite(ev["scores_tensor"]).all().item()),
                    "message": f"Defended {name}: {kept}/{n_base} baseline tags retained ({pres[name]:.2%}).",
                }
            except Exception as exc:
                trust_checks[f"defense_{name}"] = {"passed": False, "message": f"{type(exc).__name__}: {exc}"}

        if all(v is None for v in pres.values()):
            msg = "; ".join(c["message"] for c in trust_checks.values())
            return StepEvaluationResult(
                step_number=step_number, action_type=action.value, phase=EvaluationPhase.DEFENSE_VERIFICATION,
                metrics={"error": msg}, trust_checks=trust_checks,
                trust_flag={"status": "FAIL", "summary": msg},
                execution_time_seconds=time.time() - step_start, status="FAIL",
                error_message=msg, timestamp=timestamp,
            )

        raw_fgsm = self.metrics_by_action.get("RUN_FGSM", {}).get("tag_preservation_rate")
        raw_pgd = self.metrics_by_action.get("ESCALATE_PGD", {}).get("tag_preservation_rate")
        metrics: Dict[str, Any] = {"defense_name": f"Input Quantization ({levels} levels, raw-pixel space)"}
        if pres["clean"] is not None:
            metrics["defense_clean_tag_pres"] = pres["clean"]
            metrics["defense_clean_accuracy"] = round(pres["clean"] * 100.0, 2)
        for name, raw in (("fgsm", raw_fgsm), ("pgd", raw_pgd)):
            if pres[name] is None:
                continue
            metrics[f"defended_{name}_tag_pres"] = pres[name]
            metrics[f"defended_{name}_accuracy"] = round(pres[name] * 100.0, 2)
            metrics[f"defended_{name}_asr"] = round((1.0 - pres[name]) * 100.0, 2)
            if raw is not None:
                metrics[f"{name}_recovery_gain"] = pres[name] - raw

        if self.detailed_samples:
            s = self.detailed_samples[0]
            shown = "def_pgd" if pres["pgd"] is not None else ("def_fgsm" if pres["fgsm"] is not None else "def_clean")
            tags = self._stage_tags.get(shown, [])
            top = self._top_scored(tags, self._stage_scores[shown], 5)
            main = pres["pgd"] if pres["pgd"] is not None else (pres["fgsm"] if pres["fgsm"] is not None else pres["clean"])
            gain = metrics.get("pgd_recovery_gain", metrics.get("fgsm_recovery_gain"))
            s["defended_pred"] = 0
            s["defended_class"] = _format_tags([r["class"] for r in top], 5) if top else "(no tags)"
            s["recovered"] = bool(gain is not None and gain > 0.0)
            s["defended_confidence"] = round(main * 100.0, 2)
            s["defended_true_prob"] = round((pres["clean"] if pres["clean"] is not None else main) * 100.0, 2)
            s["trajectory"].append({
                "stage": f"Defended (Quant-{levels})", "step": self.pgd_steps + 1,
                "pred": s["defended_class"], "confidence": round(main * 100.0, 2),
                "true_class_prob": round(main * 100.0, 2),
                "tags_kept": int(round(main * n_base)), "tags_total": n_base,
            })
        self._rebuild_tag_views()

        trust_flag = generate_trust_flag(trust_checks)
        return StepEvaluationResult(
            step_number=step_number, action_type=action.value, phase=EvaluationPhase.DEFENSE_VERIFICATION,
            metrics=metrics, trust_checks=trust_checks, trust_flag=trust_flag,
            execution_time_seconds=time.time() - step_start,
            status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
            error_message=None if trust_flag["status"] != "FAIL" else trust_flag.get("summary"),
            timestamp=timestamp,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CIFAR-10 action implementations (legacy)
    # ─────────────────────────────────────────────────────────────────────────
    def _run_cifar_baseline(self, step_number, step_start, timestamp, action) -> StepEvaluationResult:
        images_dev = self.images.to(self.device)
        labels_dev = self.labels.to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(images_dev)
            preds = outputs.argmax(dim=1)
            self.clean_predictions = preds.detach().cpu()

        total = labels_dev.size(0)
        correct = (preds == labels_dev).sum().item()
        clean_acc = 100.0 * correct / total if total > 0 else 0.0

        sample_count = min(8, total)
        self.sample_inspection["true_labels"] = [int(x) for x in labels_dev[:sample_count].tolist()]
        self.sample_inspection["clean_preds"] = [int(x) for x in preds[:sample_count].tolist()]

        self.detailed_samples = []
        num_classes = outputs.size(1)
        softmax_outputs = torch.softmax(outputs, dim=1).detach().cpu()

        for i in range(sample_count):
            lbl = int(labels_dev[i].item())
            pred_lbl = int(preds[i].item())
            lbl_cls = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else str(lbl)
            pred_cls = CLASS_NAMES[pred_lbl] if pred_lbl < len(CLASS_NAMES) else str(pred_lbl)
            conf = float(softmax_outputs[i][pred_lbl].item()) if pred_lbl < num_classes else 1.0

            top_k = min(5, num_classes)
            top_probs, top_indices = torch.topk(softmax_outputs[i], k=top_k)
            top5_list = []
            for p_val, idx_val in zip(top_probs, top_indices):
                c_idx = int(idx_val.item())
                c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else str(c_idx)
                top5_list.append({"class": c_name, "prob": round(float(p_val.item()) * 100, 2)})

            clean_b64 = tensor_to_base64_png(self.images[i])
            true_prob = float(softmax_outputs[i][lbl].item()) if lbl < num_classes else conf
            clean_fmap = extract_conv1_features(self.model, images_dev[i:i+1])
            clean_fmap_b64 = feature_map_to_base64_png(clean_fmap, "viridis")

            self.detailed_samples.append({
                "sample_id": i + 1,
                "true_label": lbl,
                "true_class": lbl_cls,
                "clean_pred": pred_lbl,
                "clean_class": pred_cls,
                "clean_confidence": round(conf * 100.0, 2),
                "clean_top5": top5_list,
                "clean_img_b64": clean_b64,
                "clean_feature_b64": clean_fmap_b64,
                "fgsm_pred": None, "fgsm_class": None, "fgsm_confidence": 0.0, "fgsm_top5": [],
                "fgsm_l_inf": 0.0, "fgsm_img_b64": "", "fgsm_feature_b64": "", "fgsm_heatmap_b64": "", "fgsm_diff_matrix": [],
                "pgd_pred": None, "pgd_class": None, "pgd_confidence": 0.0, "pgd_top5": [],
                "pgd_l_inf": 0.0, "pgd_img_b64": "", "pgd_feature_b64": "", "feature_distortion_b64": "", "pgd_heatmap_b64": "", "pgd_diff_matrix": [],
                "fooled": False,
                "trajectory": [{"stage": "Clean Baseline", "step": 0, "pred": pred_cls, "confidence": round(conf * 100.0, 2), "true_class_prob": round(true_prob * 100.0, 2)}],
            })

        self.class_metrics = {}
        for c_idx in range(min(10, num_classes)):
            c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else f"Class {c_idx}"
            mask = (labels_dev == c_idx)
            c_tot = mask.sum().item()
            if c_tot > 0:
                c_cor = (preds[mask] == labels_dev[mask]).sum().item()
                self.class_metrics[c_name] = {
                    "total": c_tot, "clean_correct": c_cor,
                    "clean_acc": round(100.0 * c_cor / c_tot, 2),
                    "fgsm_correct": 0, "fgsm_acc": 0.0,
                    "pgd_correct": 0, "pgd_acc": 0.0,
                }

        pred_check = check_prediction_outputs(outputs.cpu(), self.labels)
        pixel_check = check_pixel_range(self.images)
        consistency_check = check_metric_consistency(clean_acc, clean_acc, 0.0)

        trust_checks = {
            "prediction_outputs": pred_check,
            "pixel_range": pixel_check,
            "metric_consistency": consistency_check,
        }
        trust_flag = generate_trust_flag(trust_checks)

        return StepEvaluationResult(
            step_number=step_number,
            action_type=action.value,
            phase=EvaluationPhase.BASELINE,
            metrics={"clean_accuracy": clean_acc, "total_samples": total, "correct_samples": correct},
            trust_checks=trust_checks,
            trust_flag=trust_flag,
            execution_time_seconds=time.time() - step_start,
            status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
            timestamp=timestamp,
        )

    def _run_cifar_fgsm(self, step_number, step_start, timestamp, action) -> StepEvaluationResult:
        images_dev = self.images.to(self.device)
        labels_dev = self.labels.to(self.device)

        attack_labels = (
            self.clean_predictions.to(self.device)
            if self.clean_predictions is not None
            else labels_dev
        )

        adv_images = fgsm_attack(self.model, images_dev, attack_labels, epsilon=self.epsilon)
        self.fgsm_adv_images = adv_images.detach().cpu()

        with torch.no_grad():
            adv_outputs = self.model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)

        total = labels_dev.size(0)
        adv_correct = (adv_preds == labels_dev).sum().item()
        adv_acc = 100.0 * adv_correct / total if total > 0 else 0.0

        sample_count = min(8, total)
        self.sample_inspection["fgsm_preds"] = [int(x) for x in adv_preds[:sample_count].tolist()]

        num_classes = adv_outputs.size(1)
        fgsm_softmax = torch.softmax(adv_outputs, dim=1).detach().cpu() if num_classes > 1 else adv_outputs.detach().cpu()

        for i in range(min(sample_count, len(self.detailed_samples))):
            s = self.detailed_samples[i]
            pred_lbl = int(adv_preds[i].item())
            pred_cls = CLASS_NAMES[pred_lbl] if pred_lbl < len(CLASS_NAMES) else str(pred_lbl)
            conf = float(fgsm_softmax[i][pred_lbl].item()) if pred_lbl < num_classes else 1.0

            top_k = min(5, num_classes)
            top_probs, top_indices = torch.topk(fgsm_softmax[i], k=top_k)
            top5_list = [{"class": CLASS_NAMES[int(idx.item())] if int(idx.item()) < len(CLASS_NAMES) else str(int(idx.item())), "prob": round(float(p.item()) * 100, 2)} for p, idx in zip(top_probs, top_indices)]

            fgsm_img = adv_images[i].detach().cpu()
            clean_img = self.images[i].detach().cpu()
            l_inf = float((fgsm_img - clean_img).abs().max().item())

            s["fgsm_pred"] = pred_lbl
            s["fgsm_class"] = pred_cls
            s["fgsm_confidence"] = round(conf * 100.0, 2)
            s["fgsm_top5"] = top5_list
            s["fgsm_l_inf"] = round(l_inf, 5)
            s["fgsm_img_b64"] = tensor_to_base64_png(fgsm_img)
            s["fgsm_heatmap_b64"] = diff_heatmap_to_base64_png(clean_img, fgsm_img, self.epsilon)
            s["fgsm_diff_matrix"] = get_diff_matrix_2d(clean_img, fgsm_img)
            fgsm_fmap = extract_conv1_features(self.model, adv_images[i:i+1])
            s["fgsm_feature_b64"] = feature_map_to_base64_png(fgsm_fmap, "viridis")
            true_lbl = s["true_label"]
            true_prob = float(fgsm_softmax[i][true_lbl].item()) if true_lbl < num_classes else 0.0
            s["trajectory"].append({"stage": "FGSM 1-Step", "step": 1, "pred": pred_cls, "confidence": round(conf * 100.0, 2), "true_class_prob": round(true_prob * 100.0, 2)})

        for c_idx in range(min(10, num_classes)):
            c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else f"Class {c_idx}"
            if c_name in self.class_metrics:
                mask = (labels_dev == c_idx)
                c_tot = mask.sum().item()
                if c_tot > 0:
                    c_cor = (adv_preds[mask] == labels_dev[mask]).sum().item()
                    self.class_metrics[c_name]["fgsm_correct"] = c_cor
                    self.class_metrics[c_name]["fgsm_acc"] = round(100.0 * c_cor / c_tot, 2)

        clean_preds_dev = self.clean_predictions.to(self.device) if self.clean_predictions is not None else labels_dev
        asr = calculate_attack_success(clean_preds_dev, adv_preds, labels_dev)
        budget_check = check_perturbation_budget(images_dev, adv_images, epsilon=self.epsilon)
        pixel_check = check_pixel_range(adv_images)
        output_check = check_prediction_outputs(adv_outputs.cpu(), self.labels)
        clean_acc = self.metrics_by_action.get("RUN_BASELINE", {}).get("clean_accuracy", 100.0)
        consistency_check = check_metric_consistency(clean_acc, adv_acc, asr)

        trust_checks = {"perturbation_budget": budget_check, "pixel_range": pixel_check, "prediction_outputs": output_check, "metric_consistency": consistency_check}
        trust_flag = generate_trust_flag(trust_checks)

        return StepEvaluationResult(
            step_number=step_number,
            action_type=action.value,
            phase=EvaluationPhase.ATTACK_EVALUATION,
            metrics={"adversarial_accuracy": adv_acc, "attack_success_rate": asr, "epsilon": self.epsilon, "max_perturbation": budget_check.get("max_perturbation", 0.0), "total_samples": total},
            trust_checks=trust_checks,
            trust_flag=trust_flag,
            execution_time_seconds=time.time() - step_start,
            status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
            timestamp=timestamp,
        )

    def _run_cifar_pgd(self, step_number, step_start, timestamp, action) -> StepEvaluationResult:
        images_dev = self.images.to(self.device)
        labels_dev = self.labels.to(self.device)

        attack_labels = self.clean_predictions.to(self.device) if self.clean_predictions is not None else labels_dev
        adv_images = pgd_attack(self.model, images_dev, attack_labels, epsilon=self.epsilon, alpha=self.pgd_alpha, steps=self.pgd_steps, random_start=True)
        self.pgd_adv_images = adv_images.detach().cpu()

        with torch.no_grad():
            adv_outputs = self.model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)

        total = labels_dev.size(0)
        adv_correct = (adv_preds == labels_dev).sum().item()
        adv_acc = 100.0 * adv_correct / total if total > 0 else 0.0

        sample_count = min(8, total)
        self.sample_inspection["pgd_preds"] = [int(x) for x in adv_preds[:sample_count].tolist()]

        num_classes = adv_outputs.size(1)
        pgd_softmax = torch.softmax(adv_outputs, dim=1).detach().cpu() if num_classes > 1 else adv_outputs.detach().cpu()

        for i in range(min(sample_count, len(self.detailed_samples))):
            s = self.detailed_samples[i]
            pred_lbl = int(adv_preds[i].item())
            pred_cls = CLASS_NAMES[pred_lbl] if pred_lbl < len(CLASS_NAMES) else str(pred_lbl)
            conf = float(pgd_softmax[i][pred_lbl].item()) if pred_lbl < num_classes else 1.0
            top_k = min(5, num_classes)
            top_probs, top_indices = torch.topk(pgd_softmax[i], k=top_k)
            top5_list = [{"class": CLASS_NAMES[int(idx.item())] if int(idx.item()) < len(CLASS_NAMES) else str(int(idx.item())), "prob": round(float(p.item()) * 100, 2)} for p, idx in zip(top_probs, top_indices)]
            pgd_img = adv_images[i].detach().cpu()
            clean_img = self.images[i].detach().cpu()
            l_inf = float((pgd_img - clean_img).abs().max().item())
            s["pgd_pred"] = pred_lbl
            s["pgd_class"] = pred_cls
            s["pgd_confidence"] = round(conf * 100.0, 2)
            s["pgd_top5"] = top5_list
            s["pgd_l_inf"] = round(l_inf, 5)
            s["pgd_img_b64"] = tensor_to_base64_png(pgd_img)
            s["pgd_heatmap_b64"] = diff_heatmap_to_base64_png(clean_img, pgd_img, self.epsilon)
            s["pgd_diff_matrix"] = get_diff_matrix_2d(clean_img, pgd_img)
            pgd_fmap = extract_conv1_features(self.model, adv_images[i:i+1])
            s["pgd_feature_b64"] = feature_map_to_base64_png(pgd_fmap, "viridis")
            clean_fmap = extract_conv1_features(self.model, images_dev[i:i+1])
            if clean_fmap is not None and pgd_fmap is not None:
                feat_dist = (pgd_fmap - clean_fmap).abs()
                s["feature_distortion_b64"] = feature_map_to_base64_png(feat_dist, "magma")
            else:
                s["feature_distortion_b64"] = ""
            s["fooled"] = (s.get("clean_class") == s["true_class"]) and (pred_cls != s["true_class"])
            true_lbl = s["true_label"]
            true_prob = float(pgd_softmax[i][true_lbl].item()) if true_lbl < num_classes else 0.0
            s["trajectory"].append({"stage": f"PGD-{self.pgd_steps} Final", "step": self.pgd_steps, "pred": pred_cls, "confidence": round(conf * 100.0, 2), "true_class_prob": round(true_prob * 100.0, 2)})

        for c_idx in range(min(10, num_classes)):
            c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else f"Class {c_idx}"
            if c_name in self.class_metrics:
                mask = (labels_dev == c_idx)
                c_tot = mask.sum().item()
                if c_tot > 0:
                    c_cor = (adv_preds[mask] == labels_dev[mask]).sum().item()
                    self.class_metrics[c_name]["pgd_correct"] = c_cor
                    self.class_metrics[c_name]["pgd_acc"] = round(100.0 * c_cor / c_tot, 2)

        clean_preds_cpu = self.clean_predictions if self.clean_predictions is not None else labels_dev.cpu()
        adv_preds_cpu = adv_preds.cpu()
        matrix_names = CLASS_NAMES[:num_classes]
        flip_counts = {c1: {c2: 0 for c2 in matrix_names} for c1 in matrix_names}
        for t_idx, p_idx in zip(clean_preds_cpu, adv_preds_cpu):
            t_i = int(t_idx.item())
            p_i = int(p_idx.item())
            if t_i < len(CLASS_NAMES) and p_i < len(CLASS_NAMES):
                t_name = CLASS_NAMES[t_i]
                p_name = CLASS_NAMES[p_i]
                if t_name in flip_counts and p_name in flip_counts[t_name]:
                    flip_counts[t_name][p_name] += 1
        self.flip_matrix = flip_counts

        clean_preds_dev = self.clean_predictions.to(self.device) if self.clean_predictions is not None else labels_dev
        asr = calculate_attack_success(clean_preds_dev, adv_preds, labels_dev)
        budget_check = check_perturbation_budget(images_dev, adv_images, epsilon=self.epsilon)
        pixel_check = check_pixel_range(adv_images)
        output_check = check_prediction_outputs(adv_outputs.cpu(), self.labels)
        clean_acc = self.metrics_by_action.get("RUN_BASELINE", {}).get("clean_accuracy", 100.0)
        consistency_check = check_metric_consistency(clean_acc, adv_acc, asr)

        trust_checks = {"perturbation_budget": budget_check, "pixel_range": pixel_check, "prediction_outputs": output_check, "metric_consistency": consistency_check}
        trust_flag = generate_trust_flag(trust_checks)

        return StepEvaluationResult(
            step_number=step_number,
            action_type=action.value,
            phase=EvaluationPhase.ATTACK_EVALUATION,
            metrics={"adversarial_accuracy": adv_acc, "attack_success_rate": asr, "epsilon": self.epsilon, "pgd_steps": self.pgd_steps, "pgd_alpha": self.pgd_alpha, "max_perturbation": budget_check.get("max_perturbation", 0.0), "total_samples": total},
            trust_checks=trust_checks,
            trust_flag=trust_flag,
            execution_time_seconds=time.time() - step_start,
            status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
            timestamp=timestamp,
        )

    def _run_cifar_defense(self, step_number, step_start, timestamp, action) -> StepEvaluationResult:
        images_dev = self.images.to(self.device)
        labels_dev = self.labels.to(self.device)
        defense_fn = lambda x: quantization_defense(x, levels=self.quantization_levels)

        def_clean_acc, def_clean_preds = calculate_accuracy(self.model, images_dev, labels_dev, defense_fn)
        def_fgsm_acc, def_fgsm_preds = 0.0, None
        def_fgsm_asr = 0.0
        if self.fgsm_adv_images is not None:
            fgsm_dev = self.fgsm_adv_images.to(self.device)
            def_fgsm_acc, def_fgsm_preds = calculate_accuracy(self.model, fgsm_dev, labels_dev, defense_fn)
            def_fgsm_asr = calculate_attack_success(def_clean_preds, def_fgsm_preds, labels_dev)

        def_pgd_acc, def_pgd_preds = 0.0, None
        def_pgd_asr = 0.0
        if self.pgd_adv_images is not None:
            pgd_dev = self.pgd_adv_images.to(self.device)
            def_pgd_acc, def_pgd_preds = calculate_accuracy(self.model, pgd_dev, labels_dev, defense_fn)
            def_pgd_asr = calculate_attack_success(def_clean_preds, def_pgd_preds, labels_dev)

        if def_pgd_preds is not None:
            sample_count = min(8, labels_dev.size(0))
            with torch.no_grad():
                def_inputs = defense_fn(pgd_dev[:sample_count])
                def_outputs = self.model(def_inputs)
                def_probs = torch.softmax(def_outputs, dim=1)

            for i in range(min(sample_count, len(self.detailed_samples))):
                s = self.detailed_samples[i]
                def_lbl = int(def_pgd_preds[i].item())
                def_cls = CLASS_NAMES[def_lbl] if def_lbl < len(CLASS_NAMES) else str(def_lbl)
                recovered = (s.get("clean_class") == s["true_class"]) and (def_cls == s["true_class"])
                s["defended_pred"] = def_lbl
                s["defended_class"] = def_cls
                s["recovered"] = recovered
                true_lbl = s["true_label"]
                true_prob = float(def_probs[i][true_lbl].item() * 100.0) if true_lbl < 10 else 0.0
                pred_conf = float(def_probs[i][def_lbl].item() * 100.0)
                s["defended_confidence"] = round(pred_conf, 2)
                s["defended_true_prob"] = round(true_prob, 2)
                s["trajectory"].append({"stage": f"Defended (Quant-{self.quantization_levels})", "step": self.pgd_steps + 1, "pred": def_cls, "confidence": round(pred_conf, 2), "true_class_prob": round(true_prob, 2)})

        raw_fgsm_acc = self.metrics_by_action.get("RUN_FGSM", {}).get("adversarial_accuracy", 0.0)
        raw_pgd_acc = self.metrics_by_action.get("ESCALATE_PGD", {}).get("adversarial_accuracy", 0.0)
        fgsm_recovery = def_fgsm_acc - raw_fgsm_acc
        pgd_recovery = def_pgd_acc - raw_pgd_acc

        trust_checks = {
            "defense_clean_consistency": check_metric_consistency(def_clean_acc, def_clean_acc, 0.0),
            "defense_fgsm_consistency": check_metric_consistency(def_clean_acc, def_fgsm_acc, def_fgsm_asr),
            "defense_pgd_consistency": check_metric_consistency(def_clean_acc, def_pgd_acc, def_pgd_asr),
        }
        trust_flag = generate_trust_flag(trust_checks)

        return StepEvaluationResult(
            step_number=step_number,
            action_type=action.value,
            phase=EvaluationPhase.DEFENSE_VERIFICATION,
            metrics={
                "defense_name": f"Input Quantization ({self.quantization_levels} levels)",
                "defense_clean_accuracy": def_clean_acc,
                "defended_fgsm_accuracy": def_fgsm_acc,
                "defended_fgsm_asr": def_fgsm_asr,
                "defended_pgd_accuracy": def_pgd_acc,
                "defended_pgd_asr": def_pgd_asr,
                "fgsm_recovery_gain": fgsm_recovery,
                "pgd_recovery_gain": pgd_recovery,
            },
            trust_checks=trust_checks,
            trust_flag=trust_flag,
            execution_time_seconds=time.time() - step_start,
            status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
            timestamp=timestamp,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Final assessment & persistence (shared for both modes)
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_final_assessment(self) -> FinalAssessment:
        timestamp = datetime.utcnow().isoformat() + "Z"
        duration = self.end_time - self.start_time

        # Only steps that SUCCEEDED may contribute evidence; a failed attack must never
        # read as "0% attack success" and turn into a robustness verdict.
        ok_actions = {st.action_type for st in self.step_history if st.status == "SUCCESS"}
        pick = lambda a: self.metrics_by_action.get(a, {}) if a in ok_actions else {}
        baseline_metrics = pick("RUN_BASELINE")
        fgsm_metrics = pick("RUN_FGSM")
        pgd_metrics = pick("ESCALATE_PGD")
        defense_metrics = pick("TEST_DEFENSE")

        if self.ram_mode:
            # Primary metric: tag preservation. clean_accuracy is the definitional 100% reference
            # (baseline tags vs themselves); the baseline tag COUNT is reported separately.
            clean_acc = 100.0 if baseline_metrics else None
            fgsm_pres = fgsm_metrics.get("tag_preservation_rate")
            pgd_pres = pgd_metrics.get("tag_preservation_rate")
            fgsm_acc = round(fgsm_pres * 100.0, 2) if fgsm_pres is not None else None
            pgd_acc = round(pgd_pres * 100.0, 2) if pgd_pres is not None else None
            fgsm_asr = fgsm_metrics.get("attack_success_rate", 0.0)
            pgd_asr = pgd_metrics.get("attack_success_rate", 0.0)
        else:
            clean_acc = baseline_metrics.get("clean_accuracy")
            fgsm_acc = fgsm_metrics.get("adversarial_accuracy")
            pgd_acc = pgd_metrics.get("adversarial_accuracy")
            fgsm_asr = fgsm_metrics.get("attack_success_rate", 0.0)
            pgd_asr = pgd_metrics.get("attack_success_rate", 0.0)

        worst_adv_acc = None
        adv_accuracies = [acc for acc in [fgsm_acc, pgd_acc] if acc is not None]
        if adv_accuracies:
            worst_adv_acc = min(adv_accuracies)

        max_asr = max(fgsm_asr, pgd_asr) if (fgsm_metrics or pgd_metrics) else None

        tested_attacks = []
        if "RUN_FGSM" in self.completed_actions:
            tested_attacks.append(f"FGSM (eps={self.epsilon:.4f})")
        if "ESCALATE_PGD" in self.completed_actions:
            tested_attacks.append(f"PGD (eps={self.epsilon:.4f}, steps={self.pgd_steps}, alpha={self.pgd_alpha:.4f})")

        tested_defenses = []
        if "TEST_DEFENSE" in self.completed_actions:
            tested_defenses.append(f"Input Quantization ({self.quantization_levels} levels)")

        trust_warnings = []
        for step in self.step_history:
            if step.trust_flag.get("status") in ("WARNING", "FAIL"):
                trust_warnings.append(f"Step {step.step_number} ({step.action_type}): {step.trust_flag.get('summary')}")

        key_findings = []
        if self.ram_mode:
            if baseline_metrics:
                key_findings.append(f"RAM++ baseline established: {int(baseline_metrics.get('tag_count', 0))} active tags detected on the investigation image.")
            if fgsm_acc is not None:
                key_findings.append(f"FGSM single-step attack: tag preservation rate {fgsm_acc:.1f}% (ASR: {fgsm_asr:.1f}%).")
            if pgd_acc is not None:
                key_findings.append(f"PGD-{self.pgd_steps} iterative attack: tag preservation rate {pgd_acc:.1f}% (ASR: {pgd_asr:.1f}%).")
            if defense_metrics:
                parts = []
                for nm in ("fgsm", "pgd"):
                    g = defense_metrics.get(f"{nm}_recovery_gain")
                    parts.append(f"{g:+.2%} vs {nm.upper()}" if g is not None else f"not measured vs {nm.upper()}")
                key_findings.append("Input quantization changed tag preservation by " + " and ".join(parts) + ".")
            for failed in (st for st in self.step_history if st.status == "FAIL" and st.action_type in ("RUN_FGSM", "ESCALATE_PGD", "TEST_DEFENSE")):
                key_findings.append(f"{failed.action_type} did not complete ({failed.error_message}); no result is claimed for it.")
        else:
            if clean_acc is not None:
                key_findings.append(f"Clean baseline accuracy established at {clean_acc:.2f}%.")
            if fgsm_acc is not None:
                key_findings.append(f"FGSM single-step attack resulted in {fgsm_acc:.2f}% accuracy (ASR: {fgsm_asr:.2f}%).")
            if pgd_acc is not None:
                key_findings.append(f"Iterative PGD-{self.pgd_steps} attack lowered accuracy to {pgd_acc:.2f}% (ASR: {pgd_asr:.2f}%).")
            if defense_metrics:
                def_fgsm_gain = defense_metrics.get("fgsm_recovery_gain", 0.0)
                def_pgd_gain = defense_metrics.get("pgd_recovery_gain", 0.0)
                key_findings.append(f"Input quantization provided accuracy delta of {def_fgsm_gain:+.2f}% vs FGSM and {def_pgd_gain:+.2f}% vs PGD.")

        # Overall status
        if self.current_phase == EvaluationPhase.FAILED:
            overall_status = "INTEGRITY_FAILED"
            overall_summary = "Evaluation was aborted due to critical preflight or execution integrity failures."
        elif self.current_phase == EvaluationPhase.BUDGET_EXHAUSTED:
            overall_status = "INCONCLUSIVE"
            overall_summary = (
                f"Evaluation stopped because test budget was reached ({len(self.completed_actions)}/{self.max_budget} steps). "
                "Robustness evidence is partial."
            )
        elif max_asr is not None and max_asr >= 40.0:
            overall_status = "CRITICAL_VULNERABILITY"
            if self.ram_mode:
                overall_summary = (
                    f"High tag suppression susceptibility detected. Peak Attack Success Rate: {max_asr:.2f}%. "
                    f"Tag preservation dropped to {worst_adv_acc:.1f}%."
                )
            else:
                overall_summary = (
                    f"High susceptibility to gradient-based attacks detected. Peak Attack Success Rate: {max_asr:.2f}%. "
                    f"Adversarial accuracy degraded to {worst_adv_acc:.2f}%."
                )
        elif max_asr is not None and max_asr >= 15.0:
            overall_status = "MODERATE_VULNERABILITY"
            overall_summary = f"Moderate vulnerability observed under Linf perturbation. Peak Attack Success Rate: {max_asr:.2f}%."
        elif clean_acc is not None and not self.ram_mode and clean_acc < 50.0:
            overall_status = "INCONCLUSIVE"
            overall_summary = "Clean baseline accuracy is below 50%, indicating a possible model configuration issue."
        elif max_asr is not None and max_asr < 15.0:
            overall_status = "ROBUST"
            if self.ram_mode:
                overall_summary = (f"Low tag disruption observed on this single image under the tested attacks "
                                   f"(peak {max_asr:.2f}% of baseline tags lost). This is not a robustness certificate.")
            else:
                overall_summary = f"Model demonstrates strong robustness under Linf perturbation. Peak ASR: {max_asr:.2f}%."
        else:
            overall_status = "INCONCLUSIVE"
            overall_summary = "Insufficient data to determine robustness verdict."

        untested_vectors = []
        if "RUN_FGSM" not in self.completed_actions:
            untested_vectors.append("FGSM single-step gradient attack")
        if "ESCALATE_PGD" not in self.completed_actions:
            untested_vectors.append(f"PGD-{self.pgd_steps} iterative gradient attack")
        if "TEST_DEFENSE" not in self.completed_actions:
            untested_vectors.append("Input quantization defense evaluation")
        untested_vectors += [
            "AutoAttack ensemble (squares, FAB, APGD)",
            "Black-box transfer attacks",
            "Adaptive defense-breaking attacks (BPDA)",
            "Semantic perturbations and style-based attacks",
        ]

        scientific_limitations = [
            "Empirical Linf evaluations do not certify robustness against all bounded perturbations.",
            "Gradient masking may cause overestimated robustness under first-order attacks.",
            "Single-image investigation mode: results do not generalize to dataset-wide statistics.",
            "Quantization defense is not adaptive-attack-resistant (BPDA can bypass it).",
        ]
        if self.ram_mode:
            scientific_limitations.append(
                "RAM++ tag thresholding: tags near the decision boundary may flip with minimal perturbation, "
                "independent of semantic attack success."
            )

        recommendations = []
        if max_asr is not None and max_asr >= 15.0:
            recommendations.append("Conduct adversarial training with PGD-augmented data to improve robustness.")
        if "TEST_DEFENSE" not in self.completed_actions:
            recommendations.append("Evaluate input preprocessing defenses (e.g., quantization, denoising) against tested attacks.")
        recommendations += [
            "Additional empirical evaluation is not a robustness certificate.",
            "Evaluate against black-box and transfer attacks to test cross-model generalization.",
        ]

        # Discoveries
        self.discoveries = []
        if self.preflight_result is not None and self.preflight_result.passed:
            self.discoveries.append({
                "id": "DISC-01",
                "title": "Target Asset Profiled",
                "unlocked": True,
                "badge": "ASSET ACQUIRED",
                "summary": f"Model '{self.model_name}' cleared all preflight integrity checks.",
                "intel": "Preflight validation confirms tensor integrity, pixel-range compliance, model callability, and hyperparameter safety bounds before any adversarial operations.",
            })
        if any(step.action_type == "RUN_BASELINE" and step.status == "SUCCESS" for step in self.step_history):
            if self.ram_mode:
                tag_n = baseline_metrics.get("tag_count", 0)
                self.discoveries.append({
                    "id": "DISC-02",
                    "title": "RAM++ Recognition Profile Established",
                    "unlocked": True,
                    "badge": "BASELINE LOCKED",
                    "summary": f"RAM++ detected {tag_n} semantic tags on the investigation image.",
                    "intel": "Baseline tag profile is the ground truth for all adversarial comparisons. Tag preservation rate measures how many baseline tags survive an attack.",
                })
            else:
                self.discoveries.append({
                    "id": "DISC-02",
                    "title": "Clean Accuracy Benchmark Established",
                    "unlocked": True,
                    "badge": "BASELINE LOCKED",
                    "summary": f"Non-adversarial baseline accuracy: {clean_acc:.2f}%.",
                    "intel": "Clean baseline provides the reference for adversarial degradation measurements.",
                })
        if pgd_metrics:
            pgd_pres_v = pgd_metrics.get("tag_preservation_rate", pgd_metrics.get("adversarial_accuracy", 100))
            pgd_metric_txt = (f"{pgd_pres_v * 100:.1f}% tag preservation" if self.ram_mode else f"{pgd_pres_v:.2f}")
            self.discoveries.append({
                "id": "DISC-03",
                "title": "Iterative Non-Linear Boundary Exploitation",
                "unlocked": True,
                "badge": "ITERATIVE EXPLOIT",
                "summary": f"PGD-{self.pgd_steps} drove the metric to {pgd_metric_txt} (peak {'tag disruption' if self.ram_mode else 'ASR'}: {pgd_asr:.2f}%).",
                "intel": "Multi-step projected gradient ascent solves an inner maximization problem. Even if a model obfuscates linear gradients, multi-step projection exposes deeper loss surface vulnerabilities.",
            })
        gain = defense_metrics.get("pgd_recovery_gain", defense_metrics.get("fgsm_recovery_gain")) if defense_metrics else None
        if defense_metrics and gain is not None:
            unlocked = gain > 0.0
            self.discoveries.append({
                "id": "DISC-04",
                "title": "Defensive Mitigation via Spatial Discretization",
                "unlocked": unlocked,
                "badge": "DEFENSE EFFECTIVE" if unlocked else "DEFENSE INSUFFICIENT",
                "summary": f"Quantization ({self.quantization_levels} levels) yielded a {gain:+.4f} change in tag-preservation fraction vs the strongest attack measured.",
                "intel": "Input quantization groups continuous pixel values into discrete bins. While it can mitigate high-frequency noise, it does not guarantee protection against adaptive attacks (e.g. BPDA).",
            })
        if self.current_phase in (EvaluationPhase.COMPLETED, EvaluationPhase.BUDGET_EXHAUSTED):
            self.discoveries.append({
                "id": "DISC-05",
                "title": "Investigation Protocol Debrief Compiled",
                "unlocked": True,
                "badge": "MISSION DEBRIEF",
                "summary": f"Investigation completed with {len(self.completed_actions)}/{self.max_budget} budget operations consumed.",
                "intel": "Audit trail is fully sealed and persisted. Provenance, trust checks, and decision rationales are available for compliance and academic evaluation.",
            })

        return FinalAssessment(
            run_id=self.run_id,
            model_name=self.model_name,
            overall_status=overall_status,
            overall_summary=overall_summary,
            clean_accuracy=clean_acc,
            worst_adversarial_accuracy=worst_adv_acc,
            max_attack_success_rate=max_asr,
            tested_attacks=tested_attacks,
            tested_defenses=tested_defenses,
            key_findings=key_findings,
            trust_warnings=trust_warnings,
            untested_vectors=untested_vectors,
            scientific_limitations=scientific_limitations,
            recommendations=recommendations,
            budget_allocated=self.max_budget,
            budget_consumed=len(self.completed_actions),
            execution_duration_seconds=duration,
            timestamp=timestamp,
        )

    def _persist_run(self) -> None:
        """Serialize complete run evidence, decisions, and assessment to disk."""
        run_data = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "ram_mode": self.ram_mode,
            "config": {
                "epsilon": self.epsilon,
                "pgd_steps": self.pgd_steps,
                "pgd_alpha": self.pgd_alpha,
                "quantization_levels": self.quantization_levels,
                "max_budget": self.max_budget,
            },
            "completed_actions": self.completed_actions,
            "preflight_result": self.preflight_result.to_dict() if self.preflight_result else None,
            "decisions": [d.to_dict() for d in self.decision_history],
            "step_history": [s.to_dict() for s in self.step_history],
            "final_assessment": self.final_assessment.to_dict() if self.final_assessment else None,
            "anomalies_detected": self.anomalies_detected,
            "sample_inspection": self.sample_inspection,
            "detailed_samples": self.detailed_samples,
            "class_metrics": self.class_metrics,
            "flip_matrix": self.flip_matrix,
            "discoveries": self.discoveries,
            "scoring_report": self.scoring_report,
        }

        filepath = os.path.join(self.results_dir, f"{self.run_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, default=_json_default)

    def get_run_summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "ram_mode": self.ram_mode,
            "current_phase": self.current_phase.value,
            "steps_completed": len(self.completed_actions),
            "remaining_budget": self.remaining_budget,
            "decisions_count": len(self.decision_history),
            "final_assessment": self.final_assessment.to_dict() if self.final_assessment else None,
        }
