"""
ModelShield Adaptive Red-Team Orchestrator Engine
Coordinates preflight checks, baseline evaluation, adaptive attack escalation,
defense evaluation, budget accounting, and evidence collection.
"""
import base64
import io
import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
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

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def tensor_to_base64_png(tensor: torch.Tensor, size=(128, 128)) -> str:
    """Convert a [3, 32, 32] float tensor in [0, 1] to base64 PNG data URI with nearest-neighbor scaling."""
    try:
        t = tensor.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)
        arr = (t.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr).resize(size, Image.NEAREST)
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


def get_diff_matrix_2d(clean: torch.Tensor, adv: torch.Tensor) -> List[List[float]]:
    """Return 32x32 channel-averaged perturbation matrix for Plotly heatmap."""
    try:
        c = clean.detach().cpu().squeeze()
        a = adv.detach().cpu().squeeze()
        diff = (a - c).abs().mean(dim=0).numpy()
        return [[round(float(v), 5) for v in row] for row in diff]
    except Exception:
        return []


def extract_conv1_features(model: torch.nn.Module, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Extract actual intermediate feature map from the first Conv2d layer of the model."""
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
    """Convert a 2D float tensor [H, W] in [0, 1] to base64 PNG using a colormap."""
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


class RedTeamOrchestrator:
    """
    Coordinates an automated, adaptive adversarial security evaluation of a model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        images: torch.Tensor,
        labels: torch.Tensor,
        config: Optional[Dict[str, Any]] = None,
        max_budget: int = 5,
        model_name: str = "Carmon2019Unlabeled",
        dataset_name: str = "CIFAR-10",
        results_dir: str = "results/runs",
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

        os.makedirs(self.results_dir, exist_ok=True)

    def run(self, progress_callback: Optional[Callable[[str, int, int], None]] = None) -> FinalAssessment:
        """
        Execute full adaptive red-team workflow from preflight to final assessment.
        """
        self.start_time = time.time()
        self.current_phase = EvaluationPhase.PREFLIGHT
        step_counter = 1

        while True:
            # Step 1: Decision Policy query
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

            # Check termination actions
            if decision.action == DecisionAction.HALT_INTEGRITY_FAILURE:
                self.current_phase = EvaluationPhase.FAILED
                break
            elif decision.action == DecisionAction.STOP_BUDGET_EXHAUSTED:
                self.current_phase = EvaluationPhase.BUDGET_EXHAUSTED
                break
            elif decision.action == DecisionAction.STOP_SUFFICIENT_EVIDENCE:
                self.current_phase = EvaluationPhase.COMPLETED
                break

            # Deduct budget for executing test
            self.remaining_budget = max(0, self.remaining_budget - decision.budget_cost)

            # Execute the selected action
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

    def _execute_action(self, action: DecisionAction, step_number: int) -> StepEvaluationResult:
        step_start = time.time()
        timestamp = datetime.utcnow().isoformat() + "Z"

        if action == DecisionAction.RUN_PREFLIGHT:
            self.current_phase = EvaluationPhase.PREFLIGHT
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

        elif action == DecisionAction.RUN_BASELINE:
            self.current_phase = EvaluationPhase.BASELINE
            self.model.eval()

            images_dev = self.images.to(self.device)
            labels_dev = self.labels.to(self.device)

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

            # Neural Attack Visualizer sample data extraction
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
                    "fgsm_pred": None,
                    "fgsm_class": None,
                    "fgsm_confidence": 0.0,
                    "fgsm_top5": [],
                    "fgsm_l_inf": 0.0,
                    "fgsm_img_b64": "",
                    "fgsm_feature_b64": "",
                    "fgsm_heatmap_b64": "",
                    "fgsm_diff_matrix": [],
                    "pgd_pred": None,
                    "pgd_class": None,
                    "pgd_confidence": 0.0,
                    "pgd_top5": [],
                    "pgd_l_inf": 0.0,
                    "pgd_img_b64": "",
                    "pgd_feature_b64": "",
                    "feature_distortion_b64": "",
                    "pgd_heatmap_b64": "",
                    "pgd_diff_matrix": [],
                    "fooled": False,
                    "trajectory": [{
                        "stage": "Clean Baseline",
                        "step": 0,
                        "pred": pred_cls,
                        "confidence": round(conf * 100.0, 2),
                        "true_class_prob": round(true_prob * 100.0, 2),
                    }],
                })

            # Class-wise metrics initialization
            self.class_metrics = {}
            for c_idx in range(min(10, num_classes)):
                c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else f"Class {c_idx}"
                mask = (labels_dev == c_idx)
                c_tot = mask.sum().item()
                if c_tot > 0:
                    c_cor = (preds[mask] == labels_dev[mask]).sum().item()
                    self.class_metrics[c_name] = {
                        "total": c_tot,
                        "clean_correct": c_cor,
                        "clean_acc": round(100.0 * c_cor / c_tot, 2),
                        "fgsm_correct": 0,
                        "fgsm_acc": 0.0,
                        "pgd_correct": 0,
                        "pgd_acc": 0.0,
                    }

            # Baseline trust checks
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
                metrics={
                    "clean_accuracy": clean_acc,
                    "total_samples": total,
                    "correct_samples": correct,
                },
                trust_checks=trust_checks,
                trust_flag=trust_flag,
                execution_time_seconds=time.time() - step_start,
                status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
                timestamp=timestamp,
            )

        elif action == DecisionAction.RUN_FGSM:
            self.current_phase = EvaluationPhase.ATTACK_EVALUATION
            images_dev = self.images.to(self.device)
            labels_dev = self.labels.to(self.device)

            # Use clean predictions as reference label per project standard
            attack_labels = (
                self.clean_predictions.to(self.device)
                if self.clean_predictions is not None
                else labels_dev
            )

            adv_images = fgsm_attack(
                self.model,
                images_dev,
                attack_labels,
                epsilon=self.epsilon,
            )
            self.fgsm_adv_images = adv_images.detach().cpu()

            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                adv_preds = adv_outputs.argmax(dim=1)

            total = labels_dev.size(0)
            adv_correct = (adv_preds == labels_dev).sum().item()
            adv_acc = 100.0 * adv_correct / total if total > 0 else 0.0

            sample_count = min(8, total)
            self.sample_inspection["fgsm_preds"] = [int(x) for x in adv_preds[:sample_count].tolist()]

            # Detailed sample updating for FGSM
            num_classes = adv_outputs.size(1)
            fgsm_softmax = torch.softmax(adv_outputs, dim=1).detach().cpu() if num_classes > 1 else adv_outputs.detach().cpu()

            for i in range(min(sample_count, len(self.detailed_samples))):
                s = self.detailed_samples[i]
                pred_lbl = int(adv_preds[i].item())
                pred_cls = CLASS_NAMES[pred_lbl] if pred_lbl < len(CLASS_NAMES) else str(pred_lbl)
                conf = float(fgsm_softmax[i][pred_lbl].item()) if pred_lbl < num_classes else 1.0

                top_k = min(5, num_classes)
                top_probs, top_indices = torch.topk(fgsm_softmax[i], k=top_k)
                top5_list = []
                for p_val, idx_val in zip(top_probs, top_indices):
                    c_idx = int(idx_val.item())
                    c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else str(c_idx)
                    top5_list.append({"class": c_name, "prob": round(float(p_val.item()) * 100, 2)})

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
                s["trajectory"].append({
                    "stage": "FGSM 1-Step",
                    "step": 1,
                    "pred": pred_cls,
                    "confidence": round(conf * 100.0, 2),
                    "true_class_prob": round(true_prob * 100.0, 2),
                })

            # Update class-wise FGSM stats
            for c_idx in range(min(10, num_classes)):
                c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else f"Class {c_idx}"
                if c_name in self.class_metrics:
                    mask = (labels_dev == c_idx)
                    c_tot = mask.sum().item()
                    if c_tot > 0:
                        c_cor = (adv_preds[mask] == labels_dev[mask]).sum().item()
                        self.class_metrics[c_name]["fgsm_correct"] = c_cor
                        self.class_metrics[c_name]["fgsm_acc"] = round(100.0 * c_cor / c_tot, 2)

            # Attack success rate among initially correct predictions
            clean_preds_dev = (
                self.clean_predictions.to(self.device)
                if self.clean_predictions is not None
                else labels_dev
            )
            asr = calculate_attack_success(clean_preds_dev, adv_preds, labels_dev)

            # Trust checks on adversarial images
            budget_check = check_perturbation_budget(images_dev, adv_images, epsilon=self.epsilon)
            pixel_check = check_pixel_range(adv_images)
            output_check = check_prediction_outputs(adv_outputs.cpu(), self.labels)
            clean_acc = self.metrics_by_action.get("RUN_BASELINE", {}).get("clean_accuracy", 100.0)
            consistency_check = check_metric_consistency(clean_acc, adv_acc, asr)

            trust_checks = {
                "perturbation_budget": budget_check,
                "pixel_range": pixel_check,
                "prediction_outputs": output_check,
                "metric_consistency": consistency_check,
            }
            trust_flag = generate_trust_flag(trust_checks)

            return StepEvaluationResult(
                step_number=step_number,
                action_type=action.value,
                phase=EvaluationPhase.ATTACK_EVALUATION,
                metrics={
                    "adversarial_accuracy": adv_acc,
                    "attack_success_rate": asr,
                    "epsilon": self.epsilon,
                    "max_perturbation": budget_check.get("max_perturbation", 0.0),
                    "total_samples": total,
                },
                trust_checks=trust_checks,
                trust_flag=trust_flag,
                execution_time_seconds=time.time() - step_start,
                status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
                timestamp=timestamp,
            )

        elif action == DecisionAction.ESCALATE_PGD:
            self.current_phase = EvaluationPhase.ATTACK_EVALUATION
            images_dev = self.images.to(self.device)
            labels_dev = self.labels.to(self.device)

            attack_labels = (
                self.clean_predictions.to(self.device)
                if self.clean_predictions is not None
                else labels_dev
            )

            adv_images = pgd_attack(
                self.model,
                images_dev,
                attack_labels,
                epsilon=self.epsilon,
                alpha=self.pgd_alpha,
                steps=self.pgd_steps,
                random_start=True,
            )
            self.pgd_adv_images = adv_images.detach().cpu()

            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                adv_preds = adv_outputs.argmax(dim=1)

            total = labels_dev.size(0)
            adv_correct = (adv_preds == labels_dev).sum().item()
            adv_acc = 100.0 * adv_correct / total if total > 0 else 0.0

            sample_count = min(8, total)
            self.sample_inspection["pgd_preds"] = [int(x) for x in adv_preds[:sample_count].tolist()]

            # Detailed sample updating for PGD
            num_classes = adv_outputs.size(1)
            pgd_softmax = torch.softmax(adv_outputs, dim=1).detach().cpu() if num_classes > 1 else adv_outputs.detach().cpu()

            for i in range(min(sample_count, len(self.detailed_samples))):
                s = self.detailed_samples[i]
                pred_lbl = int(adv_preds[i].item())
                pred_cls = CLASS_NAMES[pred_lbl] if pred_lbl < len(CLASS_NAMES) else str(pred_lbl)
                conf = float(pgd_softmax[i][pred_lbl].item()) if pred_lbl < num_classes else 1.0

                top_k = min(5, num_classes)
                top_probs, top_indices = torch.topk(pgd_softmax[i], k=top_k)
                top5_list = []
                for p_val, idx_val in zip(top_probs, top_indices):
                    c_idx = int(idx_val.item())
                    c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else str(c_idx)
                    top5_list.append({"class": c_name, "prob": round(float(p_val.item()) * 100, 2)})

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
                s["trajectory"].append({
                    "stage": f"PGD-{self.pgd_steps} Final",
                    "step": self.pgd_steps,
                    "pred": pred_cls,
                    "confidence": round(conf * 100.0, 2),
                    "true_class_prob": round(true_prob * 100.0, 2),
                })

            # Update class-wise PGD stats
            for c_idx in range(min(10, num_classes)):
                c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else f"Class {c_idx}"
                if c_name in self.class_metrics:
                    mask = (labels_dev == c_idx)
                    c_tot = mask.sum().item()
                    if c_tot > 0:
                        c_cor = (adv_preds[mask] == labels_dev[mask]).sum().item()
                        self.class_metrics[c_name]["pgd_correct"] = c_cor
                        self.class_metrics[c_name]["pgd_acc"] = round(100.0 * c_cor / c_tot, 2)

            # Build Prediction Flip Matrix (Clean class vs Adversarial target class)
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

            clean_preds_dev = (
                self.clean_predictions.to(self.device)
                if self.clean_predictions is not None
                else labels_dev
            )
            asr = calculate_attack_success(clean_preds_dev, adv_preds, labels_dev)

            # Trust checks on PGD images
            budget_check = check_perturbation_budget(images_dev, adv_images, epsilon=self.epsilon)
            pixel_check = check_pixel_range(adv_images)
            output_check = check_prediction_outputs(adv_outputs.cpu(), self.labels)
            clean_acc = self.metrics_by_action.get("RUN_BASELINE", {}).get("clean_accuracy", 100.0)
            consistency_check = check_metric_consistency(clean_acc, adv_acc, asr)

            trust_checks = {
                "perturbation_budget": budget_check,
                "pixel_range": pixel_check,
                "prediction_outputs": output_check,
                "metric_consistency": consistency_check,
            }
            trust_flag = generate_trust_flag(trust_checks)

            return StepEvaluationResult(
                step_number=step_number,
                action_type=action.value,
                phase=EvaluationPhase.ATTACK_EVALUATION,
                metrics={
                    "adversarial_accuracy": adv_acc,
                    "attack_success_rate": asr,
                    "epsilon": self.epsilon,
                    "pgd_steps": self.pgd_steps,
                    "pgd_alpha": self.pgd_alpha,
                    "max_perturbation": budget_check.get("max_perturbation", 0.0),
                    "total_samples": total,
                },
                trust_checks=trust_checks,
                trust_flag=trust_flag,
                execution_time_seconds=time.time() - step_start,
                status="SUCCESS" if trust_flag["status"] != "FAIL" else "FAIL",
                timestamp=timestamp,
            )

        elif action == DecisionAction.TEST_DEFENSE:
            self.current_phase = EvaluationPhase.DEFENSE_VERIFICATION
            images_dev = self.images.to(self.device)
            labels_dev = self.labels.to(self.device)

            defense_fn = lambda x: quantization_defense(x, levels=self.quantization_levels)

            # Clean accuracy with defense
            def_clean_acc, def_clean_preds = calculate_accuracy(
                self.model, images_dev, labels_dev, defense_fn
            )

            # Evaluate defense against FGSM
            def_fgsm_acc, def_fgsm_preds = 0.0, None
            def_fgsm_asr = 0.0
            if self.fgsm_adv_images is not None:
                fgsm_dev = self.fgsm_adv_images.to(self.device)
                def_fgsm_acc, def_fgsm_preds = calculate_accuracy(
                    self.model, fgsm_dev, labels_dev, defense_fn
                )
                def_fgsm_asr = calculate_attack_success(
                    def_clean_preds, def_fgsm_preds, labels_dev
                )

            # Evaluate defense against PGD
            def_pgd_acc, def_pgd_preds = 0.0, None
            def_pgd_asr = 0.0
            if self.pgd_adv_images is not None:
                pgd_dev = self.pgd_adv_images.to(self.device)
                def_pgd_acc, def_pgd_preds = calculate_accuracy(
                    self.model, pgd_dev, labels_dev, defense_fn
                )
                def_pgd_asr = calculate_attack_success(
                    def_clean_preds, def_pgd_preds, labels_dev
                )

            # Detailed sample updating for Defense with real model forward pass and softmax probabilities
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
                    s["trajectory"].append({
                        "stage": f"Defended (Quant-{self.quantization_levels})",
                        "step": self.pgd_steps + 1,
                        "pred": def_cls,
                        "confidence": round(pred_conf, 2),
                        "true_class_prob": round(true_prob, 2),
                    })

            # Recovery gains
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

        elif action == DecisionAction.REPEAT_VERIFICATION:
            self.current_phase = EvaluationPhase.ANOMALY_VERIFICATION
            images_dev = self.images.to(self.device)
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

    def _generate_final_assessment(self) -> FinalAssessment:
        timestamp = datetime.utcnow().isoformat() + "Z"
        duration = self.end_time - self.start_time

        # Extract baseline & attack measurements
        baseline_metrics = self.metrics_by_action.get("RUN_BASELINE", {})
        fgsm_metrics = self.metrics_by_action.get("RUN_FGSM", {})
        pgd_metrics = self.metrics_by_action.get("ESCALATE_PGD", {})
        defense_metrics = self.metrics_by_action.get("TEST_DEFENSE", {})

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

        # Compile trust warnings
        trust_warnings = []
        for step in self.step_history:
            if step.trust_flag.get("status") in ("WARNING", "FAIL"):
                trust_warnings.append(
                    f"Step {step.step_number} ({step.action_type}): {step.trust_flag.get('summary')}"
                )

        # Formulate key findings
        key_findings = []
        if clean_acc is not None:
            key_findings.append(f"Clean baseline accuracy established at {clean_acc:.2f}%.")

        if fgsm_acc is not None:
            key_findings.append(
                f"FGSM single-step attack resulted in {fgsm_acc:.2f}% accuracy (ASR: {fgsm_asr:.2f}%)."
            )

        if pgd_acc is not None:
            key_findings.append(
                f"Iterative PGD-10 attack lowered accuracy to {pgd_acc:.2f}% (ASR: {pgd_asr:.2f}%)."
            )

        if defense_metrics:
            def_fgsm_gain = defense_metrics.get("fgsm_recovery_gain", 0.0)
            def_pgd_gain = defense_metrics.get("pgd_recovery_gain", 0.0)
            key_findings.append(
                f"Input quantization provided accuracy delta of {def_fgsm_gain:+.2f}% vs FGSM and {def_pgd_gain:+.2f}% vs PGD."
            )

        # Status determination
        if self.current_phase == EvaluationPhase.FAILED:
            overall_status = "INTEGRITY_FAILED"
            overall_summary = "Evaluation was aborted due to critical preflight or execution integrity failures."
        elif self.current_phase == EvaluationPhase.BUDGET_EXHAUSTED:
            overall_status = "INCONCLUSIVE"
            overall_summary = (
                f"Evaluation stopped because test budget was reached ({len(self.completed_actions)}/{self.max_budget} steps). "
                f"Robustness evidence is partial."
            )
        elif max_asr is not None and max_asr >= 40.0:
            overall_status = "CRITICAL_VULNERABILITY"
            overall_summary = (
                f"High susceptibility to gradient-based attacks detected. Peak Attack Success Rate: {max_asr:.2f}%. "
                f"Adversarial accuracy degraded to {worst_adv_acc:.2f}%."
            )
        elif max_asr is not None and max_asr >= 15.0:
            overall_status = "MODERATE_VULNERABILITY"
            overall_summary = (
                f"Moderate vulnerability observed under Linf perturbation. Peak Attack Success Rate: {max_asr:.2f}%."
            )
        elif clean_acc is not None and clean_acc < 50.0:
            overall_status = "INCONCLUSIVE"
            overall_summary = "Clean baseline accuracy is insufficient (< 50%) for conclusive security assertions."
        else:
            overall_status = "HIGH_ROBUSTNESS_OBSERVED"
            overall_summary = (
                f"Model maintained stability against tested attacks within the Linf={self.epsilon:.4f} budget. "
                f"Peak Attack Success Rate was limited to {max_asr if max_asr is not None else 0.0:.2f}%."
            )

        # Untested vectors
        untested_vectors = [
            "AutoAttack (Standardized ensemble of APGD-CE, APGD-DLR, FAB, and Square)",
            "Carlini & Wagner L2 Optimization Attack",
            "DeepFool Minimal Perturbation Metric",
            "Black-box Decision / Score-based Attacks (NES, SimBA)",
            "Spatial Transformation and Patch Perturbations",
        ]

        # Scientific limitations
        scientific_limitations = [
            "Robustness against FGSM and PGD under Linf does NOT certify complete adversarial security.",
            "Gradient masking must not be ruled out without specific loss landscape analysis (e.g. verifying unbounded loss or non-zero gradients).",
            "Evaluation was conducted on a finite sample subset; domain shift may expose further vulnerabilities.",
            "Defenses relying on input quantization may be bypassed by BPDA (Backward Pass Differentiable Approximation).",
        ]

        # Recommendations
        recommendations = [
            "Integrate adversarial training with multi-step PGD during model optimization.",
            "Conduct evaluation against gradient-free black-box attacks (Square Attack) to detect obfuscated gradients.",
            "Evaluate adaptive attacks specifically designed to circumvent input quantization.",
            "Establish continuous red-team monitoring in inference pipelines to detect out-of-distribution adversarial perturbations.",
        ]

        # Compile evidence-grounded TryHackMe style discoveries
        self.discoveries = []
        if clean_acc is not None:
            self.discoveries.append({
                "id": "DISC-01",
                "title": "Clean Reference Anchored",
                "unlocked": True,
                "badge": "TARGET ANCHORED",
                "summary": f"Clean reference accuracy established at {clean_acc:.2f}%.",
                "intel": "Establishes baseline class boundaries. All adversarial attack success rates are strictly normalized against initially correct samples from this baseline."
            })
        if fgsm_metrics:
            unlocked = fgsm_asr > 0.0
            self.discoveries.append({
                "id": "DISC-02",
                "title": "First-Order Gradient Leakage Detected",
                "unlocked": unlocked,
                "badge": "LINEAR BREACH" if unlocked else "LINEAR RESILIENT",
                "summary": f"FGSM probe achieved {fgsm_asr:.2f}% Attack Success Rate ({fgsm_acc:.2f}% adversarial accuracy).",
                "intel": "First-order gradient ascent shifts pixels along the sign of the loss gradient. Non-zero ASR demonstrates that decision boundaries are sensitive to linear perturbations."
            })
        if pgd_metrics:
            unlocked = pgd_asr >= fgsm_asr or pgd_acc < (fgsm_acc if fgsm_acc else 100.0)
            self.discoveries.append({
                "id": "DISC-03",
                "title": "Iterative Non-Linear Boundary Exploitation",
                "unlocked": True,
                "badge": "ITERATIVE EXPLOIT",
                "summary": f"PGD-{self.pgd_steps} ascent drove adversarial accuracy to {pgd_acc:.2f}% (Peak ASR: {pgd_asr:.2f}%).",
                "intel": "Multi-step projected gradient ascent solves an inner maximization problem. Even if a model obfuscates linear gradients, multi-step projection exposes deeper loss surface vulnerabilities."
            })
        if defense_metrics:
            gain = defense_metrics.get("pgd_recovery_gain", 0.0)
            unlocked = gain > 0.0
            self.discoveries.append({
                "id": "DISC-04",
                "title": "Defensive Mitigation via Spatial Discretization",
                "unlocked": unlocked,
                "badge": "DEFENSE EFFECTIVE" if unlocked else "DEFENSE INSUFFICIENT",
                "summary": f"Quantization ({self.quantization_levels} levels) yielded a {gain:+.2f}% accuracy recovery against PGD.",
                "intel": "Input quantization groups continuous pixel values into discrete bins. While it can mitigate high-frequency noise, it does not guarantee protection against adaptive attacks (e.g. BPDA)."
            })
        if self.current_phase in (EvaluationPhase.COMPLETED, EvaluationPhase.BUDGET_EXHAUSTED):
            self.discoveries.append({
                "id": "DISC-05",
                "title": "Investigation Protocol Debrief Compiled",
                "unlocked": True,
                "badge": "MISSION DEBRIEF",
                "summary": f"Investigation completed with {len(self.completed_actions)}/{self.max_budget} budget operations consumed.",
                "intel": "Audit trail is fully sealed and persisted. Provenance, trust checks, and decision rationales are available for compliance and academic evaluation."
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
        }

        filepath = os.path.join(self.results_dir, f"{self.run_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2)

    def get_run_summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "current_phase": self.current_phase.value,
            "steps_completed": len(self.completed_actions),
            "remaining_budget": self.remaining_budget,
            "decisions_count": len(self.decision_history),
            "final_assessment": self.final_assessment.to_dict() if self.final_assessment else None,
        }

