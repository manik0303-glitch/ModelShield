"""
Model Security Regression Gate for ModelShield (Feature 15).

Compares evaluation snapshots between a Baseline model and a Candidate model,
verifies condition compatibility, detects regressions across clean and adversarial metrics,
evaluates trust checks and gradient-masking diagnostics, and issues an authoritative
PASS, WARN, or FAIL verdict with structured explanations.

Metric Convention:
All accuracy and Attack Success Rate (ASR) metrics are expressed as percentages
in the range [0.0, 100.0]. Thresholds are defined in percentage points.
Threshold allowances are inclusive: an accuracy drop or ASR surge strictly greater
than the configured threshold breaches the limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime, timezone


class GateVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


class MetricStatus(str, Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


SUPPORTED_METRICS = {
    "clean_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "fgsm_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "pgd_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "fgsm_attack_success_rate": MetricDirection.LOWER_IS_BETTER,
    "pgd_attack_success_rate": MetricDirection.LOWER_IS_BETTER,
    "worst_case_attack_success_rate": MetricDirection.LOWER_IS_BETTER,
}

ADVERSARIAL_METRIC_KEYS = {
    "fgsm_accuracy",
    "pgd_accuracy",
    "fgsm_attack_success_rate",
    "pgd_attack_success_rate",
    "worst_case_attack_success_rate",
}

PGD_METRIC_KEYS = {
    "pgd_accuracy",
    "pgd_attack_success_rate",
}


@dataclass
class EvaluationMetadata:
    """Metadata describing the conditions under which an evaluation snapshot was generated."""
    model_name: str = "UnknownModel"
    dataset_name: str = "CIFAR-10"
    num_samples: int = 0
    epsilon: float = 8.0 / 255.0
    alpha: Optional[float] = None
    pgd_steps: Optional[int] = None
    random_start: Optional[bool] = None
    timestamp: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> EvaluationMetadata:
        if not data or not isinstance(data, dict):
            return cls(model_name="UnknownModel", dataset_name="", num_samples=0, epsilon=0.0)

        eps_raw = data.get("epsilon")
        eps = float(eps_raw) if eps_raw is not None and not isinstance(eps_raw, bool) else 0.0

        ns_raw = data.get("num_samples")
        ns = int(ns_raw) if ns_raw is not None and not isinstance(ns_raw, bool) else 0

        alpha_raw = data.get("alpha")
        alpha = float(alpha_raw) if alpha_raw is not None and not isinstance(alpha_raw, bool) else None

        steps_raw = data.get("pgd_steps")
        steps = int(steps_raw) if steps_raw is not None and not isinstance(steps_raw, bool) else None

        return cls(
            model_name=str(data.get("model_name", "UnknownModel")),
            dataset_name=str(data.get("dataset_name", "")),
            num_samples=ns,
            epsilon=eps,
            alpha=alpha,
            pgd_steps=steps,
            random_start=bool(data["random_start"]) if data.get("random_start") is not None else None,
            timestamp=data.get("timestamp"),
            extra=data.get("extra", {}) if isinstance(data.get("extra"), dict) else {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSnapshot:
    """Standardized snapshot containing evaluation metrics, metadata, and optional diagnostic checks."""
    metadata: EvaluationMetadata
    metrics: Dict[str, float]
    trust_flag: Optional[Dict[str, Any]] = None
    gradient_masking: Optional[Dict[str, Any]] = None
    raw_checks: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationSnapshot:
        if not isinstance(data, dict):
            raise ValueError(f"Snapshot data must be a dictionary, got {type(data).__name__}")
        metadata_dict = data.get("metadata")
        metadata = EvaluationMetadata.from_dict(metadata_dict) if isinstance(metadata_dict, dict) else EvaluationMetadata(dataset_name="", num_samples=0, epsilon=0.0)
        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        return cls(
            metadata=metadata,
            metrics=metrics,
            trust_flag=data.get("trust_flag") if isinstance(data.get("trust_flag"), dict) else None,
            gradient_masking=data.get("gradient_masking") if isinstance(data.get("gradient_masking"), dict) else None,
            raw_checks=data.get("raw_checks") if isinstance(data.get("raw_checks"), dict) else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict() if self.metadata else {},
            "metrics": dict(self.metrics),
            "trust_flag": self.trust_flag,
            "gradient_masking": self.gradient_masking,
            "raw_checks": self.raw_checks,
        }


@dataclass
class RegressionThresholds:
    """
    Configurable tolerance boundaries for performance shifts.
    All drops and increases are expressed in percentage points ([0.0, 100.0]).
    Threshold enforcement:
      - Change <= warn_threshold: OK
      - warn_threshold < Change <= max_threshold: WARN
      - Change > max_threshold: FAIL
    """
    # Clean accuracy tolerance
    max_clean_accuracy_drop: float = 2.0
    warn_clean_accuracy_drop: float = 0.5

    # Robust accuracy tolerances
    max_pgd_accuracy_drop: float = 3.0
    warn_pgd_accuracy_drop: float = 1.0

    max_fgsm_accuracy_drop: float = 4.0
    warn_fgsm_accuracy_drop: float = 1.5

    # Attack success rate tolerances (surges)
    max_pgd_asr_increase: float = 3.0
    warn_pgd_asr_increase: float = 1.0

    max_fgsm_asr_increase: float = 4.0
    warn_fgsm_asr_increase: float = 1.5

    max_worst_case_asr_increase: float = 3.0
    warn_worst_case_asr_increase: float = 1.0

    # Policy switches
    enforce_condition_equality: bool = True
    require_trust_checks_pass: bool = True
    reject_gradient_masking: bool = True
    warn_on_inconclusive_masking: bool = True
    epsilon_tolerance: float = 1e-5


@dataclass
class MetricDelta:
    """Represents a comparative difference for a single evaluation metric."""
    metric_name: str
    baseline_value: float
    candidate_value: float
    delta: float  # candidate_value - baseline_value
    direction: MetricDirection
    status: MetricStatus
    reason: str
    warn_threshold: Optional[float] = None
    fail_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_value": round(self.baseline_value, 4),
            "candidate_value": round(self.candidate_value, 4),
            "delta": round(self.delta, 4),
            "direction": self.direction.value,
            "status": self.status.value,
            "reason": self.reason,
            "warn_threshold": self.warn_threshold,
            "fail_threshold": self.fail_threshold,
        }


@dataclass
class RegressionGateResult:
    """Final output of the Model Security Regression Gate."""
    verdict: GateVerdict
    summary: str
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    metric_deltas: Dict[str, MetricDelta] = field(default_factory=dict)
    condition_compatibility: Dict[str, Any] = field(default_factory=dict)
    baseline_metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_metadata: Dict[str, Any] = field(default_factory=dict)
    trust_flag_assessment: Optional[Dict[str, Any]] = None
    gradient_masking_assessment: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "summary": self.summary,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "passed_checks": list(self.passed_checks),
            "metric_deltas": {k: v.to_dict() for k, v in self.metric_deltas.items()},
            "condition_compatibility": self.condition_compatibility,
            "baseline_metadata": self.baseline_metadata,
            "candidate_metadata": self.candidate_metadata,
            "trust_flag_assessment": self.trust_flag_assessment,
            "gradient_masking_assessment": self.gradient_masking_assessment,
            "timestamp": self.timestamp,
        }


def validate_snapshot_integrity(
    snapshot: Any, role_name: str
) -> Tuple[bool, List[str], Optional[EvaluationSnapshot]]:
    """
    Validates that a snapshot is structurally sound, contains clean_accuracy and
    at least one adversarial metric, has valid non-empty metadata, and has only
    finite numeric metrics within [0.0, 100.0].
    """
    errors: List[str] = []

    if snapshot is None:
        return False, [f"{role_name} snapshot is None."], None

    raw_dict: Optional[Dict[str, Any]] = None
    if isinstance(snapshot, dict):
        raw_dict = snapshot
        try:
            snapshot_obj = EvaluationSnapshot.from_dict(snapshot)
        except Exception as exc:
            return False, [f"Failed to parse {role_name} snapshot: {exc}"], None
    elif isinstance(snapshot, EvaluationSnapshot):
        snapshot_obj = snapshot
    else:
        return False, [f"{role_name} snapshot must be an EvaluationSnapshot or dict, got {type(snapshot).__name__}."], None

    # Validate metadata presence
    if raw_dict is not None and ("metadata" not in raw_dict or not isinstance(raw_dict["metadata"], dict) or len(raw_dict["metadata"]) == 0):
        errors.append(f"{role_name} snapshot is missing mandatory 'metadata' dictionary.")

    meta = snapshot_obj.metadata
    if meta is None:
        errors.append(f"{role_name} snapshot is missing metadata.")
    else:
        # Require non-empty dataset_name
        if not meta.dataset_name or not isinstance(meta.dataset_name, str) or not meta.dataset_name.strip():
            errors.append(f"{role_name} metadata must include a valid non-empty 'dataset_name'.")

        # Require num_samples > 0
        if not isinstance(meta.num_samples, int) or isinstance(meta.num_samples, bool) or meta.num_samples <= 0:
            errors.append(f"{role_name} metadata 'num_samples' must be an integer greater than zero, got {meta.num_samples!r}.")

        # Require finite epsilon > 0.0
        if not isinstance(meta.epsilon, (int, float)) or isinstance(meta.epsilon, bool) or math.isnan(meta.epsilon) or math.isinf(meta.epsilon) or meta.epsilon <= 0.0:
            errors.append(f"{role_name} metadata 'epsilon' must be a finite number greater than zero, got {meta.epsilon!r}.")

        # Optional PGD parameters validation
        if meta.pgd_steps is not None:
            if not isinstance(meta.pgd_steps, int) or isinstance(meta.pgd_steps, bool) or meta.pgd_steps <= 0:
                errors.append(f"{role_name} metadata 'pgd_steps' must be a positive integer, got {meta.pgd_steps!r}.")

        if meta.alpha is not None:
            if not isinstance(meta.alpha, (int, float)) or isinstance(meta.alpha, bool) or math.isnan(meta.alpha) or math.isinf(meta.alpha) or meta.alpha <= 0.0:
                errors.append(f"{role_name} metadata 'alpha' must be a finite positive number, got {meta.alpha!r}.")

    # Validate metrics presence
    metrics = snapshot_obj.metrics
    if not isinstance(metrics, dict) or len(metrics) == 0:
        errors.append(f"{role_name} snapshot contains no metrics dictionary or it is empty.")
        return False, errors, snapshot_obj

    # 1. Require clean_accuracy
    if "clean_accuracy" not in metrics:
        errors.append(f"{role_name} snapshot is missing mandatory 'clean_accuracy' metric.")

    # 2. Require at least one adversarial metric
    has_adv = any(k in metrics for k in ADVERSARIAL_METRIC_KEYS)
    if not has_adv:
        errors.append(
            f"{role_name} snapshot must contain at least one adversarial metric among: "
            f"{sorted(ADVERSARIAL_METRIC_KEYS)}."
        )

    # 3. Validate every metric value
    for key, value in metrics.items():
        if value is None:
            errors.append(f"{role_name} metric '{key}' has value None.")
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(
                f"{role_name} metric '{key}' is non-numeric: {value!r} ({type(value).__name__})."
            )
            continue
        if math.isnan(value) or math.isinf(value):
            errors.append(f"{role_name} metric '{key}' has invalid non-finite value: {value}.")
            continue
        # Accuracy & ASR must be within percentage range [0.0, 100.0]
        if value < 0.0 or value > 100.0:
            errors.append(
                f"{role_name} metric '{key}' is out of valid percentage range [0.0, 100.0]: {value}."
            )

    if errors:
        return False, errors, snapshot_obj

    return True, [], snapshot_obj


def validate_evaluation_conditions(
    base_meta: EvaluationMetadata,
    cand_meta: EvaluationMetadata,
    thresholds: RegressionThresholds,
    base_metrics: Optional[Dict[str, float]] = None,
    cand_metrics: Optional[Dict[str, float]] = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates that the baseline and candidate evaluation runs were conducted under
    equivalent, comparable conditions.
    When both snapshots include PGD-related metrics, PGD parameters must be present
    and strictly compatible in both snapshots.
    """
    mismatches: List[str] = []
    audit: Dict[str, Any] = {
        "dataset_name": {
            "baseline": base_meta.dataset_name,
            "candidate": cand_meta.dataset_name,
            "matched": base_meta.dataset_name.lower().strip() == cand_meta.dataset_name.lower().strip(),
        },
        "num_samples": {
            "baseline": base_meta.num_samples,
            "candidate": cand_meta.num_samples,
            "matched": base_meta.num_samples == cand_meta.num_samples,
        },
        "epsilon": {
            "baseline": base_meta.epsilon,
            "candidate": cand_meta.epsilon,
            "matched": abs(base_meta.epsilon - cand_meta.epsilon) <= thresholds.epsilon_tolerance,
        },
    }

    if not audit["dataset_name"]["matched"]:
        mismatches.append(
            f"Dataset mismatch: Baseline used '{base_meta.dataset_name}', "
            f"Candidate used '{cand_meta.dataset_name}'."
        )

    if not audit["num_samples"]["matched"]:
        mismatches.append(
            f"Sample count mismatch: Baseline evaluated on {base_meta.num_samples} samples, "
            f"Candidate evaluated on {cand_meta.num_samples} samples."
        )

    if not audit["epsilon"]["matched"]:
        mismatches.append(
            f"Perturbation budget (epsilon) mismatch: Baseline={base_meta.epsilon:.6f}, "
            f"Candidate={cand_meta.epsilon:.6f} (tolerance={thresholds.epsilon_tolerance})."
        )

    # Check PGD condition requirements when PGD metrics are present in both
    b_has_pgd = False
    c_has_pgd = False
    if base_metrics is not None and cand_metrics is not None:
        b_has_pgd = any(k in base_metrics for k in PGD_METRIC_KEYS)
        c_has_pgd = any(k in cand_metrics for k in PGD_METRIC_KEYS)
    else:
        b_has_pgd = base_meta.pgd_steps is not None
        c_has_pgd = cand_meta.pgd_steps is not None

    if b_has_pgd and c_has_pgd:
        if base_meta.pgd_steps is None:
            mismatches.append("Baseline metadata is missing 'pgd_steps' required to evaluate PGD metrics.")
        if cand_meta.pgd_steps is None:
            mismatches.append("Candidate metadata is missing 'pgd_steps' required to evaluate PGD metrics.")

        if base_meta.pgd_steps is not None and cand_meta.pgd_steps is not None:
            matched_steps = base_meta.pgd_steps == cand_meta.pgd_steps
            audit["pgd_steps"] = {
                "baseline": base_meta.pgd_steps,
                "candidate": cand_meta.pgd_steps,
                "matched": matched_steps,
            }
            if not matched_steps:
                mismatches.append(
                    f"PGD steps mismatch: Baseline={base_meta.pgd_steps}, "
                    f"Candidate={cand_meta.pgd_steps}."
                )

        if base_meta.alpha is not None or cand_meta.alpha is not None:
            if base_meta.alpha is None or cand_meta.alpha is None:
                mismatches.append("PGD step size (alpha) presence mismatch: specified in one snapshot but missing in the other.")
            else:
                matched_alpha = abs(base_meta.alpha - cand_meta.alpha) <= thresholds.epsilon_tolerance
                audit["alpha"] = {
                    "baseline": base_meta.alpha,
                    "candidate": cand_meta.alpha,
                    "matched": matched_alpha,
                }
                if not matched_alpha:
                    mismatches.append(
                        f"PGD step size (alpha) mismatch: Baseline={base_meta.alpha:.6f}, "
                        f"Candidate={cand_meta.alpha:.6f}."
                    )
    elif base_meta.pgd_steps is not None and cand_meta.pgd_steps is not None:
        # Check PGD steps if both explicitly specify them, even without explicit PGD metrics
        matched_steps = base_meta.pgd_steps == cand_meta.pgd_steps
        audit["pgd_steps"] = {
            "baseline": base_meta.pgd_steps,
            "candidate": cand_meta.pgd_steps,
            "matched": matched_steps,
        }
        if not matched_steps:
            mismatches.append(
                f"PGD steps mismatch: Baseline={base_meta.pgd_steps}, "
                f"Candidate={cand_meta.pgd_steps}."
            )

    is_compatible = len(mismatches) == 0
    return is_compatible, mismatches, audit


def compute_metric_deltas(
    baseline_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
    thresholds: RegressionThresholds,
) -> Tuple[Dict[str, MetricDelta], List[str], List[str], List[str]]:
    """
    Computes comparative deltas for all mutually available metrics.
    Never compares metrics that are missing from either snapshot.
    """
    deltas: Dict[str, MetricDelta] = {}
    failures: List[str] = []
    warnings: List[str] = []
    passed: List[str] = []

    threshold_map = {
        "clean_accuracy": (thresholds.warn_clean_accuracy_drop, thresholds.max_clean_accuracy_drop),
        "pgd_accuracy": (thresholds.warn_pgd_accuracy_drop, thresholds.max_pgd_accuracy_drop),
        "fgsm_accuracy": (thresholds.warn_fgsm_accuracy_drop, thresholds.max_fgsm_accuracy_drop),
        "pgd_attack_success_rate": (thresholds.warn_pgd_asr_increase, thresholds.max_pgd_asr_increase),
        "fgsm_attack_success_rate": (thresholds.warn_fgsm_asr_increase, thresholds.max_fgsm_asr_increase),
        "worst_case_attack_success_rate": (
            thresholds.warn_worst_case_asr_increase,
            thresholds.max_worst_case_asr_increase,
        ),
    }

    mutual_keys = [k for k in SUPPORTED_METRICS if k in baseline_metrics and k in candidate_metrics]

    for key in mutual_keys:
        b_val = float(baseline_metrics[key])
        c_val = float(candidate_metrics[key])
        delta = c_val - b_val
        direction = SUPPORTED_METRICS[key]
        warn_th, max_th = threshold_map.get(key, (1.0, 3.0))

        if direction == MetricDirection.HIGHER_IS_BETTER:
            drop = -delta
            if drop > max_th:
                status = MetricStatus.FAIL
                reason = (
                    f"{key} dropped by {drop:.2f}% (from {b_val:.2f}% to {c_val:.2f}%), "
                    f"exceeding max allowable drop of {max_th:.2f}%."
                )
                failures.append(reason)
            elif drop > warn_th:
                status = MetricStatus.WARN
                reason = (
                    f"{key} dropped by {drop:.2f}% (from {b_val:.2f}% to {c_val:.2f}%), "
                    f"exceeding warning threshold of {warn_th:.2f}%."
                )
                warnings.append(reason)
            else:
                status = MetricStatus.OK
                change_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
                reason = f"{key} shifted by {change_str} (within allowable limits)."
                passed.append(f"{key}: {b_val:.2f}% -> {c_val:.2f}% (delta {change_str})")

        else:
            increase = delta
            if increase > max_th:
                status = MetricStatus.FAIL
                reason = (
                    f"{key} increased by +{increase:.2f}% (from {b_val:.2f}% to {c_val:.2f}%), "
                    f"exceeding max allowable surge of +{max_th:.2f}%."
                )
                failures.append(reason)
            elif increase > warn_th:
                status = MetricStatus.WARN
                reason = (
                    f"{key} increased by +{increase:.2f}% (from {b_val:.2f}% to {c_val:.2f}%), "
                    f"exceeding warning threshold of +{warn_th:.2f}%."
                )
                warnings.append(reason)
            else:
                status = MetricStatus.OK
                change_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
                reason = f"{key} shifted by {change_str} (within allowable limits)."
                passed.append(f"{key}: {b_val:.2f}% -> {c_val:.2f}% (delta {change_str})")

        deltas[key] = MetricDelta(
            metric_name=key,
            baseline_value=b_val,
            candidate_value=c_val,
            delta=delta,
            direction=direction,
            status=status,
            reason=reason,
            warn_threshold=warn_th,
            fail_threshold=max_th,
        )

    return deltas, failures, warnings, passed


def evaluate_trust_checks_result(
    trust_flag: Optional[Dict[str, Any]],
    require_pass: bool,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Evaluates candidate trust flag schema:
      status: "PASS" | "WARNING" | "FAIL"
      summary: str
      passed: int, failed: int, total: int
    Missing trust flag is NOT reported as passed.
    """
    if trust_flag is None:
        return None, None, None

    status = str(trust_flag.get("status", "")).upper()
    summary = trust_flag.get("summary", "No trust flag summary provided.")
    passed_cnt = trust_flag.get("passed", 0)
    failed_cnt = trust_flag.get("failed", 0)
    total_cnt = trust_flag.get("total", 0)

    assessment = {
        "status": status,
        "summary": summary,
        "passed": passed_cnt,
        "failed": failed_cnt,
        "total": total_cnt,
    }

    if status == "FAIL":
        fail_msg = f"Candidate failed ModelShield trust checks ({failed_cnt}/{total_cnt} failed): {summary}"
        return fail_msg, None, assessment
    elif status == "WARNING":
        warn_msg = f"Candidate triggered ModelShield trust checks warning: {summary}"
        return None, warn_msg, assessment
    elif status == "PASS":
        return None, None, assessment
    else:
        warn_msg = f"Candidate has unrecognized trust check status '{status}': {summary}"
        return None, warn_msg, assessment


def evaluate_gradient_masking_result(
    gradient_masking: Optional[Dict[str, Any]],
    reject_masking: bool,
    warn_on_inconclusive: bool,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Evaluates candidate gradient-masking diagnostic results schema:
      status: "NO_STRONG_SIGNAL" | "WARNING" | "INCONCLUSIVE" | "ERROR"
      summary: str
      warning_reasons: list
    Missing diagnostic is NOT reported as passed.
    """
    if gradient_masking is None:
        return None, None, None

    status = str(gradient_masking.get("status", "")).upper()
    summary = gradient_masking.get("summary", "")
    reasons = gradient_masking.get("warning_reasons", [])

    assessment = {
        "status": status,
        "summary": summary,
        "warning_reasons": reasons,
    }

    if status == "WARNING":
        detail = "; ".join(reasons) if reasons else summary
        msg = f"Candidate exhibits suspicious gradient masking signals: {detail}"
        if reject_masking:
            return msg, None, assessment
        else:
            return None, msg, assessment

    elif status == "INCONCLUSIVE":
        if warn_on_inconclusive:
            msg = f"Candidate gradient-masking diagnostics were inconclusive: {summary}"
            return None, msg, assessment
        return None, None, assessment

    elif status == "ERROR":
        msg = f"Candidate gradient-masking diagnostics encountered an error: {summary}"
        return msg, None, assessment

    elif status == "NO_STRONG_SIGNAL":
        return None, None, assessment

    else:
        msg = f"Candidate has unknown gradient-masking status '{status}': {summary}"
        return None, msg, assessment


def evaluate_regression_gate(
    baseline_snapshot: Any,
    candidate_snapshot: Any,
    thresholds: Optional[RegressionThresholds] = None,
) -> RegressionGateResult:
    """
    Evaluates whether the candidate model passes the security regression gate compared to baseline.
    Strictly follows priority: FAIL > WARN > PASS.
    """
    if thresholds is None:
        thresholds = RegressionThresholds()

    failures: List[str] = []
    warnings: List[str] = []
    passed_checks: List[str] = []

    # 1. Snapshot integrity validation
    b_ok, b_errs, base_snap = validate_snapshot_integrity(baseline_snapshot, "Baseline")
    c_ok, c_errs, cand_snap = validate_snapshot_integrity(candidate_snapshot, "Candidate")

    if not b_ok:
        failures.extend([f"Baseline Snapshot Integrity Failure: {e}" for e in b_errs])
    if not c_ok:
        failures.extend([f"Candidate Snapshot Integrity Failure: {e}" for e in c_errs])

    if not b_ok or not c_ok:
        summary = f"Regression gate rejected evaluation: snapshot integrity validation failed ({len(failures)} issue(s))."
        return RegressionGateResult(
            verdict=GateVerdict.FAIL,
            summary=summary,
            failures=failures,
            warnings=warnings,
            passed_checks=passed_checks,
            baseline_metadata=base_snap.metadata.to_dict() if base_snap and base_snap.metadata else {},
            candidate_metadata=cand_snap.metadata.to_dict() if cand_snap and cand_snap.metadata else {},
        )

    assert base_snap is not None and cand_snap is not None

    # 2. Evaluation condition compatibility
    is_compatible, condition_mismatches, audit = validate_evaluation_conditions(
        base_snap.metadata,
        cand_snap.metadata,
        thresholds,
        base_metrics=base_snap.metrics,
        cand_metrics=cand_snap.metrics,
    )

    if not is_compatible:
        if thresholds.enforce_condition_equality:
            failures.extend([f"Condition Compatibility Failure: {m}" for m in condition_mismatches])
        else:
            warnings.extend([f"Condition Compatibility Warning: {m}" for m in condition_mismatches])
    else:
        passed_checks.append(
            f"Evaluation conditions matched (Dataset: {base_snap.metadata.dataset_name}, "
            f"Samples: {base_snap.metadata.num_samples}, Epsilon: {base_snap.metadata.epsilon:.6f})"
        )

    # 3. Candidate trust checks validation
    tc_fail, tc_warn, tc_assessment = evaluate_trust_checks_result(
        cand_snap.trust_flag, thresholds.require_trust_checks_pass
    )
    if tc_fail:
        failures.append(tc_fail)
    if tc_warn:
        warnings.append(tc_warn)
    if cand_snap.trust_flag and not tc_fail and not tc_warn:
        passed_checks.append("Candidate passed ModelShield trust checks.")

    # 4. Candidate gradient masking validation
    gm_fail, gm_warn, gm_assessment = evaluate_gradient_masking_result(
        cand_snap.gradient_masking,
        thresholds.reject_gradient_masking,
        thresholds.warn_on_inconclusive_masking,
    )
    if gm_fail:
        failures.append(gm_fail)
    if gm_warn:
        warnings.append(gm_warn)
    if cand_snap.gradient_masking and not gm_fail and not gm_warn:
        passed_checks.append("Candidate passed gradient-masking diagnostics (NO_STRONG_SIGNAL).")

    # 5. Compute metric deltas
    deltas, m_failures, m_warnings, m_passed = compute_metric_deltas(
        base_snap.metrics, cand_snap.metrics, thresholds
    )
    failures.extend(m_failures)
    warnings.extend(m_warnings)
    passed_checks.extend(m_passed)

    # Verify at least one mutual metric was evaluated
    if len(deltas) == 0:
        failures.append("No mutual metrics could be compared between baseline and candidate snapshots.")

    # 6. Consolidate verdict: FAIL > WARN > PASS
    if len(failures) > 0:
        verdict = GateVerdict.FAIL
        summary = f"GATE FAILED: {len(failures)} critical regression or condition issue(s) detected."
    elif len(warnings) > 0:
        verdict = GateVerdict.WARN
        summary = f"GATE WARNING: {len(warnings)} acceptable warning-level degradation(s) observed."
    else:
        verdict = GateVerdict.PASS
        summary = f"GATE PASSED: Candidate meets or exceeds all security and accuracy thresholds ({len(passed_checks)} check(s) passed)."

    return RegressionGateResult(
        verdict=verdict,
        summary=summary,
        failures=failures,
        warnings=warnings,
        passed_checks=passed_checks,
        metric_deltas=deltas,
        condition_compatibility=audit,
        baseline_metadata=base_snap.metadata.to_dict(),
        candidate_metadata=cand_snap.metadata.to_dict(),
        trust_flag_assessment=tc_assessment,
        gradient_masking_assessment=gm_assessment,
    )


def export_gate_report_json(result: RegressionGateResult, indent: int = 2) -> str:
    """Exports regression gate result to a structured, pretty-printed JSON string."""
    return json.dumps(result.to_dict(), indent=indent)


def export_gate_report_markdown(result: RegressionGateResult) -> str:
    """
    Exports regression gate result to a clean, formatted Markdown document
    suitable for PR comments and CI/CD audit logs.
    """
    status_emoji = {
        GateVerdict.PASS: "PASS ✅",
        GateVerdict.WARN: "WARN ⚠️",
        GateVerdict.FAIL: "FAIL ❌",
    }
    metric_status_icon = {
        "OK": "✅ PASS",
        "WARN": "⚠️ WARN",
        "FAIL": "❌ FAIL",
    }

    base_meta = result.baseline_metadata
    cand_meta = result.candidate_metadata

    lines = [
        f"# 🛡️ ModelShield Security Regression Gate: {status_emoji[result.verdict]}",
        "",
        f"**Summary:** {result.summary}",
        f"**Evaluated At (UTC):** {result.timestamp}",
        "",
        "### Models & Conditions",
        f"- **Baseline:** `{base_meta.get('model_name', 'Unknown')}` (Dataset: {base_meta.get('dataset_name')}, Samples: {base_meta.get('num_samples')}, ε={base_meta.get('epsilon', 0):.6f})",
        f"- **Candidate:** `{cand_meta.get('model_name', 'Unknown')}` (Dataset: {cand_meta.get('dataset_name')}, Samples: {cand_meta.get('num_samples')}, ε={cand_meta.get('epsilon', 0):.6f})",
        "",
        "### Metric Comparison Table",
        "| Metric | Baseline | Candidate | Delta | Status | Threshold |",
        "|---|---|---|---|---|---|",
    ]

    for name, delta in result.metric_deltas.items():
        icon = metric_status_icon.get(delta.status.value, delta.status.value)
        sign = "+" if delta.delta > 0 else ""
        delta_str = f"{sign}{delta.delta:.2f}%"
        th_val = delta.fail_threshold if delta.fail_threshold is not None else "N/A"
        lines.append(
            f"| `{name}` | {delta.baseline_value:.2f}% | {delta.candidate_value:.2f}% | {delta_str} | {icon} | Max {th_val}% |"
        )

    if result.failures:
        lines.extend([
            "",
            "### ❌ Critical Violations (FAIL)",
        ])
        for fail in result.failures:
            lines.append(f"- {fail}")

    if result.warnings:
        lines.extend([
            "",
            "### ⚠️ Warning-Level Degradations (WARN)",
        ])
        for warn in result.warnings:
            lines.append(f"- {warn}")

    if result.passed_checks:
        lines.extend([
            "",
            "### ✅ Passed Checks",
        ])
        for check in result.passed_checks:
            lines.append(f"- {check}")

    lines.append("")
    return "\n".join(lines)
