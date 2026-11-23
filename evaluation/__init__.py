"""Evaluation package for ModelShield."""

from evaluation.regression_gate import (
    GateVerdict,
    MetricStatus,
    MetricDirection,
    TaskType,
    EvaluationMetadata,
    EvaluationSnapshot,
    RegressionThresholds,
    MetricDelta,
    RegressionGateResult,
    evaluate_regression_gate,
    export_gate_report_json,
    export_gate_report_markdown,
    validate_snapshot_integrity,
    CLASSIFICATION_METRICS,
    IMAGE_TAGGING_METRICS,
    SUPPORTED_METRICS,
)
from model.checkpoint_loader import (
    SupportedArchitecture,
    load_checkpoint_safely,
    load_baseline_model,
)
from evaluation.live_eval import (
    evaluate_model_live,
    is_snapshot_compatible_with_config,
    run_live_regression_pipeline,
)

__all__ = [
    "GateVerdict",
    "MetricStatus",
    "MetricDirection",
    "TaskType",
    "EvaluationMetadata",
    "EvaluationSnapshot",
    "RegressionThresholds",
    "MetricDelta",
    "RegressionGateResult",
    "evaluate_regression_gate",
    "export_gate_report_json",
    "export_gate_report_markdown",
    "validate_snapshot_integrity",
    "CLASSIFICATION_METRICS",
    "IMAGE_TAGGING_METRICS",
    "SUPPORTED_METRICS",
    "SupportedArchitecture",
    "load_checkpoint_safely",
    "load_baseline_model",
    "evaluate_model_live",
    "is_snapshot_compatible_with_config",
    "run_live_regression_pipeline",
]
