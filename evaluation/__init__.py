"""Evaluation package for ModelShield."""

from evaluation.regression_gate import (
    GateVerdict,
    MetricStatus,
    MetricDirection,
    EvaluationMetadata,
    EvaluationSnapshot,
    RegressionThresholds,
    MetricDelta,
    RegressionGateResult,
    evaluate_regression_gate,
    export_gate_report_json,
    export_gate_report_markdown,
)

__all__ = [
    "GateVerdict",
    "MetricStatus",
    "MetricDirection",
    "EvaluationMetadata",
    "EvaluationSnapshot",
    "RegressionThresholds",
    "MetricDelta",
    "RegressionGateResult",
    "evaluate_regression_gate",
    "export_gate_report_json",
    "export_gate_report_markdown",
]

