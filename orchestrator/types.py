"""
ModelShield Adaptive Red-Team Orchestrator - Type Definitions
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvaluationPhase(str, Enum):
    INITIALIZED = "INITIALIZED"
    PREFLIGHT = "PREFLIGHT"
    BASELINE = "BASELINE"
    ATTACK_EVALUATION = "ATTACK_EVALUATION"
    DEFENSE_VERIFICATION = "DEFENSE_VERIFICATION"
    ANOMALY_VERIFICATION = "ANOMALY_VERIFICATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class DecisionAction(str, Enum):
    RUN_PREFLIGHT = "RUN_PREFLIGHT"
    RUN_BASELINE = "RUN_BASELINE"
    RUN_FGSM = "RUN_FGSM"
    ESCALATE_PGD = "ESCALATE_PGD"
    TEST_DEFENSE = "TEST_DEFENSE"
    REPEAT_VERIFICATION = "REPEAT_VERIFICATION"
    STOP_BUDGET_EXHAUSTED = "STOP_BUDGET_EXHAUSTED"
    STOP_SUFFICIENT_EVIDENCE = "STOP_SUFFICIENT_EVIDENCE"
    HALT_INTEGRITY_FAILURE = "HALT_INTEGRITY_FAILURE"


@dataclass
class PreflightResult:
    passed: bool
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


@dataclass
class DecisionRecord:
    step_number: int
    action: DecisionAction
    reason: str
    supporting_evidence: Dict[str, Any]
    expected_information: str
    budget_cost: int
    remaining_budget: int
    timestamp: str
    condition_evaluated: str = ""
    threshold_used: str = ""
    measured_value: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "action": self.action.value if isinstance(self.action, DecisionAction) else str(self.action),
            "reason": self.reason,
            "supporting_evidence": self.supporting_evidence,
            "expected_information": self.expected_information,
            "budget_cost": self.budget_cost,
            "remaining_budget": self.remaining_budget,
            "timestamp": self.timestamp,
            "condition_evaluated": self.condition_evaluated,
            "threshold_used": self.threshold_used,
            "measured_value": self.measured_value,
        }


@dataclass
class StepEvaluationResult:
    step_number: int
    action_type: str
    phase: EvaluationPhase
    metrics: Dict[str, Any] = field(default_factory=dict)
    trust_checks: Dict[str, Any] = field(default_factory=dict)
    trust_flag: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    status: str = "SUCCESS"  # SUCCESS, WARNING, FAIL
    error_message: Optional[str] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "action_type": self.action_type,
            "phase": self.phase.value if isinstance(self.phase, EvaluationPhase) else str(self.phase),
            "metrics": self.metrics,
            "trust_checks": self.trust_checks,
            "trust_flag": self.trust_flag,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "status": self.status,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


@dataclass
class FinalAssessment:
    run_id: str
    model_name: str
    overall_status: str  # SECURE_UNDER_SCOPE, MODERATE_VULNERABILITY, CRITICAL_VULNERABILITY, INTEGRITY_FAILED, INCONCLUSIVE
    overall_summary: str
    clean_accuracy: Optional[float] = None
    worst_adversarial_accuracy: Optional[float] = None
    max_attack_success_rate: Optional[float] = None
    tested_attacks: List[str] = field(default_factory=list)
    tested_defenses: List[str] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    trust_warnings: List[str] = field(default_factory=list)
    untested_vectors: List[str] = field(default_factory=list)
    scientific_limitations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    budget_allocated: int = 0
    budget_consumed: int = 0
    execution_duration_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "overall_status": self.overall_status,
            "overall_summary": self.overall_summary,
            "clean_accuracy": self.clean_accuracy,
            "worst_adversarial_accuracy": self.worst_adversarial_accuracy,
            "max_attack_success_rate": self.max_attack_success_rate,
            "tested_attacks": self.tested_attacks,
            "tested_defenses": self.tested_defenses,
            "key_findings": self.key_findings,
            "trust_warnings": self.trust_warnings,
            "untested_vectors": self.untested_vectors,
            "scientific_limitations": self.scientific_limitations,
            "recommendations": self.recommendations,
            "budget_allocated": self.budget_allocated,
            "budget_consumed": self.budget_consumed,
            "execution_duration_seconds": round(self.execution_duration_seconds, 3),
            "timestamp": self.timestamp,
        }

