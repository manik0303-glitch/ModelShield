"""
ModelShield Adaptive Red-Team Orchestrator Package
"""
from orchestrator.copilot import InvestigationCopilot
from orchestrator.engine import RedTeamOrchestrator
from orchestrator.policy import AdaptiveDecisionPolicy
from orchestrator.preflight import run_preflight_checks
from orchestrator.ui import render_red_team_mission_lab
from orchestrator.types import (
    DecisionAction,
    DecisionRecord,
    EvaluationPhase,
    FinalAssessment,
    PreflightResult,
    StepEvaluationResult,
)

__all__ = [
    "RedTeamOrchestrator",
    "AdaptiveDecisionPolicy",
    "InvestigationCopilot",
    "run_preflight_checks",
    "render_red_team_mission_lab",
    "EvaluationPhase",
    "DecisionAction",
    "DecisionRecord",
    "PreflightResult",
    "StepEvaluationResult",
    "FinalAssessment",
]

