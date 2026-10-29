"""
ModelShield Adaptive Decision Policy
Deterministic, evidence-grounded decision component for selecting follow-up adversarial evaluations.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from orchestrator.types import DecisionAction, DecisionRecord, EvaluationPhase


class AdaptiveDecisionPolicy:
    """
    Deterministic decision policy for adversarial robustness testing.
    Evaluates observed metrics, integrity flags, completed test history, and budget.
    """

    def __init__(
        self,
        asr_escalation_threshold: float = 15.0,
        accuracy_drop_threshold: float = 10.0,
        defense_trigger_asr: float = 20.0,
    ):
        self.asr_escalation_threshold = asr_escalation_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.defense_trigger_asr = defense_trigger_asr

    def decide_next_action(
        self,
        step_number: int,
        current_phase: EvaluationPhase,
        preflight_passed: Optional[bool],
        preflight_errors: List[str],
        completed_actions: List[str],
        metrics_by_action: Dict[str, Dict[str, Any]],
        trust_flags_by_action: Dict[str, Dict[str, Any]],
        remaining_budget: int,
        anomalies_detected: List[str],
    ) -> DecisionRecord:
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Rule 1: Integrity preflight check
        if preflight_passed is False:
            return DecisionRecord(
                step_number=step_number,
                action=DecisionAction.HALT_INTEGRITY_FAILURE,
                reason=(
                    "Preflight integrity validation failed. Halting evaluation to prevent "
                    "invalid measurements from being misclassified as model vulnerabilities."
                ),
                supporting_evidence={
                    "preflight_errors": preflight_errors,
                    "preflight_passed": False,
                },
                expected_information="Resolve environmental, tensor shape, or parameter errors.",
                budget_cost=0,
                remaining_budget=remaining_budget,
                timestamp=timestamp,
                condition_evaluated="preflight_passed == False",
                threshold_used="preflight_passed is True",
                measured_value=f"{len(preflight_errors)} integrity errors detected",
            )

        if "RUN_PREFLIGHT" not in completed_actions and preflight_passed is None:
            return DecisionRecord(
                step_number=step_number,
                action=DecisionAction.RUN_PREFLIGHT,
                reason="Execute preflight checks to verify input dimensions, pixel range, and model execution.",
                supporting_evidence={"status": "Preflight pending"},
                expected_information="Confirm environment is ready for reliable evaluation.",
                budget_cost=1,
                remaining_budget=remaining_budget,
                timestamp=timestamp,
                condition_evaluated="'RUN_PREFLIGHT' not in completed_actions",
                threshold_used="Preflight required before testing",
                measured_value="Pending initialization",
            )

        # Rule 2: Budget enforcement check
        if remaining_budget <= 0:
            return DecisionRecord(
                step_number=step_number,
                action=DecisionAction.STOP_BUDGET_EXHAUSTED,
                reason=(
                    f"Testing halted because the configured test budget has been exhausted (all allotted evaluation "
                    f"steps consumed). The assessment is bounded to {len(completed_actions)} completed evaluations, "
                    f"and no conclusion about untested attack categories should be inferred."
                ),
                supporting_evidence={
                    "completed_actions": completed_actions,
                    "remaining_budget": remaining_budget,
                },
                expected_information="Further testing requires allocating additional test budget.",
                budget_cost=0,
                remaining_budget=0,
                timestamp=timestamp,
                condition_evaluated=f"remaining_budget ({remaining_budget}) <= 0",
                threshold_used="remaining_budget > 0",
                measured_value=f"0 remaining (consumed {len(completed_actions)})",
            )

        # Rule 3: Baseline clean evaluation
        if "RUN_BASELINE" not in completed_actions:
            return DecisionRecord(
                step_number=step_number,
                action=DecisionAction.RUN_BASELINE,
                reason=(
                    "Clean baseline accuracy must be established first to define reference performance "
                    "and ground-truth prediction alignment before applying adversarial perturbations."
                ),
                supporting_evidence={"baseline_status": "Not established"},
                expected_information="Clean model classification accuracy and prediction distributions.",
                budget_cost=1,
                remaining_budget=remaining_budget,
                timestamp=timestamp,
                condition_evaluated="'RUN_BASELINE' not in completed_actions",
                threshold_used="Clean reference benchmark required",
                measured_value="Pending baseline",
            )

        # Retrieve baseline metrics
        baseline_metrics = metrics_by_action.get("RUN_BASELINE", {})
        clean_acc = baseline_metrics.get("clean_accuracy", 0.0)

        # Rule 4: Clean model performance sanity check
        if clean_acc < 15.0 and "REPEAT_VERIFICATION" not in completed_actions:
            return DecisionRecord(
                step_number=step_number,
                action=DecisionAction.REPEAT_VERIFICATION,
                reason=(
                    f"Clean accuracy is unusually low ({clean_acc:.1f}% on CIFAR-10). Running a repeat verification "
                    f"sample to confirm whether this is genuine model degradation or a data preprocessing mismatch."
                ),
                supporting_evidence={"clean_accuracy": clean_acc},
                expected_information="Confirm whether clean accuracy is reproducible.",
                budget_cost=1,
                remaining_budget=remaining_budget,
                timestamp=timestamp,
                condition_evaluated=f"clean_acc ({clean_acc:.1f}%) < 15.0%",
                threshold_used="clean_acc >= 15.0%",
                measured_value=f"clean_acc = {clean_acc:.1f}%",
            )

        # Rule 5: Anomaly verification
        if anomalies_detected and "REPEAT_VERIFICATION" not in completed_actions:
            return DecisionRecord(
                step_number=step_number,
                action=DecisionAction.REPEAT_VERIFICATION,
                reason=(
                    f"Suspicious metric anomaly detected: {anomalies_detected[0]}. "
                    f"Verifying measurement integrity before drawing robustness conclusions."
                ),
                supporting_evidence={"anomalies": anomalies_detected},
                expected_information="Resolve anomaly through calibrated repeat test.",
                budget_cost=1,
                remaining_budget=remaining_budget,
                timestamp=timestamp,
                condition_evaluated="anomalies_detected != []",
                threshold_used="anomalies == 0",
                measured_value=str(anomalies_detected),
            )

        # Rule 6: Initial linear attack probe (FGSM)
        if "RUN_FGSM" not in completed_actions:
            return DecisionRecord(
                step_number=step_number,
                action=DecisionAction.RUN_FGSM,
                reason=(
                    "Evaluate single-step Fast Gradient Sign Method (FGSM). This measures first-order "
                    "gradient sensitivity under the configured Linf perturbation budget."
                ),
                supporting_evidence={"clean_accuracy": clean_acc},
                expected_information="Single-step adversarial accuracy, attack success rate, and perturbation norm.",
                budget_cost=1,
                remaining_budget=remaining_budget,
                timestamp=timestamp,
                condition_evaluated="'RUN_FGSM' not in completed_actions",
                threshold_used="First-order linear gradient probe required",
                measured_value=f"clean_acc = {clean_acc:.1f}%",
            )

        fgsm_metrics = metrics_by_action.get("RUN_FGSM", {})
        fgsm_acc = fgsm_metrics.get("adversarial_accuracy", clean_acc)
        fgsm_asr = fgsm_metrics.get("attack_success_rate", 0.0)
        fgsm_acc_drop = max(0.0, clean_acc - fgsm_acc)

        # Rule 7: Iterative attack escalation (PGD)
        if "ESCALATE_PGD" not in completed_actions:
            vulnerability_detected = (
                fgsm_asr >= self.asr_escalation_threshold
                or fgsm_acc_drop >= self.accuracy_drop_threshold
            )
            if vulnerability_detected:
                reason = (
                    f"PGD was selected because baseline FGSM revealed measurable vulnerability "
                    f"(clean accuracy: {clean_acc:.1f}%, FGSM accuracy: {fgsm_acc:.1f}%, ASR: {fgsm_asr:.1f}%). "
                    f"An iterative attack with multi-step projection is required to test whether non-linear "
                    f"gradient optimization further exploits the model's decision boundaries."
                )
                condition_eval = f"fgsm_asr ({fgsm_asr:.1f}%) >= {self.asr_escalation_threshold:.1f}% or drop ({fgsm_acc_drop:.1f}%) >= {self.accuracy_drop_threshold:.1f}%"
                meas_val = f"ASR: {fgsm_asr:.1f}%, Acc drop: {fgsm_acc_drop:.1f}%"
                thresh_used = f"ASR >= {self.asr_escalation_threshold:.1f}% or drop >= {self.accuracy_drop_threshold:.1f}% (Vulnerability Escalation)"
            elif remaining_budget >= 2:
                # If FGSM had minimal effect but budget permits, probe with multi-step PGD to rule out gradient masking
                reason = (
                    f"PGD was selected to rule out gradient masking or false security. FGSM had minimal effect "
                    f"(ASR: {fgsm_asr:.1f}% < {self.asr_escalation_threshold:.1f}%), so multi-step iterative optimization "
                    f"with random starts is necessary to explore the local loss landscape more thoroughly."
                )
                condition_eval = f"fgsm_asr ({fgsm_asr:.1f}%) < {self.asr_escalation_threshold:.1f}% and remaining_budget ({remaining_budget}) >= 2"
                meas_val = f"ASR: {fgsm_asr:.1f}% (Below threshold, false-security audit triggered)"
                thresh_used = "Gradient Masking Audit (ASR < threshold & budget >= 2)"
            else:
                # Insufficient budget and no vulnerability demonstrated: bypass PGD
                return DecisionRecord(
                    step_number=step_number,
                    action=DecisionAction.STOP_SUFFICIENT_EVIDENCE,
                    reason=(
                        f"PGD escalation bypassed: FGSM ASR ({fgsm_asr:.1f}%) and accuracy drop ({fgsm_acc_drop:.1f}%) "
                        f"remained strictly below escalation thresholds (ASR < {self.asr_escalation_threshold:.1f}%, drop < {self.accuracy_drop_threshold:.1f}%). "
                        f"With remaining budget limited to {remaining_budget}, the orchestrator conserves evaluation budget "
                        f"and concludes high baseline empirical robustness."
                    ),
                    supporting_evidence={
                        "clean_accuracy": clean_acc,
                        "fgsm_accuracy": fgsm_acc,
                        "fgsm_asr": fgsm_asr,
                        "accuracy_drop": fgsm_acc_drop,
                    },
                    expected_information="Conclude robustness without unnecessary iterative probe.",
                    budget_cost=0,
                    remaining_budget=remaining_budget,
                    timestamp=timestamp,
                    condition_evaluated=f"fgsm_asr ({fgsm_asr:.1f}%) < {self.asr_escalation_threshold:.1f}% and budget ({remaining_budget}) < 2",
                    threshold_used=f"ASR < {self.asr_escalation_threshold:.1f}% -> PGD Bypassed",
                    measured_value=f"ASR: {fgsm_asr:.1f}%, Drop: {fgsm_acc_drop:.1f}%",
                )

            return DecisionRecord(
                step_number=step_number,
                action=DecisionAction.ESCALATE_PGD,
                reason=reason,
                supporting_evidence={
                    "clean_accuracy": clean_acc,
                    "fgsm_accuracy": fgsm_acc,
                    "fgsm_attack_success_rate": fgsm_asr,
                    "accuracy_drop": fgsm_acc_drop,
                },
                expected_information="Iterative multi-step Linf robustness, loss trajectory, and peak perturbation.",
                budget_cost=1,
                remaining_budget=remaining_budget,
                timestamp=timestamp,
                condition_evaluated=condition_eval,
                threshold_used=thresh_used,
                measured_value=meas_val,
            )

        pgd_metrics = metrics_by_action.get("ESCALATE_PGD", {})
        pgd_acc = pgd_metrics.get("adversarial_accuracy", clean_acc)
        pgd_asr = pgd_metrics.get("attack_success_rate", 0.0)

        # Rule 8: Defense verification
        if "TEST_DEFENSE" not in completed_actions:
            highest_asr = max(fgsm_asr, pgd_asr)
            if highest_asr >= self.defense_trigger_asr or min(fgsm_acc, pgd_acc) < clean_acc - 10.0:
                reason = (
                    f"Input quantization defense was selected because adversarial attacks demonstrated "
                    f"significant effectiveness (peak ASR: {highest_asr:.1f}%). Testing input quantization "
                    f"measures whether high-frequency spatial perturbations can be mitigated by empirical discretization."
                )
            else:
                reason = (
                    f"Evaluating input quantization defense as a baseline mitigation check. Even though "
                    f"observed attack success was modest (peak ASR: {highest_asr:.1f}%), evaluating defense impact "
                    f"verifies whether quantization preserves clean accuracy without introducing gradient obfuscation."
                )

            return DecisionRecord(
                step_number=step_number,
                action=DecisionAction.TEST_DEFENSE,
                reason=reason,
                supporting_evidence={
                    "clean_accuracy": clean_acc,
                    "fgsm_attack_success_rate": fgsm_asr,
                    "pgd_attack_success_rate": pgd_asr,
                    "peak_asr": highest_asr,
                },
                expected_information="Clean and adversarial recovery rates under 16-level input quantization.",
                budget_cost=1,
                remaining_budget=remaining_budget,
                timestamp=timestamp,
                condition_evaluated=f"peak_asr ({highest_asr:.1f}%) >= {self.defense_trigger_asr:.1f}%",
                threshold_used=f"peak_asr >= {self.defense_trigger_asr:.1f}%",
                measured_value=f"peak_asr = {highest_asr:.1f}%",
            )

        # Rule 9: Stop with sufficient evidence
        return DecisionRecord(
            step_number=step_number,
            action=DecisionAction.STOP_SUFFICIENT_EVIDENCE,
            reason=(
                "Testing completed successfully. Sufficient evidence has been acquired across preflight integrity, "
                "clean baseline, single-step FGSM, multi-step iterative PGD, and input quantization defense."
            ),
            supporting_evidence={
                "clean_accuracy": clean_acc,
                "fgsm_accuracy": fgsm_acc,
                "pgd_accuracy": pgd_acc,
                "fgsm_asr": fgsm_asr,
                "pgd_asr": pgd_asr,
                "completed_tests": completed_actions,
            },
            expected_information="Comprehensive evidence-backed robustness report ready.",
            budget_cost=0,
            remaining_budget=remaining_budget,
            timestamp=timestamp,
            condition_evaluated="all core evaluation phases completed",
            threshold_used="Core testing protocol completed",
            measured_value=f"{len(completed_actions)} actions executed",
        )
