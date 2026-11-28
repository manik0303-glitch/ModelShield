"""
ModelShield Adaptive Investigation Copilot
Evidence-grounded assistant for explaining orchestrator decisions,
interpreting measurements, and guiding cybersecurity investigations.
"""
from typing import Any, Dict, List, Optional


class InvestigationCopilot:
    """
    Transparent, evidence-grounded AI copilot for ModelShield.
    Generates briefings and explains decisions using actual recorded run metrics.
    """

    def __init__(self, run_data: Optional[Dict[str, Any]] = None):
        self.run_data = run_data or {}

    def update_run_data(self, run_data: Dict[str, Any]) -> None:
        self.run_data = run_data

    def generate_briefing(
        self,
        model_name: str,
        dataset_name: str,
        epsilon: float,
        budget: int,
    ) -> Dict[str, Any]:
        """Generate pre-flight mission briefing."""
        return {
            "title": "Red-Team Investigation Briefing",
            "scope": f"Model: {model_name} on {dataset_name}",
            "threat_model": f"L-infinity norm perturbation budget (epsilon = {epsilon:.4f} ~ {int(round(epsilon * 255))}/255)",
            "budget": f"{budget} evaluation actions allocated",
            "planned_evaluations": [
                "Preflight validation: tensor dimensionality, pixel bounds, and finite output integrity",
                "Clean baseline: non-adversarial accuracy benchmark and prediction distribution",
                "FGSM probe: single-step linear loss gradient sensitivity",
                "Adaptive PGD escalation: multi-step iterative projection with random initialization",
                "Defense mitigation: input quantization recovery analysis",
            ],
            "capabilities": (
                "Identifies empirical gradient vulnerabilities, measures perturbation compliance, "
                "evaluates input discretization defense, and enforces strict testing budgets."
            ),
            "limitations_notice": (
                "Empirical evaluations bounded by Linf do not guarantee certification against "
                "AutoAttack, black-box query attacks, or adaptive defense-tailored attacks."
            ),
        }

    def explain_decision(self, decision: Dict[str, Any]) -> str:
        """Provide a detailed, grounded technical explanation for a specific decision record."""
        action = decision.get("action", "")
        reason = decision.get("reason", "")
        evidence = decision.get("supporting_evidence", {})
        budget_left = decision.get("remaining_budget", 0)

        explanation = f"### Decision Analysis: `{action}`\n\n"
        explanation += f"**Reason:** {reason}\n\n"

        if evidence:
            explanation += "**Measurements & Supporting Evidence:**\n"
            for k, v in evidence.items():
                if isinstance(v, float):
                    explanation += f"- `{k}`: **{v:.3f}**\n"
                elif isinstance(v, (int, str, bool)):
                    explanation += f"- `{k}`: `{v}`\n"
                elif isinstance(v, list):
                    explanation += f"- `{k}`: {', '.join(str(x) for x in v)}\n"
            explanation += "\n"

        explanation += f"**Remaining Test Budget:** `{budget_left}` step(s).\n"
        return explanation

    def answer_inquiry(self, query: str) -> str:
        """Answer user questions grounded strictly in actual run measurements and definitions."""
        q = query.lower().strip()

        if not self.run_data:
            return (
                "⚠️ **No active evaluation run data available.** Please execute or select an evaluation run "
                "first so I can ground my analysis in real measurements rather than generic speculation."
            )

        decisions = self.run_data.get("decisions", [])
        steps = self.run_data.get("step_history", [])
        assessment = self.run_data.get("final_assessment", {})
        config = self.run_data.get("config", {})
        detailed_samples = self.run_data.get("detailed_samples", [])
        class_metrics = self.run_data.get("class_metrics", {})

        # Question 0: Sample-specific inquiry (e.g. "Sample 1", "Why did sample 2 fail?")
        import re
        sample_match = re.search(r"sample\s*#?\s*(\d+)", q)
        if sample_match and detailed_samples:
            s_num = int(sample_match.group(1))
            target_sample = next((s for s in detailed_samples if s.get("sample_id") == s_num), None)
            if target_sample:
                ram_mode = self.run_data.get("ram_mode", False)
                fooled = target_sample.get("fooled", False)
                if ram_mode:
                    return (
                        f"**RAM++ Investigation Image Dossier:**\n\n"
                        f"- **Baseline Tags:** `{target_sample.get('clean_class', 'N/A')}`\n"
                        f"- **After FGSM:** `{target_sample.get('fgsm_class', 'N/A')}` "
                        f"(tag preservation: {target_sample.get('fgsm_confidence', 0.0):.1f}%, Linf={target_sample.get('fgsm_l_inf', 'N/A')})\n"
                        f"- **After PGD:** `{target_sample.get('pgd_class', 'N/A')}` "
                        f"(tag preservation: {target_sample.get('pgd_confidence', 0.0):.1f}%, Linf={target_sample.get('pgd_l_inf', 'N/A')})\n"
                        f"- **Status:** {'💥 TAGS SUPPRESSED BY ADVERSARIAL PERTURBATION' if fooled else '✅ TAG PROFILE RESILIENT UNDER TESTED PERTURBATIONS'}\n\n"
                        f"**Gradient Vulnerability Analysis:**\n"
                        f"RAM++ recognized `{target_sample.get('clean_class', 'N/A')}` on the unperturbed image. "
                        + (
                            f"Under adversarial perturbation, these semantic tags were suppressed, indicating "
                            f"that the tagging head's internal score activations are sensitive to L∞-bounded noise."
                            if fooled
                            else f"The tag score margins were sufficiently wide to resist the ε={config.get('epsilon', 8/255):.4f} perturbation."
                        )
                    )
                return (
                    f"**Technical Dossier for Sample {s_num}:**\n\n"
                    f"- **Ground Truth:** `{target_sample.get('true_class')}`\n"
                    f"- **Clean Prediction:** `{target_sample.get('clean_class')}` ({target_sample.get('clean_confidence')}% confidence)\n"
                    f"- **FGSM Prediction:** `{target_sample.get('fgsm_class')}` ({target_sample.get('fgsm_confidence')}% confidence, Linf={target_sample.get('fgsm_l_inf', 'N/A')})\n"
                    f"- **PGD Prediction:** `{target_sample.get('pgd_class')}` ({target_sample.get('pgd_confidence')}% confidence, Linf={target_sample.get('pgd_l_inf', 'N/A')})\n"
                    f"- **Status:** {'💥 FOOLED BY ADVERSARIAL ASCENT' if fooled else ('✅ RESILIENT UNDER TESTED PERTURBATIONS' if target_sample.get('clean_class') == target_sample.get('true_class') else '❌ INITIALLY MISCLASSIFIED')}\n\n"
                    f"**Gradient Vulnerability Analysis:**\n"
                    f"When unperturbed, the model predicted `{target_sample.get('clean_class')}`. "
                    + (
                        f"Under iterative PGD-10 gradient ascent, the feature representation was pushed across the decision boundary into `{target_sample.get('pgd_class')}`, completely collapsing true-class activation."
                        if fooled
                        else f"The decision margin for `{target_sample.get('true_class')}` remained sufficiently wide to resist the Linf={config.get('epsilon', 8/255):.4f} perturbation."
                    )
                )

        # Question 0b: Class-wise vulnerability & flip analysis
        # For RAM++ mode, class_metrics will be empty; show tag-based summary instead.
        if "class" in q or "weakness" in q or "worst class" in q or "most vulnerable" in q or "tag" in q:
            ram_mode = self.run_data.get("ram_mode", False)
            if ram_mode:
                # RAM++ tag suppression summary
                fgsm_step = next((s for s in steps if s.get("action_type") == "RUN_FGSM"), None)
                pgd_step = next((s for s in steps if s.get("action_type") == "ESCALATE_PGD"), None)
                baseline_tags = self.run_data.get("sample_inspection", {}).get("baseline_tags", [])
                fgsm_tags = self.run_data.get("sample_inspection", {}).get("fgsm_tags", [])
                pgd_tags = self.run_data.get("sample_inspection", {}).get("pgd_tags", [])
                fgsm_pres = fgsm_step["metrics"].get("tag_preservation_rate", "N/A") if fgsm_step else "N/A"
                pgd_pres = pgd_step["metrics"].get("tag_preservation_rate", "N/A") if pgd_step else "N/A"
                disappeared = [t for t in baseline_tags if t not in pgd_tags] if pgd_tags else []
                return (
                    f"**RAM++ Tag Suppression Analysis:**\n\n"
                    f"- **Baseline tags detected:** {len(baseline_tags)} tags\n"
                    f"- **After FGSM:** tag preservation = `{fgsm_pres:.2%}` ({len(fgsm_tags)} tags remain)\n" if isinstance(fgsm_pres, float) else
                    f"- **After FGSM:** tag preservation = `{fgsm_pres}` \n"
                    + (
                        f"- **After PGD:** tag preservation = `{pgd_pres:.2%}` ({len(pgd_tags)} tags remain)\n" if isinstance(pgd_pres, float) else
                        f"- **After PGD:** tag preservation = `{pgd_pres}`\n"
                    )
                    + (f"- **Most suppressed tags:** {', '.join(disappeared[:10])}\n" if disappeared else "")
                    + "\nTag suppression measures how many of RAM++'s recognized semantic concepts "
                    "are eliminated by imperceptible perturbations. High suppression rates indicate "
                    "vulnerability in the model's tagging head."
                )
            if class_metrics:
                sorted_classes = sorted(
                    class_metrics.items(),
                    key=lambda item: item[1].get("pgd_acc", item[1].get("fgsm_acc", 100.0)),
                )
                breakdown = "\n".join(
                    f"- `{cls}`: Clean {m.get('clean_acc')}% → FGSM {m.get('fgsm_acc')}% → PGD {m.get('pgd_acc')}%"
                    for cls, m in sorted_classes
                )
                worst_cls, worst_m = sorted_classes[0]
                return (
                    f"**Class-wise Robustness & Vulnerability Analysis:**\n\n"
                    f"The most vulnerable class under gradient attacks was **`{worst_cls}`**, which degraded to "
                    f"`{worst_m.get('pgd_acc')}%` adversarial accuracy under PGD.\n\n"
                    f"**Full Class Breakdown:**\n"
                    f"{breakdown}\n\n"
                    f"Disproportionate drops in specific classes highlight asymmetric geometry in the model's decision boundaries."
                )

        # Question 1: Why PGD?
        if "why" in q and ("pgd" in q or "escalat" in q):
            pgd_decision = next((d for d in decisions if "PGD" in d.get("action", "")), None)
            if pgd_decision:
                return (
                    f"**Why the orchestrator chose PGD:**\n\n"
                    f"{pgd_decision.get('reason')}\n\n"
                    f"*Supporting measurements:*\n"
                    f"- Clean Accuracy: `{pgd_decision.get('supporting_evidence', {}).get('clean_accuracy', 'N/A')}%`\n"
                    f"- FGSM Adversarial Accuracy: `{pgd_decision.get('supporting_evidence', {}).get('fgsm_accuracy', 'N/A')}%`\n"
                    f"- FGSM Attack Success Rate: `{pgd_decision.get('supporting_evidence', {}).get('fgsm_attack_success_rate', 'N/A')}%`\n\n"
                    f"PGD (Projected Gradient Descent) performs multi-step iterative ascent within the Linf epsilon ball. "
                    f"While FGSM only computes a single gradient step, PGD solves an inner optimization problem that reveals "
                    f"whether the model suffers from non-linear decision boundary collapse."
                )
            else:
                return "PGD was not selected during this run or has not yet been reached in the decision pipeline."

        # Question 2: Why did testing stop?
        if "why" in q and ("stop" in q or "halt" in q or "end" in q or "finish" in q):
            last_decision = decisions[-1] if decisions else None
            if last_decision:
                return (
                    f"**Why testing concluded:**\n\n"
                    f"- **Terminal Action:** `{last_decision.get('action')}`\n"
                    f"- **Rationale:** {last_decision.get('reason')}\n"
                    f"- **Budget Consumed:** `{len(self.run_data.get('completed_actions', []))}` of `{config.get('max_budget', 'N/A')}` allotted steps.\n\n"
                    f"The orchestrator operates under strict budget limits and evidence-sufficiency rules to guarantee deterministic termination."
                )
            return "Execution history is currently empty."

        # Question 3: Difference between ASR and Adversarial Accuracy
        if "difference" in q or ("success rate" in q and "accuracy" in q) or "asr" in q:
            clean_acc = assessment.get("clean_accuracy", 0.0)
            worst_adv = assessment.get("worst_adversarial_accuracy", 0.0)
            max_asr = assessment.get("max_attack_success_rate", 0.0)

            return (
                "**Difference Between Adversarial Accuracy and Attack Success Rate (ASR):**\n\n"
                "1. **Adversarial Accuracy:**\n"
                "   - Defined as: `(Number of adversarial images classified correctly) / (Total images tested)`\n"
                f"   - In this run: Clean accuracy = **{clean_acc:.2f}%**, Adversarial accuracy = **{worst_adv:.2f}%**.\n\n"
                "2. **Attack Success Rate (ASR):**\n"
                "   - Crucial distinction: **ASR evaluates only the examples the clean model got right!**\n"
                "   - Defined as: `(Initially correct examples fooled by perturbation) / (Initially correct examples)`\n"
                f"   - In this run: Peak ASR = **{max_asr:.2f}%**.\n\n"
                "If a model already misclassifies an image when clean, that image is excluded from the ASR denominator. "
                "This ensures ASR measures genuine vulnerability caused by the adversarial perturbation, rather than baseline model inaccuracy."
            )

        # Question 4: Evidence supporting finding
        if "evidence" in q or "finding" in q or "support" in q:
            findings = assessment.get("key_findings", [])
            findings_text = "\n".join(f"- {f}" for f in findings) if findings else "No findings recorded yet."
            return (
                f"**Evidence Supporting the Final Assessment:**\n\n"
                f"Overall Assessment: **{assessment.get('overall_status', 'UNKNOWN')}**\n\n"
                f"**Key Empirical Measurements:**\n"
                f"{findings_text}\n\n"
                f"- Clean Baseline: `{assessment.get('clean_accuracy', 'N/A')}%`\n"
                f"- Worst Adversarial Accuracy: `{assessment.get('worst_adversarial_accuracy', 'N/A')}%`\n"
                f"- Peak Attack Success Rate: `{assessment.get('max_attack_success_rate', 'N/A')}%`\n"
                f"- Trust Checks: `{len(assessment.get('trust_warnings', []))} warning(s)` recorded."
            )

        # Question 5: Untested vectors
        if "untested" in q or "what else" in q or "not tested" in q:
            untested = assessment.get("untested_vectors", [])
            items = "\n".join(f"- {u}" for u in untested) if untested else "Untested vector list unavailable."
            return (
                f"**Attack Vectors Not Yet Tested:**\n\n"
                f"{items}\n\n"
                f"⚠️ *Important Security Rule:* Passing bounded FGSM and PGD evaluates first-order gradient sensitivity "
                f"under Linf norm, but does not prove security against AutoAttack, black-box decision attacks, or physical-world patches."
            )

        # Question 6: Presentation limitations
        if "limitation" in q or "present" in q or "college" in q or "project" in q:
            limits = assessment.get("scientific_limitations", [])
            items = "\n".join(f"- {l}" for l in limits) if limits else "No limitations compiled."
            return (
                f"**Key Scientific Limitations to Mention in Your Presentation:**\n\n"
                f"{items}\n\n"
                f"💡 *Tip for evaluators:* Emphasize that ModelShield follows rigorous AI security standards by "
                f"separating empirical evaluation from formal verification and avoiding false claims of complete immunity."
            )

        # Question 7: Defense impact
        if "defense" in q or "quantiz" in q:
            defense_step = next((s for s in steps if s.get("action_type") == "TEST_DEFENSE"), None)
            if defense_step:
                m = defense_step.get("metrics", {})
                return (
                    f"**Input Quantization Defense Performance:**\n\n"
                    f"- Clean Accuracy with Defense: `{m.get('defense_clean_accuracy', 'N/A')}%`\n"
                    f"- Defended FGSM Accuracy: `{m.get('defended_fgsm_accuracy', 'N/A')}%` (Gain: `{m.get('fgsm_recovery_gain', 0.0):+.2f}%`)\n"
                    f"- Defended PGD Accuracy: `{m.get('defended_pgd_accuracy', 'N/A')}%` (Gain: `{m.get('pgd_recovery_gain', 0.0):+.2f}%`)\n\n"
                    f"Quantization rounds continuous pixel values into discrete bins ({config.get('quantization_levels', 16)} levels). "
                    f"While this can neutralize small high-frequency perturbations, it does not guarantee protection against "
                    f"adaptive attackers who backpropagate through surrogate gradient approximations (BPDA)."
                )
            return "Defense evaluation has not been conducted in this run."

        # Generic grounded fallback
        ram_mode = self.run_data.get("ram_mode", False)
        clean_label = "Baseline Tag Count" if ram_mode else "Clean Accuracy"
        clean_val = assessment.get("clean_accuracy", "N/A")
        adv_label = "Worst Tag Preservation (%)" if ram_mode else "Worst Adversarial Accuracy"
        return (
            f"**Investigation Summary based on Run `{self.run_data.get('run_id')}`:**\n\n"
            f"- Model: `{self.run_data.get('model_name')}`\n"
            f"- Mode: `{'RAM++ Multi-label Tagging' if ram_mode else 'CIFAR-10 Classification'}`\n"
            f"- Status: **{assessment.get('overall_status', 'IN_PROGRESS')}**\n"
            f"- {clean_label}: `{clean_val}`\n"
            f"- {adv_label}: `{assessment.get('worst_adversarial_accuracy', 'N/A')}`\n"
            f"- Peak ASR: `{assessment.get('max_attack_success_rate', 'N/A')}%`\n"
            f"- Steps Completed: `{len(self.run_data.get('completed_actions', []))}` of `{config.get('max_budget', 'N/A')}`\n\n"
            f"You can ask me specific questions like:\n"
            f"- *'Why did the orchestrator choose PGD?'*\n"
            f"- *'What does the tag preservation rate mean?'*\n"
            f"- *'What attack vectors have not been tested?'*\n"
            f"- *'What limitations should I mention when presenting this result?'*\n"
            f"- *'What is the difference between attack success rate and tag preservation?'*"
        )

