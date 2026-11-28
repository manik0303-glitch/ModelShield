import streamlit as st
from pathlib import Path
import torch
import io
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor
from evaluation.defense_evaluation import evaluate_defense

from model.load_model import load_model
from robustbench.data import load_cifar10
from attacks.ensemble import evaluate_attack_ensemble
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from evaluation.trust_checks import (
    check_perturbation_budget,
    check_pixel_range,
    check_prediction_outputs,
    check_metric_consistency,
)
from evaluation.trust_flag import generate_trust_flag
from evaluation.gradient_masking import analyze_gradient_masking
import json
from datetime import datetime, timezone
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
    validate_snapshot_integrity,
)
from model.checkpoint_loader import (
    SupportedArchitecture,
    load_checkpoint_safely,
    load_baseline_model,
    infer_architecture_from_state_dict,
    clean_state_dict,
)
from evaluation.live_eval import (
    evaluate_model_live,
    is_snapshot_compatible_with_config,
    run_live_regression_pipeline,
    load_cifar10_samples,
)
from model.ram_loader import (
    load_ram_model,
    load_ram_original_model,
    load_ram_candidate_model,
    DEFAULT_RAM_CHECKPOINT,
)
from evaluation.ram_eval import (
    SimulatedDegradedRAMWrapper,
    ensure_benchmark_dataset,
    load_dataset_images,
    evaluate_ram_model_live,
    run_live_ram_regression_pipeline,
)
from evaluation.ram_gate_visuals import (
    inject_cyber_lab_css,
    render_model_identity_panels,
    render_execution_pipeline,
    render_security_verdict_panel,
    render_comparative_analytics,
    render_tag_impact_and_forensics,
)


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ModelShield",
    page_icon="🛡️",
    layout="wide",
)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# -------------------- MODEL AND DATA --------------------
@st.cache_resource
def get_model():
    model, device = load_model()
    model.eval()
    return model, device


@st.cache_data
def get_cifar_dataset(n_examples=32):
    images, labels = load_cifar10(n_examples=n_examples)
    return images, labels


def image_tensor_from_pil(pil_image):
    """Prepare an RGB image for the CIFAR-10 model: [1, 3, 32, 32], values [0,1]."""
    resized = pil_image.convert("RGB").resize((32, 32))
    tensor = to_tensor(resized).unsqueeze(0)
    return resized, tensor


def tensor_to_display(tensor):
    return (
        tensor.detach().cpu()[0]
        .permute(1, 2, 0)
        .clamp(0, 1)
        .numpy()
    )


def predict_image(model, image_tensor, device):
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        prediction = int(logits.argmax(dim=1).item())
    return prediction, float(probabilities[prediction].item() * 100), probabilities




def snapshot_from_cifar_session(role: str = "Snapshot"):
    """Extracts a serializable EvaluationSnapshot from active CIFAR-10 evaluation session state."""
    if st.session_state.get("cifar_eval_results") is None:
        return None
    results = st.session_state["cifar_eval_results"]
    dataset_epsilon = st.session_state.get("cifar_eval_epsilon")
    dataset_pgd_steps = st.session_state.get("cifar_eval_pgd_steps")
    trust_flag = st.session_state.get("cifar_eval_trust_flag")
    gm = st.session_state.get("gm_diag_results")

    if dataset_epsilon is None or dataset_pgd_steps is None or "total_images" not in results:
        return None

    clean_trust_flag = None
    if trust_flag and isinstance(trust_flag, dict):
        clean_trust_flag = {
            "status": str(trust_flag.get("status", "")),
            "summary": str(trust_flag.get("summary", "")),
            "passed": int(trust_flag.get("passed", 0)),
            "failed": int(trust_flag.get("failed", 0)),
            "total": int(trust_flag.get("total", 0)),
        }

    clean_gm = None
    if gm and isinstance(gm, dict):
        clean_gm = {
            "status": str(gm.get("status", "")),
            "summary": str(gm.get("summary", "")),
            "warning_reasons": list(gm.get("warning_reasons", [])),
        }

    metadata = EvaluationMetadata(
        model_name=f"Carmon2019Unlabeled_{role}",
        dataset_name="CIFAR-10",
        num_samples=int(results["total_images"]),
        epsilon=float(dataset_epsilon),
        alpha=float(min(2 / 255.0, dataset_epsilon)),
        pgd_steps=int(dataset_pgd_steps),
        random_start=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    metrics = {
        "clean_accuracy": float(results["clean_accuracy"]),
        "fgsm_accuracy": float(results["fgsm_accuracy"]),
        "pgd_accuracy": float(results["pgd_accuracy"]),
        "fgsm_attack_success_rate": float(results["fgsm_attack_success_rate"]),
        "pgd_attack_success_rate": float(results["pgd_attack_success_rate"]),
        "worst_case_attack_success_rate": float(results["worst_case_attack_success_rate"]),
    }
    return EvaluationSnapshot(
        metadata=metadata,
        metrics=metrics,
        trust_flag=clean_trust_flag,
        gradient_masking=clean_gm,
    )


def _render_cifar_regression_gate(device: torch.device):
    tab_live, tab_manual = st.tabs([
        "🚀 Live Model Evaluation",
        "📁 Snapshot Comparison & JSON Upload",
    ])

    # =========================================================================
    # TAB 1: LIVE MODEL EVALUATION (STAGES 1 - 4)
    # =========================================================================
    with tab_live:
        st.subheader("Live Model Upload & Genuine Robustness Evaluation")
        st.write(
            "Upload a new PyTorch model checkpoint (.pt or .pth) to automatically evaluate it on CIFAR-10 "
            "alongside the baseline model. Generates standardized evaluation snapshots and computes "
            "authoritative regression verdicts without requiring manual JSON creation."
        )

        st.markdown("---")
        col_st1, col_st2 = st.columns(2)

        # STAGE 1: SELECT BASELINE
        with col_st1:
            st.markdown("#### Stage 1 – Select Baseline Model")
            base_option = st.radio(
                "Baseline Model Source",
                [
                    "Carmon2019Unlabeled (Default CIFAR-10 Linf Baseline)",
                    "Reuse Active / Saved Baseline Snapshot",
                    "Upload Custom Baseline Checkpoint (.pt / .pth)",
                ],
                key="cifar_live_base_option",
            )

            baseline_model_loaded = None
            baseline_arch_resolved = SupportedArchitecture.WIDERESNET_28_10_CARMON
            baseline_name_resolved = "Carmon2019Unlabeled"

            if base_option == "Carmon2019Unlabeled (Default CIFAR-10 Linf Baseline)":
                carmon_path = "models/cifar10/Linf/Carmon2019Unlabeled.pt"
                import os
                if os.path.isfile(carmon_path):
                    st.success("✅ Baseline checkpoint found on disk (`Carmon2019Unlabeled.pt`, 38.1M params).")
                else:
                    st.info("ℹ️ Baseline will be loaded via RobustBench / load_model().")
                st.caption("Architecture: WideResNet-28-10 with sub_block1 · Dataset: CIFAR-10")

            elif base_option == "Reuse Active / Saved Baseline Snapshot":
                saved_base = st.session_state.get("gate_cifar_baseline_snapshot") or st.session_state.get("gate_baseline_snapshot")
                if saved_base is not None and getattr(saved_base.metadata, "task_type", "classification") == "classification":
                    st.success(f"✅ Loaded Snapshot: `{saved_base.metadata.model_name}`")
                    st.caption(
                        f"Samples: {saved_base.metadata.num_samples} · ε: {saved_base.metadata.epsilon:.6f} · "
                        f"Clean: {saved_base.metrics.get('clean_accuracy', 0.0):.2f}% · PGD: {saved_base.metrics.get('pgd_accuracy', 0.0):.2f}%"
                    )
                else:
                    st.warning("No classification baseline snapshot saved yet. Evaluate below or run CIFAR-10 evaluation first.")
                    if st.session_state.get("cifar_eval_results") is not None:
                        if st.button("Use Active CIFAR-10 Evaluation as Baseline", key="cifar_live_btn_use_active_base"):
                            snap = snapshot_from_cifar_session(role="Baseline")
                            if snap:
                                st.session_state["gate_cifar_baseline_snapshot"] = snap
                                st.session_state["gate_baseline_snapshot"] = snap
                                st.rerun()

            else:  # Custom baseline checkpoint
                base_upload = st.file_uploader(
                    "Upload Baseline Checkpoint (.pt, .pth)",
                    type=["pt", "pth"],
                    key="cifar_live_custom_base_upload",
                )
                if base_upload is not None:
                    baseline_name_resolved = st.text_input("Baseline Model Name", value=Path(base_upload.name).stem, key="cifar_live_base_name_input")
                    baseline_arch_resolved = st.selectbox(
                        "Baseline Architecture",
                        SupportedArchitecture.ALL,
                        index=0,
                        key="cifar_live_base_arch_select",
                    )
                    try:
                        base_bytes = base_upload.getvalue()
                        baseline_model_loaded, _, base_info = load_checkpoint_safely(
                            io.BytesIO(base_bytes),
                            architecture=baseline_arch_resolved,
                            device=device,
                        )
                        st.success(f"Baseline checkpoint verified: {base_info['total_parameters']:,} parameters.")
                    except Exception as exc:
                        st.error(f"Failed to load baseline checkpoint: {exc}")

        # STAGE 2: UPLOAD CANDIDATE
        with col_st2:
            st.markdown("#### Stage 2 – Upload Candidate Model")
            cand_upload = st.file_uploader(
                "Upload Candidate PyTorch Checkpoint (.pt, .pth)",
                type=["pt", "pth"],
                key="cifar_live_candidate_upload",
            )

            candidate_model_loaded = None
            candidate_name_resolved = "CandidateModel"
            candidate_arch_resolved = SupportedArchitecture.WIDERESNET_28_10_CARMON

            if cand_upload is not None:
                cand_bytes = cand_upload.getvalue()
                inferred_arch = None
                try:
                    temp_raw = torch.load(io.BytesIO(cand_bytes), map_location="cpu", weights_only=False)
                    if isinstance(temp_raw, dict):
                        csd = clean_state_dict(temp_raw)
                        inferred_arch = infer_architecture_from_state_dict(csd)
                except Exception:
                    pass

                default_idx = 0
                if inferred_arch and inferred_arch in SupportedArchitecture.ALL:
                    default_idx = SupportedArchitecture.ALL.index(inferred_arch)
                    st.info(f"💡 Detected architecture: **{inferred_arch}**")

                candidate_name_resolved = st.text_input(
                    "Candidate Model Name",
                    value=Path(cand_upload.name).stem,
                    key="cifar_live_cand_name_input",
                )

                candidate_arch_resolved = st.selectbox(
                    "Confirm or Select Candidate Architecture",
                    SupportedArchitecture.ALL,
                    index=default_idx,
                    key="cifar_live_cand_arch_select",
                    help="Select the model architecture matching your checkpoint weights.",
                )

                try:
                    candidate_model_loaded, _, cand_info = load_checkpoint_safely(
                        io.BytesIO(cand_bytes),
                        architecture=candidate_arch_resolved,
                        device=device,
                    )
                    st.success(
                        f"✅ Candidate verified: **{cand_info['total_parameters']:,}** parameters. "
                        f"Forward pass validated for CIFAR-10 shape (32×32)."
                    )
                except Exception as exc:
                    candidate_model_loaded = None
                    st.error(f"❌ Candidate verification failed: {exc}")
            else:
                st.info("Upload a candidate PyTorch checkpoint to enable live evaluation.")

        # STAGE 3: CONFIGURE EVALUATION
        st.markdown("---")
        st.markdown("#### Stage 3 – Configure Evaluation Settings")
        st.caption("Settings will be strictly applied to both Baseline and Candidate models for condition equality.")

        cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns(4)
        with cfg_c1:
            st.selectbox("Dataset", ["CIFAR-10"], index=0, disabled=True, key="cifar_live_dataset_select")
            live_samples = st.selectbox("Sample Count (Images)", [8, 16, 32, 64], index=1, key="cifar_live_samples_select")
        with cfg_c2:
            live_eps_pixels = st.slider("Attack Epsilon (pixels / 255)", 1, 16, 8, key="cifar_live_eps_slider")
            live_epsilon = live_eps_pixels / 255.0
            st.caption(f"ε = {live_epsilon:.6f}")
        with cfg_c3:
            live_pgd_steps = st.slider("PGD Iterations", 1, 20, 10, key="cifar_live_pgd_steps_slider")
        with cfg_c4:
            live_alpha = st.number_input(
                "PGD Step Size (α)",
                min_value=0.001,
                max_value=0.1,
                value=float(min(2 / 255.0, live_epsilon)),
                step=0.001,
                format="%.6f",
                key="cifar_live_alpha_input",
            )

        # STAGE 4: RUN EVALUATION
        st.markdown("---")
        st.markdown("#### Stage 4 – Run Live Evaluation")

        can_execute_live = candidate_model_loaded is not None or cand_upload is not None

        if st.button("🚀 Evaluate Models & Run Security Gate", type="primary", key="cifar_btn_run_live_eval", disabled=not can_execute_live):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(msg: str, frac: float):
                status_text.text(f"⏳ {msg}")
                progress_bar.progress(min(1.0, max(0.0, frac)))

            try:
                b_model = None
                existing_snap = None

                if base_option == "Reuse Active / Saved Baseline Snapshot":
                    existing_snap = st.session_state.get("gate_cifar_baseline_snapshot") or st.session_state.get("gate_baseline_snapshot")
                    if existing_snap is None:
                        raise ValueError("No baseline snapshot available to reuse.")
                elif base_option == "Upload Custom Baseline Checkpoint (.pt / .pth)":
                    b_model = baseline_model_loaded
                    if b_model is None:
                        raise ValueError("Custom baseline model checkpoint was not loaded successfully.")
                else:
                    update_progress("Loading Carmon2019Unlabeled baseline model...", 0.05)
                    b_model, b_name, b_info = load_baseline_model(device=device)

                c_model = candidate_model_loaded
                if c_model is None and cand_upload is not None:
                    c_model, _, _ = load_checkpoint_safely(
                        io.BytesIO(cand_upload.getvalue()),
                        architecture=candidate_arch_resolved,
                        device=device,
                    )

                if c_model is None:
                    raise ValueError("Candidate model could not be loaded.")

                thresholds = RegressionThresholds(
                    max_clean_accuracy_drop=float(st.session_state.get("cifar_th_m_clean", 2.0)),
                    warn_clean_accuracy_drop=float(st.session_state.get("cifar_th_w_clean", 0.5)),
                    max_pgd_accuracy_drop=float(st.session_state.get("cifar_th_m_pgd", 3.0)),
                    warn_pgd_accuracy_drop=float(st.session_state.get("cifar_th_w_pgd", 1.0)),
                    max_pgd_asr_increase=float(st.session_state.get("cifar_th_m_asr", 3.0)),
                    warn_pgd_asr_increase=float(st.session_state.get("cifar_th_w_asr", 1.0)),
                    enforce_condition_equality=bool(st.session_state.get("cifar_th_enforce_cond", True)),
                    require_trust_checks_pass=bool(st.session_state.get("cifar_th_req_trust", True)),
                    reject_gradient_masking=bool(st.session_state.get("cifar_th_rej_mask", True)),
                )

                b_snap, c_snap, gate_res = run_live_regression_pipeline(
                    baseline_model=b_model,
                    candidate_model=c_model,
                    baseline_name=baseline_name_resolved,
                    candidate_name=candidate_name_resolved,
                    candidate_architecture=candidate_arch_resolved,
                    baseline_architecture=baseline_arch_resolved,
                    existing_baseline_snapshot=existing_snap,
                    dataset_name="CIFAR-10",
                    n_samples=int(live_samples),
                    epsilon=float(live_epsilon),
                    alpha=float(live_alpha),
                    pgd_steps=int(live_pgd_steps),
                    thresholds=thresholds,
                    device=device,
                    progress_callback=update_progress,
                )

                st.session_state["gate_cifar_baseline_snapshot"] = b_snap
                st.session_state["gate_cifar_candidate_snapshot"] = c_snap
                st.session_state["gate_cifar_result"] = gate_res
                st.session_state["gate_baseline_snapshot"] = b_snap
                st.session_state["gate_candidate_snapshot"] = c_snap
                st.session_state["gate_result"] = gate_res

                progress_bar.progress(1.0)
                status_text.text("✅ Live evaluation and regression gate completed successfully!")
                st.success("🎉 Live evaluation complete! Scroll down to review comparative metrics and gate verdict.")

            except Exception as exc:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Live evaluation failed: {exc}")

    # =========================================================================
    # TAB 2: MANUAL SNAPSHOT COMPARISON & JSON UPLOAD
    # =========================================================================
    with tab_manual:
        st.subheader("Manual Evaluation Snapshots & JSON Upload")
        st.write(
            "Load existing evaluation snapshots from completed CIFAR-10 evaluation sessions or upload "
            "standardized JSON snapshot files directly."
        )

        base_snap: EvaluationSnapshot | None = st.session_state.get("gate_cifar_baseline_snapshot") or (
            st.session_state.get("gate_baseline_snapshot") if st.session_state.get("gate_baseline_snapshot") and getattr(st.session_state["gate_baseline_snapshot"].metadata, "task_type", "classification") == "classification" else None
        )
        cand_snap: EvaluationSnapshot | None = st.session_state.get("gate_cifar_candidate_snapshot") or (
            st.session_state.get("gate_candidate_snapshot") if st.session_state.get("gate_candidate_snapshot") and getattr(st.session_state["gate_candidate_snapshot"].metadata, "task_type", "classification") == "classification" else None
        )

        col_base, col_cand = st.columns(2)

        with col_base:
            st.subheader("1. Baseline Snapshot")
            if base_snap is not None:
                st.success(f"**Loaded:** `{base_snap.metadata.model_name}`")
                st.caption(
                    f"Dataset: **{base_snap.metadata.dataset_name}** ({base_snap.metadata.num_samples} samples) · "
                    f"ε: **{base_snap.metadata.epsilon:.6f}** · "
                    f"PGD steps: **{base_snap.metadata.pgd_steps}**"
                )
                b_m = base_snap.metrics
                st.write(
                    f"- Clean Acc: **{b_m.get('clean_accuracy', 0.0):.2f}%**\n"
                    f"- PGD Acc: **{b_m.get('pgd_accuracy', 0.0):.2f}%**\n"
                    f"- PGD ASR: **{b_m.get('pgd_attack_success_rate', 0.0):.2f}%**\n"
                    f"- Worst-case ASR: **{b_m.get('worst_case_attack_success_rate', 0.0):.2f}%**"
                )
                if base_snap.trust_flag:
                    tf_status = base_snap.trust_flag.get("status", "UNKNOWN")
                    st.caption(f"Trust checks: **{tf_status}** ({base_snap.trust_flag.get('passed', 0)}/{base_snap.trust_flag.get('total', 0)})")
                if base_snap.gradient_masking:
                    gm_status = base_snap.gradient_masking.get("status", "UNKNOWN")
                    st.caption(f"Gradient masking: **{gm_status}**")

                b_btn_col1, b_btn_col2 = st.columns(2)
                with b_btn_col1:
                    if st.button("Use Active Evaluation", key="cifar_btn_base_use_active"):
                        snap = snapshot_from_cifar_session(role="Baseline")
                        if snap is not None:
                            st.session_state["gate_cifar_baseline_snapshot"] = snap
                            st.session_state["gate_baseline_snapshot"] = snap
                            st.session_state["gate_cifar_result"] = None
                            st.rerun()
                        else:
                            st.warning("No completed CIFAR-10 evaluation in session state.")
                with b_btn_col2:
                    if st.button("Clear Baseline", key="cifar_btn_base_clear"):
                        st.session_state["gate_cifar_baseline_snapshot"] = None
                        st.session_state["gate_cifar_result"] = None
                        st.rerun()
            else:
                st.info("No baseline snapshot loaded.")
                if st.session_state.get("cifar_eval_results") is not None:
                    if st.button("Use Active CIFAR-10 Evaluation", key="cifar_btn_base_set_active"):
                        snap = snapshot_from_cifar_session(role="Baseline")
                        if snap is not None:
                            st.session_state["gate_cifar_baseline_snapshot"] = snap
                            st.session_state["gate_baseline_snapshot"] = snap
                            st.session_state["gate_cifar_result"] = None
                            st.rerun()
                else:
                    st.caption("Run an evaluation in *Test CIFAR-10 Dataset* or upload a JSON snapshot below.")

            up_base = st.file_uploader("Upload Baseline JSON", type=["json"], key="cifar_upload_baseline_json")
            if up_base is not None:
                try:
                    data = json.load(up_base)
                    snap = EvaluationSnapshot.from_dict(data)
                    task_type_val = getattr(snap.metadata, "task_type", "classification")
                    if task_type_val in ("image_tagging", "ram_tagging"):
                        st.error(f"❌ Incompatible Snapshot: This snapshot is for image tagging (task_type='{task_type_val}'), not classification. Please switch to Image Tagging (RAM++) above.")
                    else:
                        ok, errs, val_snap = validate_snapshot_integrity(snap, "Baseline")
                        if ok and val_snap:
                            st.session_state["gate_cifar_baseline_snapshot"] = val_snap
                            st.session_state["gate_baseline_snapshot"] = val_snap
                            st.session_state["gate_cifar_result"] = None
                            st.success("Baseline snapshot loaded from JSON.")
                        else:
                            st.error(f"Invalid snapshot JSON: {'; '.join(errs)}")
                except Exception as exc:
                    st.error(f"Failed to parse JSON file: {exc}")

        with col_cand:
            st.subheader("2. Candidate Snapshot")
            if cand_snap is not None:
                st.success(f"**Loaded:** `{cand_snap.metadata.model_name}`")
                st.caption(
                    f"Dataset: **{cand_snap.metadata.dataset_name}** ({cand_snap.metadata.num_samples} samples) · "
                    f"ε: **{cand_snap.metadata.epsilon:.6f}** · "
                    f"PGD steps: **{cand_snap.metadata.pgd_steps}**"
                )
                c_m = cand_snap.metrics
                st.write(
                    f"- Clean Acc: **{c_m.get('clean_accuracy', 0.0):.2f}%**\n"
                    f"- PGD Acc: **{c_m.get('pgd_accuracy', 0.0):.2f}%**\n"
                    f"- PGD ASR: **{c_m.get('pgd_attack_success_rate', 0.0):.2f}%**\n"
                    f"- Worst-case ASR: **{c_m.get('worst_case_attack_success_rate', 0.0):.2f}%**"
                )
                if cand_snap.trust_flag:
                    tf_status = cand_snap.trust_flag.get("status", "UNKNOWN")
                    st.caption(f"Trust checks: **{tf_status}** ({cand_snap.trust_flag.get('passed', 0)}/{cand_snap.trust_flag.get('total', 0)})")
                if cand_snap.gradient_masking:
                    gm_status = cand_snap.gradient_masking.get("status", "UNKNOWN")
                    st.caption(f"Gradient masking: **{gm_status}**")

                c_btn_col1, c_btn_col2 = st.columns(2)
                with c_btn_col1:
                    if st.button("Use Active Evaluation", key="cifar_btn_cand_use_active"):
                        snap = snapshot_from_cifar_session(role="Candidate")
                        if snap is not None:
                            st.session_state["gate_cifar_candidate_snapshot"] = snap
                            st.session_state["gate_candidate_snapshot"] = snap
                            st.session_state["gate_cifar_result"] = None
                            st.rerun()
                        else:
                            st.warning("No completed CIFAR-10 evaluation in session state.")
                with c_btn_col2:
                    if st.button("Clear Candidate", key="cifar_btn_cand_clear"):
                        st.session_state["gate_cifar_candidate_snapshot"] = None
                        st.session_state["gate_cifar_result"] = None
                        st.rerun()
            else:
                st.info("No candidate snapshot loaded.")
                if st.session_state.get("cifar_eval_results") is not None:
                    if st.button("Use Active CIFAR-10 Evaluation", key="cifar_btn_cand_set_active"):
                        snap = snapshot_from_cifar_session(role="Candidate")
                        if snap is not None:
                            st.session_state["gate_cifar_candidate_snapshot"] = snap
                            st.session_state["gate_candidate_snapshot"] = snap
                            st.session_state["gate_cifar_result"] = None
                            st.rerun()
                else:
                    st.caption("Run an evaluation in *Test CIFAR-10 Dataset* or upload a JSON snapshot below.")

            up_cand = st.file_uploader("Upload Candidate JSON", type=["json"], key="cifar_upload_candidate_json")
            if up_cand is not None:
                try:
                    data = json.load(up_cand)
                    snap = EvaluationSnapshot.from_dict(data)
                    task_type_val = getattr(snap.metadata, "task_type", "classification")
                    if task_type_val in ("image_tagging", "ram_tagging"):
                        st.error(f"❌ Incompatible Snapshot: This snapshot is for image tagging (task_type='{task_type_val}'), not classification. Please switch to Image Tagging (RAM++) above.")
                    else:
                        ok, errs, val_snap = validate_snapshot_integrity(snap, "Candidate")
                        if ok and val_snap:
                            st.session_state["gate_cifar_candidate_snapshot"] = val_snap
                            st.session_state["gate_candidate_snapshot"] = val_snap
                            st.session_state["gate_cifar_result"] = None
                            st.success("Candidate snapshot loaded from JSON.")
                        else:
                            st.error(f"Invalid snapshot JSON: {'; '.join(errs)}")
                except Exception as exc:
                    st.error(f"Failed to parse JSON file: {exc}")

    # =========================================================================
    # RECONCILED RESULTS & STAGES 5 & 6 (CIFAR-10)
    # =========================================================================
    st.divider()

    base_snap_curr: EvaluationSnapshot | None = st.session_state.get("gate_cifar_baseline_snapshot")
    cand_snap_curr: EvaluationSnapshot | None = st.session_state.get("gate_cifar_candidate_snapshot")

    with st.expander("⚙️ Regression Gate Thresholds & Policies", expanded=False):
        st.write("Configure allowable performance shifts (all values in percentage points):")
        th_c1, th_c2, th_c3 = st.columns(3)
        with th_c1:
            st.markdown("**Clean Accuracy Drop**")
            w_clean = st.number_input("Warning Drop (%)", min_value=0.0, max_value=20.0, value=0.5, step=0.1, key="cifar_th_w_clean")
            m_clean = st.number_input("Failure Drop (%)", min_value=0.0, max_value=50.0, value=2.0, step=0.1, key="cifar_th_m_clean")
            if w_clean > m_clean:
                st.warning("Warning threshold should be <= failure threshold.")

        with th_c2:
            st.markdown("**PGD Robust Accuracy Drop**")
            w_pgd = st.number_input("Warning Drop (%)", min_value=0.0, max_value=20.0, value=1.0, step=0.1, key="cifar_th_w_pgd")
            m_pgd = st.number_input("Failure Drop (%)", min_value=0.0, max_value=50.0, value=3.0, step=0.1, key="cifar_th_m_pgd")
            if w_pgd > m_pgd:
                st.warning("Warning threshold should be <= failure threshold.")

        with th_c3:
            st.markdown("**PGD ASR Surge**")
            w_asr = st.number_input("Warning Surge (%)", min_value=0.0, max_value=20.0, value=1.0, step=0.1, key="cifar_th_w_asr")
            m_asr = st.number_input("Failure Surge (%)", min_value=0.0, max_value=50.0, value=3.0, step=0.1, key="cifar_th_m_asr")
            if w_asr > m_asr:
                st.warning("Warning threshold should be <= failure threshold.")

        st.markdown("**Strict Policy Controls**")
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            enforce_cond = st.checkbox("Enforce condition equality", value=True, key="cifar_th_enforce_cond", help="Rejects comparisons if dataset, sample count, epsilon, or PGD steps differ.")
        with p_c2:
            req_trust = st.checkbox("Require candidate trust checks PASS", value=True, key="cifar_th_req_trust", help="Rejects candidate if perturbation budget, pixel range, or output shapes fail.")
        with p_c3:
            rej_mask = st.checkbox("Reject gradient masking WARNING", value=True, key="cifar_th_rej_mask", help="Rejects candidate if suspicious gradient masking signals are detected.")

    can_run = base_snap_curr is not None and cand_snap_curr is not None

    if st.button("Evaluate Gate on Loaded Snapshots", type="secondary", key="cifar_btn_run_gate", disabled=not can_run):
        thresholds = RegressionThresholds(
            max_clean_accuracy_drop=float(m_clean),
            warn_clean_accuracy_drop=float(w_clean),
            max_pgd_accuracy_drop=float(m_pgd),
            warn_pgd_accuracy_drop=float(w_pgd),
            max_pgd_asr_increase=float(m_asr),
            warn_pgd_asr_increase=float(w_asr),
            enforce_condition_equality=bool(enforce_cond),
            require_trust_checks_pass=bool(req_trust),
            reject_gradient_masking=bool(rej_mask),
        )
        try:
            with st.spinner("Evaluating security regression gate..."):
                res = evaluate_regression_gate(base_snap_curr, cand_snap_curr, thresholds=thresholds)
                st.session_state["gate_cifar_result"] = res
                st.session_state["gate_result"] = res
        except Exception as exc:
            st.error(f"Regression gate execution failed: {exc}")

    # STAGE 5: COMPARE RESULTS
    res: RegressionGateResult | None = st.session_state.get("gate_cifar_result")
    if res is not None:
        st.markdown("### Stage 5 – Regression Comparison Results")
        if res.verdict == GateVerdict.PASS:
            st.success(f"### {res.verdict.value} ✅\n\n{res.summary}")
        elif res.verdict == GateVerdict.WARN:
            st.warning(f"### {res.verdict.value} ⚠️\n\n{res.summary}")
        else:
            st.error(f"### {res.verdict.value} ❌\n\n{res.summary}")

        st.subheader("Comparative Metric Analysis")
        if res.metric_deltas:
            rows = []
            for k, delta in res.metric_deltas.items():
                icon = "✅ PASS" if delta.status == MetricStatus.OK else ("⚠️ WARN" if delta.status == MetricStatus.WARN else "❌ FAIL")
                sign = "+" if delta.delta > 0 else ""
                th_display = f"Max {delta.fail_threshold:.1f}%" if delta.fail_threshold is not None else "N/A"
                rows.append({
                    "Metric": k,
                    "Baseline": f"{delta.baseline_value:.2f}%",
                    "Candidate": f"{delta.candidate_value:.2f}%",
                    "Delta (Cand - Base)": f"{sign}{delta.delta:.2f}%",
                    "Status": icon,
                    "Threshold Limit": th_display,
                    "Diagnostic Explanation": delta.reason,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if res.failures:
            st.error("#### ❌ Critical Violations")
            for f in res.failures:
                st.write(f"- {f}")

        if res.warnings:
            st.warning("#### ⚠️ Warning-Level Findings")
            for w in res.warnings:
                st.write(f"- {w}")

        if res.passed_checks:
            with st.expander("Passed Sanity & Robustness Checks", expanded=False):
                for p in res.passed_checks:
                    st.write(f"- {p}")

        if res.condition_compatibility:
            with st.expander("Evaluation Condition Audit", expanded=False):
                audit_rows = []
                for cond, details in res.condition_compatibility.items():
                    if isinstance(details, dict):
                        audit_rows.append({
                            "Condition": cond,
                            "Baseline": str(details.get("baseline")),
                            "Candidate": str(details.get("candidate")),
                            "Compatible": "✅ YES" if details.get("matched") else "❌ NO",
                        })
                st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

        # STAGE 6: EXPORT REPORT
        st.markdown("### Stage 6 – Export Evaluation Artifacts")
        json_report = export_gate_report_json(res)
        md_report = export_gate_report_markdown(res)

        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
        with col_dl1:
            st.download_button(
                "📥 Download JSON Gate Report",
                data=json_report,
                file_name="model_security_regression_report.json",
                mime="application/json",
                key="cifar_dl_gate_json",
            )
        with col_dl2:
            st.download_button(
                "📄 Download Markdown Summary",
                data=md_report,
                file_name="model_security_regression_summary.md",
                mime="text/markdown",
                key="cifar_dl_gate_md",
            )
        with col_dl3:
            if base_snap_curr:
                st.download_button(
                    "📥 Baseline Snapshot (JSON)",
                    data=json.dumps(base_snap_curr.to_dict(), indent=2),
                    file_name="baseline_evaluation_snapshot.json",
                    mime="application/json",
                    key="cifar_dl_base_snap_json",
                )
        with col_dl4:
            if cand_snap_curr:
                st.download_button(
                    "📥 Candidate Snapshot (JSON)",
                    data=json.dumps(cand_snap_curr.to_dict(), indent=2),
                    file_name="candidate_evaluation_snapshot.json",
                    mime="application/json",
                    key="cifar_dl_cand_snap_json",
                )


def _render_ram_regression_gate(device: torch.device):
    inject_cyber_lab_css()
    tab_live, tab_manual = st.tabs([
        "🚀 Live Model Evaluation",
        "📁 Snapshot Comparison & JSON Upload",
    ])

    # =========================================================================
    # TAB 1: LIVE MODEL EVALUATION (STAGES 1 - 4)
    # =========================================================================
    with tab_live:
        st.subheader("Live Model Upload & Genuine Robustness Evaluation")
        st.write(
            "Evaluate **RAM++ (`ram_plus`)** or original **RAM (`ram`)** checkpoints on a standardized "
            "image suite alongside the official RAM++ baseline. Generates standardized evaluation snapshots and computes "
            "authoritative regression verdicts across multi-label semantic metrics without manual JSON creation."
        )

        render_execution_pipeline(
            gate_result=st.session_state.get("gate_ram_result"),
            has_baseline=st.session_state.get("gate_ram_baseline_snapshot") is not None,
            has_candidate=st.session_state.get("gate_ram_candidate_snapshot") is not None,
        )

        st.markdown("---")
        col_st1, col_st2 = st.columns(2)

        # STAGE 1: SELECT BASELINE
        with col_st1:
            st.markdown("#### Stage 1 – Select Baseline Model")
            base_option = st.radio(
                "Baseline Model Source",
                [
                    "Pretrained RAM++ Swin-Large (Official Baseline)",
                    "Reuse Active / Saved Baseline Snapshot",
                    "Upload Custom Baseline Checkpoint (.pt / .pth)",
                ],
                key="ram_live_base_option",
            )

            baseline_model_loaded = None
            baseline_name_resolved = "RAMPlus_SwinLarge_Official"
            baseline_arch_resolved = "ram_plus_swin_large"
            baseline_ckpt_resolved = "ram_plus_swin_large_14m.pth"

            if base_option == "Pretrained RAM++ Swin-Large (Official Baseline)":
                ram_pth = "pretrained/ram_plus_swin_large_14m.pth"
                if Path(ram_pth).is_file():
                    st.success("✅ Baseline checkpoint found on disk (`pretrained/ram_plus_swin_large_14m.pth`, 174.5M params).")
                else:
                    st.warning("⚠️ Baseline checkpoint file not found at `pretrained/ram_plus_swin_large_14m.pth`.")
                st.caption("Architecture: RAM++ (`ram_plus`) · Backbone: Swin-Large · Input: 384×384 · Multi-Label Tagging")

            elif base_option == "Reuse Active / Saved Baseline Snapshot":
                saved_base = st.session_state.get("gate_ram_baseline_snapshot")
                if saved_base is not None and getattr(saved_base.metadata, "task_type", None) in ("image_tagging", "ram_tagging"):
                    st.success(f"✅ Loaded Snapshot: `{saved_base.metadata.model_name}`")
                    st.caption(
                        f"Samples: {saved_base.metadata.num_samples} · ε: {saved_base.metadata.epsilon:.6f} · "
                        f"Jaccard: {saved_base.metrics.get('mean_jaccard_similarity', 0.0):.1f}% · Agreement: {saved_base.metrics.get('tag_agreement_rate', 0.0):.1f}%"
                    )
                else:
                    st.warning("No RAM++ baseline snapshot saved yet. Run live evaluation below or upload a snapshot in Tab 2.")

            else:  # Custom baseline checkpoint
                base_upload = st.file_uploader(
                    "Upload Baseline RAM++ Checkpoint (.pt, .pth)",
                    type=["pt", "pth"],
                    key="ram_live_custom_base_upload",
                )
                if base_upload is not None:
                    baseline_name_resolved = st.text_input("Baseline Model Name", value=Path(base_upload.name).stem, key="ram_live_base_name_input")
                    baseline_ckpt_resolved = base_upload.name
                    try:
                        base_bytes = base_upload.getvalue()
                        baseline_model_loaded, _, base_dev, base_info = load_ram_candidate_model(
                            io.BytesIO(base_bytes),
                            device=str(device),
                            architecture="ram_plus",
                        )
                        st.success(f"✅ Custom baseline checkpoint verified ({base_info.get('architecture')}) on {base_dev}.")
                    except Exception as exc:
                        st.error(f"Failed to load baseline checkpoint: {exc}")

        # STAGE 2: UPLOAD CANDIDATE
        with col_st2:
            st.markdown("#### Stage 2 – Upload Candidate Model")
            cand_arch_choice = st.selectbox(
                "Candidate Architecture",
                [
                    "RAM++ (ram_plus · Swin-Large)",
                    "RAM (ram · Swin-Large) — Coming Soon",
                ],
                index=0,
                key="ram_live_cand_arch",
                help="RAM++ (ram_plus) is active. Original RAM (ram) architecture support is ready and marked Coming Soon while checkpoint download is paused.",
            )
            is_orig_ram_coming_soon = "Coming Soon" in cand_arch_choice
            cand_arch_key = "ram" if is_orig_ram_coming_soon else "ram_plus"
            candidate_arch_resolved = "ram_swin_large" if is_orig_ram_coming_soon else "ram_plus_swin_large"

            candidate_model_loaded = None
            candidate_name_resolved = "RAMPlus_SelfConsistency_Candidate"
            candidate_ckpt_resolved = "ram_plus_swin_large_14m.pth"
            cand_mode = "Baseline as Candidate (Self-Consistency — PASS Demo)"

            if is_orig_ram_coming_soon:
                st.info(
                    "🚧 **Original RAM (`ram` · Swin-Large) — Coming Soon**\n\n"
                    "Architecture-ready loader (`load_ram_original_model`) is preserved in `model/ram_loader.py`, "
                    "and `ram_swin_large_14m.pth` download is paused. Switch back to **RAM++ (`ram_plus`)** above to run evaluations."
                )
            else:
                cand_modes = [
                    "Baseline as Candidate (Self-Consistency — PASS Demo)",
                    "Simulated Moderate Degradation (25% Tag Drop — Controlled WARN Demo)",
                    "Simulated Severe Degradation (75% Tag Drop — Controlled FAIL Demo)",
                    "Upload Custom RAM++ Checkpoint (.pt / .pth)",
                ]

                cand_mode = st.radio(
                    "Candidate Model Source",
                    cand_modes,
                    index=0,
                    key="ram_live_cand_mode",
                )

                if cand_mode == "Upload Custom RAM++ Checkpoint (.pt / .pth)":
                    cand_upload = st.file_uploader(
                        "Upload Candidate RAM++ Checkpoint (.pt, .pth)",
                        type=["pt", "pth"],
                        key="ram_live_candidate_upload",
                        help="Upload RAM++ (ram_plus) weights (.pt or .pth) with Swin-Large backbone.",
                    )

                    if cand_upload is not None:
                        cand_bytes = cand_upload.getvalue()
                        candidate_ckpt_resolved = cand_upload.name
                        candidate_name_resolved = st.text_input(
                            "Candidate Model Name",
                            value=Path(cand_upload.name).stem,
                            key="ram_live_cand_name_input",
                        )
                        st.caption("Architecture: RAM++ (`ram_plus`) · Backbone: Swin-Large · Input: 384×384")
                        try:
                            candidate_model_loaded, _, cand_dev, cand_info = load_ram_candidate_model(
                                io.BytesIO(cand_bytes),
                                device=str(device),
                                architecture="ram_plus",
                            )
                            st.success(f"✅ Candidate verified: {cand_info.get('architecture')} on {cand_dev}.")
                        except Exception as exc:
                            candidate_model_loaded = None
                            st.error(f"❌ Candidate verification failed: {exc}")
                    else:
                        st.info("Upload a candidate RAM++ PyTorch checkpoint (.pt, .pth) to evaluate.")

                elif cand_mode == "Simulated Moderate Degradation (25% Tag Drop — Controlled WARN Demo)":
                    candidate_name_resolved = st.text_input(
                        "Candidate Model Name",
                        value="RAMPlus_Simulated_25pct_Drop_Demo",
                        key="ram_live_cand_warn_name_input",
                    )
                    candidate_ckpt_resolved = "ram_plus_swin_large_14m.pth (simulated 25% tag drop)"
                    st.warning(
                        "⚠️ **Controlled Demonstration (Not a Real Trained Model):** "
                        "Wraps official RAM++ with `SimulatedDegradedRAMWrapper(drop_fraction=0.25)` "
                        "to demonstrate a moderate regression (WARN)."
                    )

                elif cand_mode == "Simulated Severe Degradation (75% Tag Drop — Controlled FAIL Demo)":
                    candidate_name_resolved = st.text_input(
                        "Candidate Model Name",
                        value="RAMPlus_Simulated_75pct_Drop_Demo",
                        key="ram_live_cand_fail_name_input",
                    )
                    candidate_ckpt_resolved = "ram_plus_swin_large_14m.pth (simulated 75% tag drop)"
                    st.warning(
                        "⚠️ **Controlled Demonstration (Not a Real Trained Model):** "
                        "Wraps official RAM++ with `SimulatedDegradedRAMWrapper(drop_fraction=0.75)` "
                        "to demonstrate a severe regression (FAIL)."
                    )

                else:
                    candidate_name_resolved = st.text_input(
                        "Candidate Model Name",
                        value="RAMPlus_SelfConsistency_Candidate",
                        key="ram_live_cand_demo_name_input",
                    )
                    st.info("ℹ️ Self-consistency check: evaluates the official RAM++ baseline against itself to verify PASS conditions.")

        # STAGE 3: CONFIGURE EVALUATION
        st.markdown("---")
        st.markdown("#### Stage 3 – Configure Evaluation Settings")
        st.caption("Benchmark dataset and perturbation conditions are strictly applied to both Baseline and Candidate models.")

        cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
        with cfg_c1:
            dataset_source = st.selectbox(
                "Benchmark Image Dataset",
                [
                    "Standard RAM++ Benchmark Suite (data/ram_benchmark/)",
                    "Upload Custom Image Archive (.zip)",
                    "Custom Local Directory",
                ],
                key="ram_live_dataset_source",
            )
            live_samples = st.selectbox("Sample Count (Images)", [4, 8, 16], index=0, key="ram_live_samples_select")

        custom_zip_file = None
        custom_dir_path = None
        if dataset_source == "Upload Custom Image Archive (.zip)":
            custom_zip_file = st.file_uploader("Upload Image ZIP archive", type=["zip"], key="ram_live_zip_upload")
        elif dataset_source == "Custom Local Directory":
            custom_dir_path = st.text_input("Directory path containing images", value="data/ram_benchmark", key="ram_live_dir_input")

        with cfg_c2:
            live_eps_pixels = st.slider("Perturbation Budget ε (pixels / 255)", 1, 16, 8, key="ram_live_eps_slider")
            live_epsilon = live_eps_pixels / 255.0
            st.caption(f"L∞ ε = {live_epsilon:.6f} ({live_eps_pixels}/255)")

        with cfg_c3:
            st.markdown("**Perturbation Engine**")
            st.info("L∞-Bounded High-Frequency Adversarial Perturbation (Deterministic, norm-bounded)")

        # STAGE 4: RUN EVALUATION
        st.markdown("---")
        st.markdown("#### Stage 4 – Run Live Evaluation")

        can_execute_live = (not is_orig_ram_coming_soon) and (
            cand_mode in (
                "Baseline as Candidate (Self-Consistency — PASS Demo)",
                "Simulated Moderate Degradation (25% Tag Drop — Controlled WARN Demo)",
                "Simulated Severe Degradation (75% Tag Drop — Controlled FAIL Demo)",
            )
            or (candidate_model_loaded is not None)
        )

        if st.button("🚀 Evaluate RAM++ Models & Run Security Gate", type="primary", key="ram_btn_run_live_eval", disabled=not can_execute_live):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(msg: str, frac: float):
                status_text.text(f"⏳ {msg}")
                progress_bar.progress(min(1.0, max(0.0, frac)))

            try:
                # 1. Resolve dataset
                update_progress("Preparing benchmark dataset images...", 0.05)
                if dataset_source == "Upload Custom Image Archive (.zip)":
                    if custom_zip_file is None:
                        raise ValueError("Please upload a ZIP file containing images.")
                    images, dataset_label = load_dataset_images(
                        source="zip",
                        uploaded_zip=custom_zip_file.getvalue(),
                        max_samples=int(live_samples),
                    )
                elif dataset_source == "Custom Local Directory":
                    if not custom_dir_path or not Path(custom_dir_path).is_dir():
                        raise ValueError(f"Directory '{custom_dir_path}' does not exist.")
                    images, dataset_label = load_dataset_images(
                        source="directory",
                        dir_path=custom_dir_path,
                        max_samples=int(live_samples),
                    )
                else:
                    images, dataset_label = load_dataset_images(
                        source="benchmark",
                        dir_path="data/ram_benchmark",
                        max_samples=int(live_samples),
                    )

                if not images:
                    raise ValueError("No valid images found in the selected dataset.")

                # 2. Resolve baseline model
                b_model = None
                ram_preprocess = None
                ram_device = device
                existing_snap = None
                if base_option == "Reuse Active / Saved Baseline Snapshot":
                    existing_snap = st.session_state.get("gate_ram_baseline_snapshot")
                    if existing_snap is None:
                        raise ValueError("No baseline snapshot available to reuse.")
                elif base_option == "Upload Custom Baseline Checkpoint (.pt / .pth)":
                    b_model = baseline_model_loaded
                    if b_model is None:
                        raise ValueError("Custom baseline model checkpoint was not loaded successfully.")
                else:
                    update_progress("Loading official RAM++ Swin-Large baseline model...", 0.15)
                    b_model, ram_preprocess, ram_device = load_ram_model()

                # 3. Resolve candidate model
                c_model = candidate_model_loaded
                if cand_mode == "Baseline as Candidate (Self-Consistency — PASS Demo)":
                    if b_model is not None:
                        c_model = b_model
                    else:
                        c_model, ram_preprocess, ram_device = load_ram_model()
                elif cand_mode == "Simulated Moderate Degradation (25% Tag Drop — Controlled WARN Demo)":
                    base_ref = b_model
                    if base_ref is None:
                        base_ref, ram_preprocess, ram_device = load_ram_model()
                    c_model = SimulatedDegradedRAMWrapper(base_ref, drop_fraction=0.25, seed=42)
                elif cand_mode == "Simulated Severe Degradation (75% Tag Drop — Controlled FAIL Demo)":
                    base_ref = b_model
                    if base_ref is None:
                        base_ref, ram_preprocess, ram_device = load_ram_model()
                    c_model = SimulatedDegradedRAMWrapper(base_ref, drop_fraction=0.75, seed=42)
                if c_model is None:
                    raise ValueError("Candidate model could not be loaded.")

                # 4. Construct thresholds
                thresholds = RegressionThresholds(
                    max_jaccard_drop=float(st.session_state.get("ram_th_m_jaccard", 8.0)),
                    warn_jaccard_drop=float(st.session_state.get("ram_th_w_jaccard", 3.0)),
                    max_agreement_drop=float(st.session_state.get("ram_th_m_agreement", 6.0)),
                    warn_agreement_drop=float(st.session_state.get("ram_th_w_agreement", 2.0)),
                    max_retention_drop=float(st.session_state.get("ram_th_m_retention", 8.0)),
                    warn_retention_drop=float(st.session_state.get("ram_th_w_retention", 3.0)),
                    max_tags_lost_increase=float(st.session_state.get("ram_th_m_lost", 2.5)),
                    warn_tags_lost_increase=float(st.session_state.get("ram_th_w_lost", 1.0)),
                    max_new_tags_increase=float(st.session_state.get("ram_th_m_new", 2.5)),
                    warn_new_tags_increase=float(st.session_state.get("ram_th_w_new", 1.0)),
                    max_latency_increase=float(st.session_state.get("ram_th_m_latency", 150.0)),
                    warn_latency_increase=float(st.session_state.get("ram_th_w_latency", 50.0)),
                    enforce_condition_equality=bool(st.session_state.get("ram_th_enforce_cond", True)),
                    require_trust_checks_pass=bool(st.session_state.get("ram_th_req_trust", True)),
                    reject_gradient_masking=bool(st.session_state.get("ram_th_rej_mask", False)),
                    enable_semantic_consistency=bool(st.session_state.get("ram_th_enable_semantic", True)),
                    min_semantic_consistency=float(st.session_state.get("ram_th_min_semantic", 60.0)),
                    max_semantic_consistency_drop=float(st.session_state.get("ram_th_drop_semantic", 10.0)),
                )

                # 5. Execute live RAM/RAM++ regression pipeline
                forensic_out: dict = {}
                b_snap, c_snap, gate_res = run_live_ram_regression_pipeline(
                    baseline_model=b_model,
                    candidate_model=c_model,
                    preprocess=ram_preprocess,
                    dataset_images=images,
                    baseline_name=baseline_name_resolved,
                    candidate_name=candidate_name_resolved,
                    existing_baseline_snapshot=existing_snap,
                    dataset_name=dataset_label,
                    epsilon=float(live_epsilon),
                    thresholds=thresholds,
                    device=ram_device,
                    progress_callback=update_progress,
                    baseline_architecture=baseline_arch_resolved,
                    candidate_architecture=candidate_arch_resolved,
                    baseline_checkpoint_name=baseline_ckpt_resolved,
                    candidate_checkpoint_name=candidate_ckpt_resolved,
                    forensic_out=forensic_out,
                )

                st.session_state["gate_ram_baseline_snapshot"] = b_snap
                st.session_state["gate_ram_candidate_snapshot"] = c_snap
                st.session_state["gate_ram_result"] = gate_res
                st.session_state["gate_ram_forensics"] = forensic_out
                st.session_state["gate_baseline_snapshot"] = b_snap
                st.session_state["gate_candidate_snapshot"] = c_snap
                st.session_state["gate_result"] = gate_res

                progress_bar.progress(1.0)
                status_text.text("✅ RAM++ live evaluation and regression gate completed successfully!")
                st.success("🎉 RAM++ evaluation complete! Scroll down to inspect the Comparative Metric Analysis table and Gate Verdict.")

            except Exception as exc:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Live RAM++ evaluation failed: {exc}")

    # =========================================================================
    # TAB 2: MANUAL SNAPSHOT COMPARISON & JSON UPLOAD
    # =========================================================================
    with tab_manual:
        st.subheader("Manual Evaluation Snapshots & JSON Upload")
        st.write(
            "Load existing evaluation snapshots from completed RAM++ evaluation sessions or upload "
            "standardized JSON snapshot files directly."
        )

        base_snap: EvaluationSnapshot | None = st.session_state.get("gate_ram_baseline_snapshot") or (
            st.session_state.get("gate_baseline_snapshot") if st.session_state.get("gate_baseline_snapshot") and getattr(st.session_state["gate_baseline_snapshot"].metadata, "task_type", None) in ("image_tagging", "ram_tagging") else None
        )
        cand_snap: EvaluationSnapshot | None = st.session_state.get("gate_ram_candidate_snapshot") or (
            st.session_state.get("gate_candidate_snapshot") if st.session_state.get("gate_candidate_snapshot") and getattr(st.session_state["gate_candidate_snapshot"].metadata, "task_type", None) in ("image_tagging", "ram_tagging") else None
        )

        col_base, col_cand = st.columns(2)

        with col_base:
            st.subheader("1. Baseline Snapshot")
            if base_snap is not None:
                st.success(f"**Loaded:** `{base_snap.metadata.model_name}`")
                task_badge = f"Task: **{base_snap.metadata.task_type}** · " if getattr(base_snap.metadata, "task_type", None) else ""
                st.caption(
                    f"{task_badge}Dataset: **{base_snap.metadata.dataset_name}** ({base_snap.metadata.num_samples} samples) · "
                    f"ε: **{base_snap.metadata.epsilon:.6f}**"
                )
                b_m = base_snap.metrics
                st.write(
                    f"- Mean Jaccard Similarity: **{b_m.get('mean_jaccard_similarity', 0.0):.1f}%**\n"
                    f"- Tag Agreement Rate: **{b_m.get('tag_agreement_rate', 0.0):.1f}%**\n"
                    f"- Tag Retention Rate: **{b_m.get('mean_tag_retention_rate', 0.0):.1f}%**\n"
                    f"- Semantic Consistency: **{b_m.get('semantic_consistency', 0.0):.1f}%**\n"
                    f"- Avg Tags Detected: **{b_m.get('avg_tags_detected', 0.0):.1f} tags**\n"
                    f"- Tags Lost / Image: **{b_m.get('tags_lost_per_image', 0.0):.2f} tags**\n"
                    f"- New Tags / Image: **{b_m.get('new_tags_per_image', 0.0):.2f} tags**\n"
                    f"- Mean Latency: **{b_m.get('mean_latency_ms', 0.0):.1f} ms**"
                )
                if base_snap.trust_flag:
                    tf_status = base_snap.trust_flag.get("status", "UNKNOWN")
                    st.caption(f"Trust checks: **{tf_status}** ({base_snap.trust_flag.get('passed', 0)}/{base_snap.trust_flag.get('total', 0)})")
                if base_snap.gradient_masking:
                    gm_status = base_snap.gradient_masking.get("status", "UNKNOWN")
                    st.caption(f"Gradient masking: **{gm_status}**")

                if st.button("Clear Baseline", key="ram_btn_base_clear"):
                    st.session_state["gate_ram_baseline_snapshot"] = None
                    st.session_state["gate_ram_result"] = None
                    st.rerun()
            else:
                st.info("No baseline snapshot loaded. Run a live evaluation or upload a JSON snapshot below.")

            up_base = st.file_uploader("Upload Baseline JSON", type=["json"], key="ram_upload_baseline_json")
            if up_base is not None:
                try:
                    data = json.load(up_base)
                    snap = EvaluationSnapshot.from_dict(data)
                    task_type_val = getattr(snap.metadata, "task_type", "classification")
                    if task_type_val == "classification":
                        st.error(f"❌ Incompatible Snapshot: This snapshot is for image classification (task_type='{task_type_val}'), not image tagging. Please switch to Image Classification (CIFAR-10) above.")
                    else:
                        ok, errs, val_snap = validate_snapshot_integrity(snap, "Baseline")
                        if ok and val_snap:
                            st.session_state["gate_ram_baseline_snapshot"] = val_snap
                            st.session_state["gate_baseline_snapshot"] = val_snap
                            st.session_state["gate_ram_result"] = None
                            st.success("Baseline snapshot loaded from JSON.")
                        else:
                            st.error(f"Invalid snapshot JSON: {'; '.join(errs)}")
                except Exception as exc:
                    st.error(f"Failed to parse JSON file: {exc}")

        with col_cand:
            st.subheader("2. Candidate Snapshot")
            if cand_snap is not None:
                st.success(f"**Loaded:** `{cand_snap.metadata.model_name}`")
                c_task_badge = f"Task: **{cand_snap.metadata.task_type}** · " if getattr(cand_snap.metadata, "task_type", None) else ""
                st.caption(
                    f"{c_task_badge}Dataset: **{cand_snap.metadata.dataset_name}** ({cand_snap.metadata.num_samples} samples) · "
                    f"ε: **{cand_snap.metadata.epsilon:.6f}**"
                )
                c_m = cand_snap.metrics
                st.write(
                    f"- Mean Jaccard Similarity: **{c_m.get('mean_jaccard_similarity', 0.0):.1f}%**\n"
                    f"- Tag Agreement Rate: **{c_m.get('tag_agreement_rate', 0.0):.1f}%**\n"
                    f"- Tag Retention Rate: **{c_m.get('mean_tag_retention_rate', 0.0):.1f}%**\n"
                    f"- Semantic Consistency: **{c_m.get('semantic_consistency', 0.0):.1f}%**\n"
                    f"- Avg Tags Detected: **{c_m.get('avg_tags_detected', 0.0):.1f} tags**\n"
                    f"- Tags Lost / Image: **{c_m.get('tags_lost_per_image', 0.0):.2f} tags**\n"
                    f"- New Tags / Image: **{c_m.get('new_tags_per_image', 0.0):.2f} tags**\n"
                    f"- Mean Latency: **{c_m.get('mean_latency_ms', 0.0):.1f} ms**"
                )
                if cand_snap.trust_flag:
                    tf_status = cand_snap.trust_flag.get("status", "UNKNOWN")
                    st.caption(f"Trust checks: **{tf_status}** ({cand_snap.trust_flag.get('passed', 0)}/{cand_snap.trust_flag.get('total', 0)})")
                if cand_snap.gradient_masking:
                    gm_status = cand_snap.gradient_masking.get("status", "UNKNOWN")
                    st.caption(f"Gradient masking: **{gm_status}**")

                if st.button("Clear Candidate", key="ram_btn_cand_clear"):
                    st.session_state["gate_ram_candidate_snapshot"] = None
                    st.session_state["gate_ram_result"] = None
                    st.rerun()
            else:
                st.info("No candidate snapshot loaded. Run a live evaluation or upload a JSON snapshot below.")

            up_cand = st.file_uploader("Upload Candidate JSON", type=["json"], key="ram_upload_candidate_json")
            if up_cand is not None:
                try:
                    data = json.load(up_cand)
                    snap = EvaluationSnapshot.from_dict(data)
                    task_type_val = getattr(snap.metadata, "task_type", "classification")
                    if task_type_val == "classification":
                        st.error(f"❌ Incompatible Snapshot: This snapshot is for image classification (task_type='{task_type_val}'), not image tagging. Please switch to Image Classification (CIFAR-10) above.")
                    else:
                        ok, errs, val_snap = validate_snapshot_integrity(snap, "Candidate")
                        if ok and val_snap:
                            st.session_state["gate_ram_candidate_snapshot"] = val_snap
                            st.session_state["gate_candidate_snapshot"] = val_snap
                            st.session_state["gate_ram_result"] = None
                            st.success("Candidate snapshot loaded from JSON.")
                        else:
                            st.error(f"Invalid snapshot JSON: {'; '.join(errs)}")
                except Exception as exc:
                    st.error(f"Failed to parse JSON file: {exc}")

    # =========================================================================
    # RECONCILED RESULTS & STAGES 5 & 6 (RAM++)
    # =========================================================================
    st.divider()

    base_snap_curr: EvaluationSnapshot | None = st.session_state.get("gate_ram_baseline_snapshot")
    cand_snap_curr: EvaluationSnapshot | None = st.session_state.get("gate_ram_candidate_snapshot")

    with st.expander("⚙️ Regression Gate Thresholds & Policies", expanded=False):
        st.write("Configure allowable multi-label tagging performance shifts for RAM++ foundation models:")
        th_c1, th_c2, th_c3 = st.columns(3)
        with th_c1:
            st.markdown("**Mean Jaccard Similarity Drop**")
            w_jaccard = st.number_input("Warning Drop (%)", min_value=0.0, max_value=30.0, value=3.0, step=0.5, key="ram_th_w_jaccard")
            m_jaccard = st.number_input("Failure Drop (%)", min_value=0.0, max_value=50.0, value=8.0, step=0.5, key="ram_th_m_jaccard")

            st.markdown("**Tag Retention Drop**")
            w_ret = st.number_input("Warning Drop (%)", min_value=0.0, max_value=30.0, value=3.0, step=0.5, key="ram_th_w_retention")
            m_ret = st.number_input("Failure Drop (%)", min_value=0.0, max_value=50.0, value=8.0, step=0.5, key="ram_th_m_retention")

        with th_c2:
            st.markdown("**Tag Agreement Rate Drop**")
            w_agreement = st.number_input("Warning Drop (%)", min_value=0.0, max_value=30.0, value=2.0, step=0.5, key="ram_th_w_agreement")
            m_agreement = st.number_input("Failure Drop (%)", min_value=0.0, max_value=50.0, value=6.0, step=0.5, key="ram_th_m_agreement")

            st.markdown("**Tags Lost / Image Surge**")
            w_lost = st.number_input("Warning Surge (tags)", min_value=0.0, max_value=10.0, value=1.0, step=0.2, key="ram_th_w_lost")
            m_lost = st.number_input("Failure Surge (tags)", min_value=0.0, max_value=20.0, value=2.5, step=0.5, key="ram_th_m_lost")

        with th_c3:
            st.markdown("**New Tags / Image Surge**")
            w_new = st.number_input("Warning Surge (tags)", min_value=0.0, max_value=10.0, value=1.0, step=0.2, key="ram_th_w_new")
            m_new = st.number_input("Failure Surge (tags)", min_value=0.0, max_value=20.0, value=2.5, step=0.5, key="ram_th_m_new")

            st.markdown("**Mean Latency Increase**")
            w_lat = st.number_input("Warning Increase (ms)", min_value=0.0, max_value=500.0, value=50.0, step=10.0, key="ram_th_w_latency")
            m_lat = st.number_input("Failure Increase (ms)", min_value=0.0, max_value=2000.0, value=150.0, step=25.0, key="ram_th_m_latency")

        st.markdown("**Strict Policy Controls**")
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            enforce_cond = st.checkbox("Enforce condition equality", value=True, key="ram_th_enforce_cond", help="Rejects comparisons if dataset, sample count, or epsilon differ.")
        with p_c2:
            req_trust = st.checkbox("Require candidate trust checks PASS", value=True, key="ram_th_req_trust", help="Rejects candidate if perturbation budget, pixel range, or tensor shapes fail.")
        with p_c3:
            rej_mask = st.checkbox("Reject gradient masking WARNING", value=False, key="ram_th_rej_mask", help="Flags suspicious gradient masking signals if present.")

        st.markdown("**Semantic Consistency Policy (RAM++)**")
        sc_c1, sc_c2, sc_c3 = st.columns(3)
        with sc_c1:
            enable_semantic = st.checkbox(
                "Enable Semantic Consistency Check",
                value=True,
                key="ram_th_enable_semantic",
                help="Requires candidate model to preserve semantic visual tags under attack evaluated by RAM++.",
            )
        with sc_c2:
            min_semantic = st.number_input(
                "Min Semantic Consistency (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=5.0,
                key="ram_th_min_semantic",
                disabled=not enable_semantic,
            )
        with sc_c3:
            drop_semantic = st.number_input(
                "Max Consistency Drop vs Base (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                key="ram_th_drop_semantic",
                disabled=not enable_semantic,
            )

    # =========================================================================
    # SEMANTIC CONSISTENCY EVALUATION SECTION (RAM++ SINGLE-IMAGE TOOL)
    # =========================================================================
    with st.expander("🏷️ Semantic Tag Consistency Inspector (Single Image Pair)", expanded=False):
        st.write(
            "Quickly test and inspect whether adversarial perturbations alter or destroy high-level visual semantics. "
            "Runs the **RAM++ foundation model** on both the original clean image and its adversarial counterpart, "
            "extracting semantic English tags and computing Jaccard overlap, lost tags, and semantic consistency."
        )

        col_sc_img1, col_sc_img2 = st.columns(2)
        with col_sc_img1:
            clean_up = st.file_uploader(
                "Upload Clean Image (or original)",
                type=["jpg", "jpeg", "png", "webp"],
                key="ram_sc_clean_upload",
            )
        with col_sc_img2:
            adv_up = st.file_uploader(
                "Upload Adversarial Image (perturbed)",
                type=["jpg", "jpeg", "png", "webp"],
                key="ram_sc_adv_upload",
            )

        if st.button("🔍 Run Semantic Consistency Analysis (RAM++)", type="primary", key="ram_btn_run_sc_eval"):
            if clean_up is None or adv_up is None:
                st.warning("Please upload both a clean image and an adversarial image to evaluate.")
            else:
                try:
                    from evaluation.semantic_consistency import evaluate_semantic_consistency_images

                    with st.spinner("Loading RAM++ model and generating semantic tags..."):
                        ram_model, ram_preprocess, ram_device = load_ram_model()
                        img_c = Image.open(clean_up).convert("RGB")
                        img_a = Image.open(adv_up).convert("RGB")
                        sc_result = evaluate_semantic_consistency_images(
                            img_c, img_a, ram_model, ram_preprocess, device=ram_device
                        )
                        st.session_state["active_semantic_result"] = sc_result

                    st.success("✅ Semantic consistency analysis complete!")
                except Exception as exc:
                    st.error(f"Failed to execute RAM++ semantic evaluation: {exc}")

        active_sc = st.session_state.get("active_semantic_result")
        if active_sc is not None:
            st.markdown("---")
            st.subheader("Semantic Consistency Findings")

            min_th_val = float(st.session_state.get("ram_th_min_semantic", 60.0))
            is_pass = active_sc.semantic_consistency >= min_th_val
            if is_pass:
                st.success(
                    f"**Semantic Check: PASS ✅** — Semantic consistency is **{active_sc.semantic_consistency:.1f}%**, "
                    f"satisfying the minimum threshold of {min_th_val:.1f}%."
                )
            else:
                st.error(
                    f"**Semantic Check: FAIL ❌** — Semantic consistency is **{active_sc.semantic_consistency:.1f}%**, "
                    f"falling below the minimum threshold of {min_th_val:.1f}%."
                )

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Semantic Consistency", f"{active_sc.semantic_consistency:.1f}%")
            m_col2.metric("Jaccard Overlap Ratio", f"{active_sc.overlap_ratio:.3f}")
            m_col3.metric("Tags Lost", str(active_sc.tags_lost_count))
            m_col4.metric("New Tags Injected", str(active_sc.new_tags_count))

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**Clean Image Tags:**")
                st.info(", ".join(active_sc.clean_tags) if active_sc.clean_tags else "*(None)*")
                st.markdown("**Retained Overlap Tags:**")
                st.success(", ".join(active_sc.overlap_tags) if active_sc.overlap_tags else "*(None)*")
            with t_col2:
                st.markdown("**Adversarial Image Tags:**")
                st.info(", ".join(active_sc.adversarial_tags) if active_sc.adversarial_tags else "*(None)*")
                st.markdown("**Lost Tags (Destroyed by perturbation):**")
                st.error(", ".join(active_sc.lost_tags) if active_sc.lost_tags else "*(None)*")
                st.markdown("**New Tags (Injected by perturbation):**")
                st.warning(", ".join(active_sc.new_tags) if active_sc.new_tags else "*(None)*")

    can_run = base_snap_curr is not None and cand_snap_curr is not None

    if st.button("Evaluate Gate on Loaded Snapshots", type="secondary", key="ram_btn_run_gate", disabled=not can_run):
        thresholds = RegressionThresholds(
            max_jaccard_drop=float(m_jaccard),
            warn_jaccard_drop=float(w_jaccard),
            max_agreement_drop=float(m_agreement),
            warn_agreement_drop=float(w_agreement),
            max_retention_drop=float(m_ret),
            warn_retention_drop=float(w_ret),
            max_tags_lost_increase=float(m_lost),
            warn_tags_lost_increase=float(w_lost),
            max_new_tags_increase=float(m_new),
            warn_new_tags_increase=float(w_new),
            max_latency_increase=float(m_lat),
            warn_latency_increase=float(w_lat),
            enforce_condition_equality=bool(enforce_cond),
            require_trust_checks_pass=bool(req_trust),
            reject_gradient_masking=bool(rej_mask),
            enable_semantic_consistency=bool(enable_semantic),
            min_semantic_consistency=float(min_semantic),
            max_semantic_consistency_drop=float(drop_semantic),
        )
        try:
            with st.spinner("Evaluating security regression gate..."):
                res = evaluate_regression_gate(base_snap_curr, cand_snap_curr, thresholds=thresholds)
                st.session_state["gate_ram_result"] = res
                st.session_state["gate_result"] = res
        except Exception as exc:
            st.error(f"Regression gate execution failed: {exc}")

    # STAGE 5: COMPARE RESULTS
    res: RegressionGateResult | None = st.session_state.get("gate_ram_result")
    if res is not None:
        st.markdown("### Stage 5 – Security Investigation & Regression Results")
        render_execution_pipeline(gate_result=res)
        render_model_identity_panels(res.baseline_metadata, res.candidate_metadata)
        render_security_verdict_panel(res)

        if base_snap_curr is not None and cand_snap_curr is not None:
            render_comparative_analytics(res, base_snap_curr, cand_snap_curr)

        render_tag_impact_and_forensics(st.session_state.get("gate_ram_forensics"))

        st.subheader("Comparative Metric Analysis")
        if res.metric_deltas:
            rows = []
            for k, delta in res.metric_deltas.items():
                icon = "✅ PASS" if delta.status == MetricStatus.OK else ("⚠️ WARN" if delta.status == MetricStatus.WARN else "❌ FAIL")
                sign = "+" if delta.delta > 0 else ""

                if "latency" in k:
                    unit_str = " ms"
                    fmt = ".1f"
                elif "tags" in k:
                    unit_str = " tags"
                    fmt = ".2f"
                else:
                    unit_str = "%"
                    fmt = ".2f"

                th_display = f"Max {delta.fail_threshold:{fmt}}{unit_str}" if delta.fail_threshold is not None else "N/A"
                rows.append({
                    "Metric": k,
                    "Baseline": f"{delta.baseline_value:{fmt}}{unit_str}",
                    "Candidate": f"{delta.candidate_value:{fmt}}{unit_str}",
                    "Delta (Cand - Base)": f"{sign}{delta.delta:{fmt}}{unit_str}",
                    "Status": icon,
                    "Threshold Limit": th_display,
                    "Diagnostic Explanation": delta.reason,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if res.failures:
            st.error("#### ❌ Critical Violations")
            for f in res.failures:
                st.write(f"- {f}")

        if res.warnings:
            st.warning("#### ⚠️ Warning-Level Findings")
            for w in res.warnings:
                st.write(f"- {w}")

        if res.passed_checks:
            with st.expander("Passed Sanity & Robustness Checks", expanded=False):
                for p in res.passed_checks:
                    st.write(f"- {p}")

        if res.condition_compatibility:
            with st.expander("Evaluation Condition Audit", expanded=False):
                audit_rows = []
                for cond, details in res.condition_compatibility.items():
                    if isinstance(details, dict):
                        audit_rows.append({
                            "Condition": cond,
                            "Baseline": str(details.get("baseline")),
                            "Candidate": str(details.get("candidate")),
                            "Compatible": "✅ YES" if details.get("matched") else "❌ NO",
                        })
                st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

        if res.semantic_consistency_assessment:
            sc_info = res.semantic_consistency_assessment
            st.markdown("#### 🏷️ Semantic Consistency Assessment (RAM++)")
            sc_stat = sc_info.get("status", "UNKNOWN")
            if sc_stat == "PASS":
                st.success(f"**PASS ✅**: {sc_info.get('summary')}")
            elif sc_stat == "WARN":
                st.warning(f"**WARN ⚠️**: {sc_info.get('summary')}")
            else:
                st.error(f"**FAIL ❌**: {sc_info.get('summary')}")

            cand_score = sc_info.get("candidate_consistency")
            if cand_score is not None:
                c_det = sc_info.get("candidate_details", {})
                st.write(
                    f"- Candidate Consistency: **{cand_score:.1f}%** (Min Threshold: {sc_info.get('min_threshold', 60.0)}%)\n"
                    f"- Tags Retained: **{len(c_det.get('overlap_tags', []))}** · "
                    f"Lost: **{c_det.get('tags_lost_count', 0)}** · "
                    f"New: **{c_det.get('new_tags_count', 0)}**"
                )

        # STAGE 6: EXPORT REPORT
        st.markdown("### Stage 6 – Export Evaluation Artifacts")
        json_report = export_gate_report_json(res)
        md_report = export_gate_report_markdown(res)

        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

        with col_dl1:
            st.download_button(
                "📥 Download JSON Gate Report",
                data=json_report,
                file_name="model_security_regression_report.json",
                mime="application/json",
                key="ram_dl_gate_json",
            )
        with col_dl2:
            st.download_button(
                "📄 Download Markdown Summary",
                data=md_report,
                file_name="model_security_regression_summary.md",
                mime="text/markdown",
                key="ram_dl_gate_md",
            )
        with col_dl3:
            if base_snap_curr:
                st.download_button(
                    "📥 Baseline Snapshot (JSON)",
                    data=json.dumps(base_snap_curr.to_dict(), indent=2),
                    file_name="baseline_evaluation_snapshot.json",
                    mime="application/json",
                    key="ram_dl_base_snap_json",
                )
        with col_dl4:
            if cand_snap_curr:
                st.download_button(
                    "📥 Candidate Snapshot (JSON)",
                    data=json.dumps(cand_snap_curr.to_dict(), indent=2),
                    file_name="candidate_evaluation_snapshot.json",
                    mime="application/json",
                    key="ram_dl_cand_snap_json",
                )


def render_regression_gate_workspace():
    st.header("⚖️ Model Security Regression Gate")
    st.write(
        "Evaluate and compare AI model checkpoints against standardized benchmarks, "
        "verify evaluation condition compatibility, detect performance and adversarial robustness regressions, "
        "and issue authoritative PASS, WARN, or FAIL verdicts with structured reports."
    )

    model_family = st.radio(
        "Select Model Evaluation Family",
        [
            "Image Classification (CIFAR-10)",
            "Image Tagging (RAM++)",
        ],
        horizontal=True,
        key="reg_gate_model_family",
        help="Switch between single-label CIFAR-10 classification robustness testing and multi-label RAM++ semantic tagging regression testing.",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_family == "Image Classification (CIFAR-10)":
        _render_cifar_regression_gate(device)
    else:
        _render_ram_regression_gate(device)


# -------------------- HEADER / NAVIGATION --------------------
st.title("🛡️ ModelShield")
st.subheader("Adversarial Robustness Evaluation Dashboard")
st.write(
    "Explore model predictions, generate adversarial examples with FGSM or PGD, "
    "and evaluate robustness on CIFAR-10."
)

page = st.sidebar.radio(
    "Choose a workspace",
    [
        "RAM++ Image Tagging",
        "Test Your Own Image",
        "Create an Adversarial Image",
        "Test CIFAR-10 Dataset",
        "⚖️ Model Security Regression Gate",
    ],
)
st.sidebar.caption("Model: Carmon2019Unlabeled · Dataset: CIFAR-10 · Input size: 32×32")


# -------------------- RAM++ IMAGE TAGGING --------------------
if page == "RAM++ Image Tagging":
    st.header("RAM++ Image Tagging")
    st.write("Upload an image to generate image tags using the RAM++ model.")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="rampp_upload",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Generate Tags", type="primary", key="rampp_generate"):
            try:
                from model.ram_loader import load_ram_model
                from ram import inference

                with st.spinner("Loading RAM++ and generating tags..."):
                    model, preprocess, device = load_ram_model()
                    tensor = preprocess(image).unsqueeze(0).to(device)
                    tags_en, tags_cn = inference.inference_ram(tensor, model)

                st.success(f"Tagging complete (device: {device})")
                st.subheader("Detected Tags")
                st.write(tags_en)

                with st.expander("Chinese tags"):
                    st.write(tags_cn)

            except Exception as e:
                st.error(f"RAM++ could not process this image: {e}")

# -------------------- OPTION 1: TEST OWN IMAGE --------------------
elif page == "Test Your Own Image":
    st.header("1. Test Your Own Image")
    st.write(
        "Upload an image to see the model's predicted CIFAR-10 class and confidence. "
        "The model expects 32×32 RGB images, so your image is resized automatically."
    )
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        key="predict_upload",
    )

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            resized_image, image_tensor = image_tensor_from_pil(original_image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Uploaded image", use_container_width=True)
            with col2:
                st.image(resized_image, caption="Model input (32 × 32)", use_container_width=True)

            if st.button("Test Image", type="primary", key="test_own_image"):
                with st.spinner("Loading model and predicting..."):
                    model, device = get_model()
                    prediction, confidence, probabilities = predict_image(model, image_tensor, device)

                st.success(f"Predicted class: **{CLASS_NAMES[prediction]}**")
                st.metric("Confidence in predicted class", f"{confidence:.2f}%")
                probability_table = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Probability (%)": (probabilities.numpy() * 100).round(2),
                }).sort_values("Probability (%)", ascending=False)
                st.subheader("Class probabilities")
                st.bar_chart(probability_table.set_index("Class"))
                st.caption(
                    "Important: this model was trained on CIFAR-10's ten classes. "
                    "Predictions for unrelated real-world images may be unreliable."
                )
        except Exception as exc:
            st.error(f"Could not read or test this image: {exc}")
    else:
        st.info("Upload an image to begin.")


# -------------------- OPTION 2: CREATE ADVERSARIAL IMAGE --------------------
elif page == "Create an Adversarial Image":
    st.header("2. Create an Adversarial Image")
    st.write(
        "Upload an image, choose FGSM or PGD, and generate a small perturbation intended "
        "to change the model's prediction."
    )
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        key="attack_upload",
    )

    attack_type = st.selectbox("Attack method", ["FGSM", "PGD"], key="attack_method")
    epsilon_pixels = st.slider(
        "Perturbation budget (epsilon, pixel levels / 255)",
        min_value=1, max_value=16, value=8, key="custom_epsilon_slider",
    )
    epsilon = epsilon_pixels / 255.0
    pgd_steps = 10
    if attack_type == "PGD":
        pgd_steps = st.slider(
            "PGD iterations", min_value=1, max_value=20, value=10, key="custom_pgd_steps_slider"
        )

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            resized_image, image_tensor_cpu = image_tensor_from_pil(original_image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Uploaded image", use_container_width=True)
            with col2:
                st.image(resized_image, caption="Resized model input (32 × 32)", use_container_width=True)

            if st.button("Generate Adversarial Image", type="primary", key="make_adversarial"):
                with st.spinner("Generating attack... this may take a little while on CPU."):
                    model, device = get_model()
                    image_tensor = image_tensor_cpu.to(device)
                    model.eval()
                    original_prediction, original_confidence, _ = predict_image(model, image_tensor, device)
                    # The predicted class is used as the reference label because a custom
                    # upload usually has no verified CIFAR-10 ground-truth label.
                    attack_labels = torch.tensor([original_prediction], device=device, dtype=torch.long)

                    if attack_type == "FGSM":
                        adversarial_tensor = fgsm_attack(
                            model, image_tensor, attack_labels, epsilon=epsilon
                        )
                    else:
                        adversarial_tensor = pgd_attack(
                            model,
                            image_tensor,
                            attack_labels,
                            epsilon=epsilon,
                            alpha=min(2 / 255, epsilon),
                            steps=pgd_steps,
                            random_start=True,
                        )

                    adversarial_prediction, adversarial_confidence, _ = predict_image(
                        model, adversarial_tensor, device
                    )
                    max_perturbation = float((adversarial_tensor - image_tensor).abs().max().item())
                    pixel_min = float(adversarial_tensor.min().item())
                    pixel_max = float(adversarial_tensor.max().item())
                    budget_passed = max_perturbation <= epsilon + 1e-6
                    range_passed = pixel_min >= -1e-6 and pixel_max <= 1.0 + 1e-6

                st.subheader("Attack Results")
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    st.image(resized_image, caption="Original model input", use_container_width=True)
                    st.write(f"**Prediction:** {CLASS_NAMES[original_prediction]}")
                    st.metric("Original confidence", f"{original_confidence:.2f}%")
                with result_col2:
                    st.image(tensor_to_display(adversarial_tensor), caption=f"{attack_type} adversarial image", use_container_width=True)
                    st.write(f"**Prediction:** {CLASS_NAMES[adversarial_prediction]}")
                    st.metric("Adversarial confidence", f"{adversarial_confidence:.2f}%")

                if original_prediction != adversarial_prediction:
                    st.warning("The model's predicted class changed after the attack.")
                else:
                    st.info("The model retained its original predicted class. An attack can still change confidence without changing the class.")

                st.subheader("Perturbation Validation")
                m1, m2, m3 = st.columns(3)
                m1.metric("Maximum perturbation", f"{max_perturbation:.6f}")
                m2.metric("Allowed epsilon", f"{epsilon:.6f}")
                m3.metric("Pixel range", f"[{pixel_min:.4f}, {pixel_max:.4f}]")
                if budget_passed:
                    st.success("PASS: perturbation is within the configured budget.")
                else:
                    st.error("FAIL: perturbation exceeds the configured budget.")
                if range_passed:
                    st.success("PASS: adversarial pixels remain within [0, 1].")
                else:
                    st.error("FAIL: adversarial pixels are outside [0, 1].")
                st.caption(
                    "For custom images, the original model prediction is used as the attack reference. "
                    "This tests whether the prediction changes; it does not establish the true label."
                )
        except Exception as exc:
            st.error(f"Could not generate the adversarial image: {exc}")
    else:
        st.info("Upload an image to configure and run an attack.")


# -------------------- OPTION 3: TEST CIFAR-10 DATASET --------------------
elif page == "Test CIFAR-10 Dataset":
    st.header("3. Test the CIFAR-10 Dataset")
    st.write(
        "Evaluate the pretrained model on a small CIFAR-10 sample, compare clean predictions "
        "with FGSM and PGD attacks, and run ModelShield's trust checks."
    )
    with st.sidebar:
        st.subheader("CIFAR-10 Evaluation Settings")
        dataset_size = st.selectbox("Number of test images", [8, 16, 32], index=2)
        dataset_epsilon_pixels = st.slider(
            "Dataset attack epsilon (pixel levels / 255)",
            min_value=1, max_value=16, value=8, key="dataset_epsilon_slider",
        )
        dataset_pgd_steps = st.slider(
            "Dataset PGD iterations", min_value=1, max_value=20, value=10, key="dataset_pgd_steps_slider"
        )

    if st.button("Run CIFAR-10 Evaluation", type="primary", key="run_cifar_evaluation"):
        dataset_epsilon = dataset_epsilon_pixels / 255.0
        try:
            with st.spinner("Loading CIFAR-10 and evaluating attacks. CPU runs may take a while..."):
                model, device = get_model()
                images, labels = get_cifar_dataset(dataset_size)
                images = images.to(device)
                labels = labels.to(device)
                results = evaluate_attack_ensemble(
                    model,
                    images,
                    labels,
                    epsilon=dataset_epsilon,
                    alpha=min(2 / 255, dataset_epsilon),
                    pgd_steps=dataset_pgd_steps,
                )
                with torch.no_grad():
                    clean_outputs = model(images)
                checks = {
                    "Perturbation budget (FGSM)": check_perturbation_budget(
                        images, results["fgsm_images"], dataset_epsilon
                    ),
                    "Perturbation budget (PGD)": check_perturbation_budget(
                        images, results["pgd_images"], dataset_epsilon
                    ),
                    "Pixel range (FGSM)": check_pixel_range(results["fgsm_images"]),
                    "Pixel range (PGD)": check_pixel_range(results["pgd_images"]),
                    "Prediction outputs": check_prediction_outputs(clean_outputs, labels),
                    "Metric consistency": check_metric_consistency(
                        results["clean_accuracy"],
                        results["fgsm_accuracy"],
                        results["worst_case_attack_success_rate"],
                    ),
                }
                trust_flag = generate_trust_flag(checks)

                # Persist in session state to maintain interactive state across widget reruns
                st.session_state["cifar_eval_results"] = results
                st.session_state["cifar_eval_checks"] = checks
                st.session_state["cifar_eval_trust_flag"] = trust_flag
                st.session_state["cifar_eval_images"] = images
                st.session_state["cifar_eval_labels"] = labels
                st.session_state["cifar_eval_epsilon"] = dataset_epsilon
                st.session_state["cifar_eval_pgd_steps"] = dataset_pgd_steps
                # Clear previous diagnostics when new base evaluation runs
                st.session_state["gm_diag_results"] = None

        except Exception as exc:
            st.error(f"CIFAR-10 evaluation failed: {exc}")

    # Render base evaluation results if available in session state
    if st.session_state.get("cifar_eval_results") is not None:
        results = st.session_state["cifar_eval_results"]
        checks = st.session_state["cifar_eval_checks"]
        trust_flag = st.session_state["cifar_eval_trust_flag"]
        images = st.session_state["cifar_eval_images"]
        labels = st.session_state["cifar_eval_labels"]
        dataset_epsilon = st.session_state["cifar_eval_epsilon"]
        dataset_pgd_steps = st.session_state["cifar_eval_pgd_steps"]

        st.caption(
            f"Model: Carmon2019Unlabeled · Samples: {results['total_images']} · "
            f"Epsilon: {dataset_epsilon:.6f} · PGD iterations: {dataset_pgd_steps}"
        )
        st.subheader("Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clean accuracy", f"{results['clean_accuracy']:.2f}%")
        c2.metric("FGSM accuracy", f"{results['fgsm_accuracy']:.2f}%")
        c3.metric("PGD accuracy", f"{results['pgd_accuracy']:.2f}%")
        c4.metric("Worst-case attack success", f"{results['worst_case_attack_success_rate']:.2f}%")

        accuracy_df = pd.DataFrame(
            {"Accuracy (%)": [results["clean_accuracy"], results["fgsm_accuracy"], results["pgd_accuracy"]]},
            index=["Clean", "FGSM", "PGD"],
        )
        st.bar_chart(accuracy_df)

        st.subheader("Worst-Case Attack Analysis")
        a, b, c = st.columns(3)
        a.metric("Initially correct images", int(results["initially_correct_count"]))
        b.metric("Fooled by either attack", int(results["worst_case_fooled_count"]))
        c.metric("Worst-case ASR", f"{results['worst_case_attack_success_rate']:.2f}%")

        st.subheader("Trust Flag")
        status = trust_flag["status"]
        summary = trust_flag.get("summary", "Trust checks completed.")
        if status == "PASS":
            st.success(f"PASS: {summary}")
        elif status == "WARNING":
            st.warning(f"WARNING: {summary}")
        else:
            st.error(f"FAIL: {summary}")
        st.write(f"Checks passed: {trust_flag['passed']} / {trust_flag['total']}")
        check_rows = [
            {
                "Check": name,
                "Status": "PASS" if result["passed"] else "FAIL",
                "Details": result["message"],
            }
            for name, result in checks.items()
        ]
        st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)

        st.subheader("⚖️ Model Security Regression Gate Snapshots")
        st.caption("Capture this evaluation run for comparison in the Regression Gate workspace:")
        sb_col1, sb_col2 = st.columns(2)
        with sb_col1:
            if st.button("📌 Set as Regression Baseline", key="btn_save_cifar_baseline"):
                snap = snapshot_from_cifar_session(role="Baseline")
                if snap is not None:
                    st.session_state["gate_baseline_snapshot"] = snap
                    st.session_state["gate_result"] = None
                    st.success("Current evaluation saved as Regression Baseline.")
                else:
                    st.error("Failed to construct snapshot: missing required evaluation results.")
        with sb_col2:
            if st.button("🎯 Set as Regression Candidate", key="btn_save_cifar_candidate"):
                snap = snapshot_from_cifar_session(role="Candidate")
                if snap is not None:
                    st.session_state["gate_candidate_snapshot"] = snap
                    st.session_state["gate_result"] = None
                    st.success("Current evaluation saved as Regression Candidate.")
                else:
                    st.error("Failed to construct snapshot: missing required evaluation results.")

        st.subheader("Compare a CIFAR-10 Image")
        image_index = st.slider(
            "Select a test image", min_value=0, max_value=len(labels) - 1, value=0,
            key="dataset_image_index",
        )
        clean_image = images[image_index:image_index + 1]
        fgsm_image = results["fgsm_images"][image_index:image_index + 1]
        pgd_image = results["pgd_images"][image_index:image_index + 1]
        clean_pred = int(results["clean_predictions"][image_index].item())
        fgsm_pred = int(results["fgsm_predictions"][image_index].item())
        pgd_pred = int(results["pgd_predictions"][image_index].item())
        true_label = int(labels[image_index].item())
        i1, i2, i3 = st.columns(3)
        with i1:
            st.image(tensor_to_display(clean_image), caption="Original", use_container_width=True)
            st.write(f"True label: **{CLASS_NAMES[true_label]}**")
            st.write(f"Prediction: **{CLASS_NAMES[clean_pred]}**")
        with i2:
            st.image(tensor_to_display(fgsm_image), caption="FGSM", use_container_width=True)
            st.write(f"Prediction: **{CLASS_NAMES[fgsm_pred]}**")
            st.write("Correct" if fgsm_pred == true_label else "Misclassified")
        with i3:
            st.image(tensor_to_display(pgd_image), caption="PGD", use_container_width=True)
            st.write(f"Prediction: **{CLASS_NAMES[pgd_pred]}**")
            st.write("Correct" if pgd_pred == true_label else "Misclassified")

        export_df = pd.DataFrame({
            "Image_Index": list(range(len(labels))),
            "True_Label": [CLASS_NAMES[int(x.item())] for x in labels],
            "Clean_Prediction": [CLASS_NAMES[int(x.item())] for x in results["clean_predictions"]],
            "FGSM_Prediction": [CLASS_NAMES[int(x.item())] for x in results["fgsm_predictions"]],
            "PGD_Prediction": [CLASS_NAMES[int(x.item())] for x in results["pgd_predictions"]],
        })
        st.download_button(
            "Download CIFAR-10 Results (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="modelshield_cifar10_results.csv",
            mime="text/csv",
        )

        # -------------------- GRADIENT-MASKING DIAGNOSTICS --------------------
        st.divider()
        st.subheader("🔬 Gradient-Masking Diagnostic Suite")
        st.write(
            "Diagnose whether evaluation gradients are genuine or masked/obfuscated (Athalye et al., ICML 2018). "
            "Evaluates input-gradient health, 1D loss landscape ray monotonicity, PGD restart sensitivity, "
            "attack-strength consistency across steps and budgets, FGSM vs PGD discrepancy, and a zero-order random exploration baseline."
        )

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            diag_sample_cap = st.selectbox(
                "Diagnostic sample cap (CPU optimization)",
                [8, 16, 32],
                index=min(1, [8, 16, 32].index(len(labels)) if len(labels) in [8, 16, 32] else 1),
                key="gm_diag_sample_cap",
                help="Limits the number of clean samples evaluated by the diagnostic runner to ensure responsive execution.",
            )
        with col_cfg2:
            include_random = st.checkbox(
                "Include Zero-Order (Gradient-Free) Random Search check",
                value=True,
                key="gm_include_random_search",
                help="Tests whether purely uniform random noise fools the model more effectively than gradient descent.",
            )

        if st.button("Run Gradient-Masking Diagnostics", type="secondary", key="run_gm_diagnostics"):
            try:
                with st.spinner("Running gradient-masking diagnostics (loss ray, restarts, strength sweep)..."):
                    model, device = get_model()
                    diag_results = analyze_gradient_masking(
                        model=model,
                        images=images,
                        labels=labels,
                        epsilon=dataset_epsilon,
                        alpha=min(2 / 255, dataset_epsilon),
                        pgd_steps=dataset_pgd_steps,
                        pgd_restarts=2,
                        baseline_pgd_asr=results.get("pgd_attack_success_rate"),
                        baseline_fgsm_asr=results.get("fgsm_attack_success_rate"),
                        run_random_search=include_random,
                        max_diagnostic_samples=diag_sample_cap,
                    )
                    st.session_state["gm_diag_results"] = diag_results
            except Exception as exc:
                st.error(f"Gradient-masking diagnostic failed: {exc}")

        # Render diagnostic results if present in session state
        if st.session_state.get("gm_diag_results") is not None:
            gm = st.session_state["gm_diag_results"]
            gm_status = gm.get("status", "INCONCLUSIVE")
            gm_summary = gm.get("summary", "")

            st.write("---")
            if gm_status == "WARNING":
                st.warning(f"⚠️ **STATUS: {gm_status}**\n\n{gm_summary}")
            elif gm_status == "NO_STRONG_SIGNAL":
                st.success(f"✅ **STATUS: {gm_status}**\n\n{gm_summary}")
            else:
                st.info(f"ℹ️ **STATUS: {gm_status}**\n\n{gm_summary}")

            if gm.get("warning_reasons"):
                st.markdown("**Identified Warning Reasons:**")
                for reason in gm["warning_reasons"]:
                    st.markdown(f"- ⚠️ {reason}")

            # Top-level diagnostic metrics
            signals = gm.get("signals", {})
            g_health = signals.get("gradient_health", {})
            l_ray = signals.get("loss_landscape_ray", {})
            p_rest = signals.get("pgd_restart_sensitivity", {})
            fg_pgd = signals.get("fgsm_vs_pgd", {})

            st.markdown("#### Diagnostic Measurements")
            m1, m2, m3, m4 = st.columns(4)
            med_l2 = g_health.get("median_l2_norm")
            m1.metric("Median Input Grad L2", f"{med_l2:.4e}" if med_l2 is not None else "N/A")
            ray_slope = l_ray.get("slope")
            m2.metric("Loss Ray Slope (ΔL)", f"{ray_slope:+.4f}" if ray_slope is not None else "N/A")
            p_spread = p_rest.get("spread_percentage_points")
            m3.metric("PGD Restart Spread", f"{p_spread:.2f}%" if p_spread is not None else "N/A")
            fg_asr = fg_pgd.get("fgsm_success_rate", 0.0)
            pg_asr = fg_pgd.get("pgd_success_rate", 0.0)
            m4.metric("FGSM vs PGD Delta", f"{fg_asr - pg_asr:+.2f}%")

            # Interactive charts
            st.markdown("#### Visual Diagnostics")
            ch_col1, ch_col2 = st.columns(2)

            with ch_col1:
                st.caption("1D Loss Landscape along Gradient Ray: clamp(x + α · sign(∇x L))")
                if l_ray.get("losses") and len(l_ray["losses"]) > 0:
                    ray_df = pd.DataFrame(
                        {"Loss": l_ray["losses"]},
                        index=[f"α = {a:.4f}" for a in l_ray["alphas"]],
                    )
                    st.line_chart(ray_df)
                else:
                    st.info("Loss ray profile not available.")

            with ch_col2:
                st.caption("Attack Strength Trajectory (PGD Step Count)")
                sig_strength = signals.get("attack_strength_consistency", {})
                step_data = sig_strength.get("step_sweep", {})
                if step_data.get("steps") and step_data.get("success_rates"):
                    str_df = pd.DataFrame(
                        {"Attack Success Rate (%)": step_data["success_rates"]},
                        index=[f"{s} steps" for s in step_data["steps"]],
                    )
                    st.line_chart(str_df)
                else:
                    st.info("Strength trajectory not available.")

            # Detailed Expanders for Each Signal
            st.markdown("#### Detailed Signal Breakdown")

            with st.expander("Signal 1: Input Gradient Health & Numerical Sanity", expanded=False):
                st.write(f"**Status:** `{g_health.get('status', 'N/A')}`")
                st.write(f"**Details:** {g_health.get('details', '')}")
                gh_df = pd.DataFrame({
                    "Metric": ["Median L2 Norm", "Median Mean Absolute Magnitude", "Near-Zero Element Fraction", "Non-Finite Values Present"],
                    "Value": [
                        str(g_health.get("median_l2_norm")),
                        str(g_health.get("median_mean_abs")),
                        f"{float(g_health.get('median_near_zero_fraction', 0) or 0) * 100:.2f}%",
                        str(g_health.get("non_finite")),
                    ],
                })
                st.dataframe(gh_df, use_container_width=True, hide_index=True)

            with st.expander("Signal 2: 1D Loss Landscape Directional Monotonicity", expanded=False):
                st.write(f"**Status:** `{l_ray.get('status', 'N/A')}`")
                st.write(f"**Details:** {l_ray.get('details', '')}")
                if l_ray.get("alphas"):
                    lr_df = pd.DataFrame({
                        "Alpha Perturbation": [f"{a:.5f}" for a in l_ray["alphas"]],
                        "Measured Cross-Entropy Loss": [f"{l:.6f}" for l in l_ray["losses"]],
                    })
                    st.dataframe(lr_df, use_container_width=True, hide_index=True)

            with st.expander("Signal 3: PGD Random Restart Sensitivity", expanded=False):
                st.write(f"**Status:** `{p_rest.get('status', 'N/A')}`")
                st.write(f"**Details:** {p_rest.get('details', '')}")
                if p_rest.get("restart_success_rates"):
                    pr_df = pd.DataFrame({
                        "Restart": [f"Restart #{i+1}" for i in range(len(p_rest["restart_success_rates"]))],
                        "Attack Success Rate (%)": [f"{r:.2f}%" for r in p_rest["restart_success_rates"]],
                    })
                    st.dataframe(pr_df, use_container_width=True, hide_index=True)

            with st.expander("Signal 4: Attack Strength Consistency (Steps & Epsilon)", expanded=False):
                st.write(f"**Status:** `{sig_strength.get('status', 'N/A')}`")
                st.write(f"**Details:** {sig_strength.get('details', '')}")
                if sig_strength.get("non_monotonic_drops"):
                    st.warning("Non-monotonic drops detected:")
                    for d in sig_strength["non_monotonic_drops"]:
                        st.write(f"- {d}")
                eps_data = sig_strength.get("epsilon_sweep", {})
                if eps_data.get("epsilons"):
                    eps_df = pd.DataFrame({
                        "Epsilon Budget": [f"{e:.4f}" for e in eps_data["epsilons"]],
                        "Attack Success Rate (%)": [f"{r:.2f}%" for r in eps_data.get("success_rates", [])],
                    })
                    st.dataframe(eps_df, use_container_width=True, hide_index=True)

            with st.expander("Signal 5: FGSM vs Iterative PGD Comparison", expanded=False):
                st.write(f"**Status:** `{fg_pgd.get('status', 'N/A')}`")
                st.write(f"**Details:** {fg_pgd.get('details', '')}")

            with st.expander("Signal 6: Zero-Order (Gradient-Free) Random Search Baseline", expanded=False):
                sig_rand = signals.get("gradient_free_baseline", {})
                st.write(f"**Status:** `{sig_rand.get('status', 'N/A')}`")
                st.write(f"**Details:** {sig_rand.get('details', '')}")
                if sig_rand.get("status") != "NOT_RUN":
                    st.metric("Random Search Success Rate", f"{sig_rand.get('random_search_success_rate', 0.0):.2f}%")

            # Actionable remediation guidance
            if gm.get("actionable_guidance"):
                st.markdown("#### 💡 Actionable Remediation Guidance")
                st.info("\n".join(f"- {g}" for g in gm["actionable_guidance"]))

            # Limitations
            st.caption("· ".join(gm.get("limitations", [])))

            # Export options
            d_json = json.dumps(gm, indent=2)
            c1_exp, c2_exp = st.columns(2)
            c1_exp.download_button(
                "Download Diagnostic Report (JSON)",
                data=d_json.encode("utf-8"),
                file_name="modelshield_gradient_masking_report.json",
                mime="application/json",
            )

            # Flattened summary CSV
            flat_diag = {
                "Overall_Status": [gm_status],
                "Summary": [gm_summary],
                "Warning_Count": [len(gm.get("warning_reasons", []))],
                "Median_Input_Grad_L2": [med_l2],
                "Loss_Ray_Slope": [ray_slope],
                "PGD_Restart_Spread": [p_spread],
                "FGSM_ASR": [fg_asr],
                "PGD_ASR": [pg_asr],
                "Random_Search_ASR": [signals.get("gradient_free_baseline", {}).get("random_search_success_rate", "N/A")],
            }
            c2_exp.download_button(
                "Download Diagnostic Summary (CSV)",
                data=pd.DataFrame(flat_diag).to_csv(index=False).encode("utf-8"),
                file_name="modelshield_gradient_masking_summary.csv",
                mime="text/csv",
            )

    else:
        st.info("Choose the evaluation settings in the sidebar, then click **Run CIFAR-10 Evaluation**.")
elif page == "⚖️ Model Security Regression Gate":
    render_regression_gate_workspace()


# =====================================================
# DEFENSE VERIFICATION LAB
# =====================================================

st.divider()

st.header("🛡️ Defense Verification Lab")

st.write(
    "Compare the original model against an experimental "
    "input-quantization defense using CIFAR-10 images."
)

st.info(
    "This is an experimental defense. Improvements against "
    "the attacks tested here do not guarantee robustness "
    "against adaptive attacks."
)

# Defense settings
defense_levels = st.selectbox(
    "Quantization levels",
    options=[4, 8, 16, 32, 64],
    index=2,
    key="defense_levels",
)

defense_epsilon_pixels = st.slider(
    "Perturbation budget (epsilon)",
    min_value=1,
    max_value=16,
    value=8,
    key="defense_epsilon",
)

defense_pgd_steps = st.slider(
    "PGD iterations",
    min_value=1,
    max_value=10,
    value=3,
    key="defense_pgd_steps",
)

defense_sample_count = st.selectbox(
    "Number of CIFAR-10 images",
    options=[8, 16, 32],
    index=0,
    key="defense_sample_count",
)

run_defense_button = st.button(
    "Run Defense Verification",
    type="primary",
    key="run_defense_verification",
)

if run_defense_button:

    try:

        with st.spinner(
            "Evaluating original and defended models..."
        ):

            model, device = get_model()

            images, labels = load_cifar10(
                n_examples=defense_sample_count
            )

            images = images.to(device)
            labels = labels.to(device)

            defense_results = evaluate_defense(
                model=model,
                images=images,
                labels=labels,
                epsilon=defense_epsilon_pixels / 255,
                pgd_steps=defense_pgd_steps,
                quantization_levels=defense_levels,
            )

            st.session_state[
                "defense_results"
            ] = defense_results

            st.session_state[
                "defense_settings"
            ] = {
                "levels": defense_levels,
                "epsilon": defense_epsilon_pixels / 255,
                "pgd_steps": defense_pgd_steps,
                "sample_count": defense_sample_count,
            }

    except Exception as exc:

        st.error(
            f"Defense verification failed: {exc}"
        )


# Display saved results
if st.session_state.get("defense_results") is not None:

    defense_results = st.session_state["defense_results"]

    original = defense_results["Original"]
    defended = defense_results["Quantization"]

    st.subheader("Performance Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Original Clean Accuracy",
            f"{original['clean_accuracy']:.2f}%",
        )

        st.metric(
            "Defended Clean Accuracy",
            f"{defended['clean_accuracy']:.2f}%",
            delta=(
                f"{defended['clean_accuracy'] - original['clean_accuracy']:.2f} "
                "percentage points"
            ),
        )

    with col2:

        st.metric(
            "Original FGSM Accuracy",
            f"{original['fgsm_accuracy']:.2f}%",
        )

        st.metric(
            "Defended FGSM Accuracy",
            f"{defended['fgsm_accuracy']:.2f}%",
            delta=(
                f"{defended['fgsm_accuracy'] - original['fgsm_accuracy']:.2f} "
                "percentage points"
            ),
        )

    with col3:

        st.metric(
            "Original PGD Accuracy",
            f"{original['pgd_accuracy']:.2f}%",
        )

        st.metric(
            "Defended PGD Accuracy",
            f"{defended['pgd_accuracy']:.2f}%",
            delta=(
                f"{defended['pgd_accuracy'] - original['pgd_accuracy']:.2f} "
                "percentage points"
            ),
        )

    # Comparison chart
    comparison_data = pd.DataFrame(
        {
            "Original Model": [
                original["clean_accuracy"],
                original["fgsm_accuracy"],
                original["pgd_accuracy"],
            ],
            "Quantization Defense": [
                defended["clean_accuracy"],
                defended["fgsm_accuracy"],
                defended["pgd_accuracy"],
            ],
        },
        index=[
            "Clean Accuracy",
            "FGSM Accuracy",
            "PGD Accuracy",
        ],
    )

    st.subheader("Accuracy Comparison")

    st.bar_chart(comparison_data)

    # Attack success rates
    st.subheader("Attack Success Rate")

    attack_data = pd.DataFrame(
        {
            "Original Model": [
                original["fgsm_attack_success_rate"],
                original["pgd_attack_success_rate"],
            ],
            "Quantization Defense": [
                defended["fgsm_attack_success_rate"],
                defended["pgd_attack_success_rate"],
            ],
        },
        index=["FGSM", "PGD"],
    )

    st.bar_chart(attack_data)

    # Explain the result
    st.subheader("Defense Summary")

    clean_change = (
        defended["clean_accuracy"]
        - original["clean_accuracy"]
    )

    pgd_change = (
        defended["pgd_accuracy"]
        - original["pgd_accuracy"]
    )

    st.write(
        f"Clean accuracy change: {clean_change:+.2f} "
        "percentage points."
    )

    st.write(
        f"PGD accuracy change: {pgd_change:+.2f} "
        "percentage points."
    )

    if pgd_change > 0 and clean_change >= -2:

        st.success(
            "The defense improved PGD accuracy in this run "
            "while keeping the clean-accuracy reduction "
            "within 2 percentage points. Further attack "
            "validation is still required."
        )

    elif pgd_change <= 0:

        st.warning(
            "The defense did not improve PGD accuracy "
            "in this run. It may require a different "
            "configuration or defense method."
        )

    else:

        st.warning(
            "The defense improved PGD accuracy but also "
            "reduced clean accuracy by more than "
            "2 percentage points. Review the trade-off."
        )

    st.caption(
        "Results are specific to this model, dataset subset, "
        "configuration, and attack implementation. They are "
        "not a security certification."
    )
