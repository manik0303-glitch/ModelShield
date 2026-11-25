import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from model.load_model import load_model
from model.dataset import TaggedImageDataset
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
from evaluation.masking_simulation import run_masking_simulation, _HardQuantizedPipeline
from evaluation.prediction_utils import predict_multilabel, is_correct_multilabel, is_fooled_multilabel
from evaluation.defense_evaluation import evaluate_defense

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ModelShield · RAM++",
    page_icon="🛡️",
    layout="wide",
)

# -------------------- CYBER-SECURITY UI STYLING --------------------
st.markdown("""
<style>
.ms-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
    margin-bottom: 1rem;
}
.ms-hero-validated {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-left: 5px solid #10B981;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.ms-hero-warning {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.4) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-left: 5px solid #F59E0B;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.ms-metric-box {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    text-align: left;
}
.ms-metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.25rem;
}
.ms-metric-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.2;
}
.ms-metric-sub {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 0.25rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------- MODEL & DATA CACHING --------------------
@st.cache_resource
def get_model():
    model, device, tag_list, class_threshold, transform = load_model()
    model.eval()
    return model, device, tag_list, class_threshold, transform


def image_tensor_from_pil(pil_image, transform):
    rgb = pil_image.convert("RGB")
    tensor = transform(rgb).unsqueeze(0)
    resized_preview = rgb.resize((384, 384))
    return resized_preview, tensor


def tensor_to_display(tensor):
    # Unnormalize [0, 1] approximately for viewing
    img = tensor.detach().cpu()[0].permute(1, 2, 0).clamp(0, 1).numpy()
    return img


# -------------------- HEADER / NAVIGATION --------------------
st.title("🛡️ ModelShield · RAM++")
st.subheader("Adversarial Robustness & Gradient-Masking Detector for Recognize Anything Plus")
st.write(
    "Evaluate multi-label open-vocabulary tagging under adversarial perturbation, "
    "and diagnose gradient-masking artifacts (Athalye et al., ICML 2018)."
)

page = st.sidebar.radio(
    "Choose a workspace",
    [
        "Test Your Own Image",
        "Create an Adversarial Image",
        "Dataset Tagging Robustness",
        "Gradient Masking Lab",
    ],
)
st.sidebar.caption("Detector: RAM++ (Recognize Anything Plus) · Multi-Label · Input: 384×384")


# -------------------- OPTION 1: TEST OWN IMAGE --------------------
if page == "Test Your Own Image":
    st.header("1. Test Your Own Image")
    st.write(
        "Upload an image to inspect RAM++ predicted tags and confidence levels. "
        "Images are transformed to 384×384 standard resolution."
    )
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        key="predict_upload",
    )

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            model, device, tag_list, class_threshold, transform = get_model()
            resized_image, image_tensor = image_tensor_from_pil(original_image, transform)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Uploaded image", use_container_width=True)
            with col2:
                st.image(resized_image, caption="RAM++ input (384 × 384)", use_container_width=True)

            if st.button("Run RAM++ Recognition", type="primary", key="test_own_image"):
                with st.spinner("Executing RAM++ forward pass..."):
                    logits, pred_binary = predict_multilabel(
                        model, image_tensor, threshold=class_threshold, device=device
                    )
                    probs = logits.sigmoid()[0].cpu().numpy()
                    pred_tags_idx = (pred_binary[0] == 1).nonzero(as_tuple=True)[0].cpu().numpy()

                st.subheader("Recognized Tags")
                if len(pred_tags_idx) > 0:
                    detected_tags = [tag_list[i] for i in pred_tags_idx]
                    detected_confs = [probs[i] * 100 for i in pred_tags_idx]
                    tag_df = pd.DataFrame({
                        "Tag": detected_tags,
                        "Confidence (%)": [round(c, 2) for c in detected_confs],
                    }).sort_values("Confidence (%)", ascending=False)
                    st.dataframe(tag_df, use_container_width=True, hide_index=True)
                    st.bar_chart(tag_df.set_index("Tag"))
                else:
                    st.info("No tags crossed the confidence threshold for this image.")
        except Exception as exc:
            st.error(f"Error evaluating image: {exc}")
    else:
        st.info("Upload an image to begin.")


# -------------------- OPTION 2: CREATE ADVERSARIAL IMAGE --------------------
elif page == "Create an Adversarial Image":
    st.header("2. Create an Adversarial Image (Tag-Flipping Attack)")
    st.write(
        "Generate multi-label adversarial perturbations against RAM++ using FGSM or PGD "
        "guided by binary cross-entropy with logits."
    )
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="attack_upload"
    )
    attack_type = st.selectbox("Attack method", ["FGSM", "PGD"], key="attack_method")
    epsilon_pixels = st.slider(
        "Perturbation budget (epsilon, pixel levels / 255)",
        min_value=1, max_value=16, value=4, key="custom_epsilon_slider",
    )
    epsilon = epsilon_pixels / 255.0
    pgd_steps = 10
    if attack_type == "PGD":
        pgd_steps = st.slider("PGD iterations", min_value=1, max_value=20, value=10)

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            model, device, tag_list, class_threshold, transform = get_model()
            resized_image, image_tensor_cpu = image_tensor_from_pil(original_image, transform)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Uploaded image", use_container_width=True)
            with col2:
                st.image(resized_image, caption="RAM++ input (384 × 384)", use_container_width=True)

            if st.button("Generate Adversarial Image", type="primary"):
                with st.spinner("Generating adversarial perturbation..."):
                    image_tensor = image_tensor_cpu.to(device)
                    # Clean predictions as target reference
                    logits_clean, pred_clean_bin = predict_multilabel(
                        model, image_tensor, threshold=class_threshold, device=device
                    )
                    attack_labels = pred_clean_bin.float()

                    if attack_type == "FGSM":
                        adv_tensor = fgsm_attack(
                            model, image_tensor, attack_labels, epsilon=epsilon, multi_label=True
                        )
                    else:
                        adv_tensor = pgd_attack(
                            model, image_tensor, attack_labels, epsilon=epsilon,
                            alpha=min(2 / 255, epsilon), steps=pgd_steps, random_start=True, multi_label=True
                        )

                    logits_adv, pred_adv_bin = predict_multilabel(
                        model, adv_tensor, threshold=class_threshold, device=device
                    )

                clean_tags = [tag_list[i] for i in (pred_clean_bin[0] == 1).nonzero(as_tuple=True)[0].cpu().numpy()]
                adv_tags = [tag_list[i] for i in (pred_adv_bin[0] == 1).nonzero(as_tuple=True)[0].cpu().numpy()]

                dropped_tags = set(clean_tags) - set(adv_tags)
                added_tags = set(adv_tags) - set(clean_tags)

                st.subheader("Attack Outcomes")
                r1, r2 = st.columns(2)
                with r1:
                    st.image(resized_image, caption="Original Input", use_container_width=True)
                    st.write(f"**Clean Tags:** {', '.join(clean_tags) if clean_tags else 'None'}")
                with r2:
                    st.image(tensor_to_display(adv_tensor), caption=f"{attack_type} Adversarial Image", use_container_width=True)
                    st.write(f"**Adversarial Tags:** {', '.join(adv_tags) if adv_tags else 'None'}")

                if dropped_tags:
                    st.warning(f"🔻 **Tags Suppressed by Attack:** {', '.join(dropped_tags)}")
                if added_tags:
                    st.error(f"🔺 **False Tags Injected by Attack:** {', '.join(added_tags)}")
                if not dropped_tags and not added_tags:
                    st.info("Tag set remained unchanged under this perturbation budget.")
        except Exception as exc:
            st.error(f"Adversarial generation failed: {exc}")


# -------------------- OPTION 3: DATASET TAGGING ROBUSTNESS --------------------
elif page == "Dataset Tagging Robustness":
    st.header("3. Dataset Tagging Robustness & Gradient Masking Diagnostics")
    st.write(
        "Evaluate RAM++ across multi-label test images, compare clean tagging accuracy "
        "with FGSM/PGD attacks, and execute the 6-signal gradient masking diagnostic suite."
    )

    model, device, tag_list, class_threshold, transform = get_model()

    dataset_epsilon_pixels = st.slider(
        "Attack epsilon (pixel levels / 255)", min_value=1, max_value=16, value=4, key="ds_eps"
    )
    dataset_epsilon = dataset_epsilon_pixels / 255.0
    dataset_pgd_steps = st.slider("PGD steps", min_value=1, max_value=20, value=10, key="ds_pgd")

    if st.button("Run Multi-Label Evaluation & Diagnostics", type="primary"):
        with st.spinner("Evaluating multi-label batch and computing gradient masking suite..."):
            # Synthetic multi-label test batch
            torch.manual_seed(123)
            test_images = torch.rand(8, 3, 384, 384).to(device)
            with torch.no_grad():
                test_labels = (model(test_images).sigmoid() > class_threshold.to(device)).float()

            results = evaluate_attack_ensemble(
                model=model,
                images=test_images,
                labels=test_labels,
                epsilon=dataset_epsilon,
                alpha=min(1 / 255, dataset_epsilon),
                pgd_steps=dataset_pgd_steps,
                multi_label=True,
                threshold=class_threshold,
            )

            diag_res = analyze_gradient_masking(
                model=model,
                images=test_images,
                labels=test_labels,
                epsilon=dataset_epsilon,
                alpha=min(1 / 255, dataset_epsilon),
                pgd_steps=dataset_pgd_steps,
                pgd_restarts=2,
                baseline_pgd_asr=results.get("pgd_attack_success_rate"),
                baseline_fgsm_asr=results.get("fgsm_attack_success_rate"),
                run_random_search=True,
                multi_label=True,
                threshold=class_threshold,
            )

            st.session_state["ram_eval_results"] = results
            st.session_state["ram_diag_results"] = diag_res

    if st.session_state.get("ram_eval_results") is not None:
        res = st.session_state["ram_eval_results"]
        diag = st.session_state["ram_diag_results"]

        st.subheader("Multi-Label Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clean Exact Accuracy", f"{res['clean_accuracy']:.2f}%")
        c2.metric("FGSM Exact Accuracy", f"{res['fgsm_accuracy']:.2f}%")
        c3.metric("PGD Exact Accuracy", f"{res['pgd_accuracy']:.2f}%")
        c4.metric("Worst-Case ASR", f"{res['worst_case_attack_success_rate']:.2f}%")

        st.divider()
        st.subheader("🔬 Gradient-Masking Diagnostic Verdict for RAM++")
        status = diag.get("status", "INCONCLUSIVE")
        summary = diag.get("summary", "")

        if status == "WARNING":
            st.warning(f"⚠️ **STATUS: {status}**\n\n{summary}")
        elif status == "NO_STRONG_SIGNAL":
            st.success(f"✅ **STATUS: {status}**\n\n{summary}")
        else:
            st.info(f"ℹ️ **STATUS: {status}**\n\n{summary}")

        if diag.get("warning_reasons"):
            st.markdown("**Identified Warning Reasons:**")
            for r in diag["warning_reasons"]:
                st.markdown(f"- ⚠️ {r}")

        signals = diag.get("signals", {})
        st.subheader("Diagnostic Measurements")
        m1, m2, m3, m4 = st.columns(4)
        med_l2 = signals.get("gradient_health", {}).get("median_l2_norm")
        m1.metric("Median Input Grad L2", f"{med_l2:.4e}" if med_l2 is not None else "N/A")
        ray_slope = signals.get("loss_landscape_ray", {}).get("slope")
        m2.metric("Loss Ray Slope (ΔL)", f"{ray_slope:+.4f}" if ray_slope is not None else "N/A")
        p_spread = signals.get("pgd_restart_sensitivity", {}).get("spread_percentage_points")
        m3.metric("PGD Restart Spread", f"{p_spread:.2f}%" if p_spread is not None else "N/A")
        fgsm_pgd_delta = signals.get("fgsm_vs_pgd", {}).get("delta")
        m4.metric("FGSM vs PGD Delta", f"{fgsm_pgd_delta:+.2f}%" if fgsm_pgd_delta is not None else "N/A")

        # ----------------- VISUAL DIAGNOSTICS CHARTS -----------------
        st.subheader("Visual Diagnostics")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            ray_info = signals.get("loss_landscape_ray", {})
            alphas = ray_info.get("alphas", [])
            losses = ray_info.get("losses", [])
            if alphas and losses:
                fig_ray = go.Figure()
                fig_ray.add_trace(go.Scatter(
                    x=[f"α = {a:.4f}" for a in alphas],
                    y=losses,
                    mode="lines+markers",
                    line=dict(color="#38BDF8", width=3),
                    marker=dict(size=8, color="#0284C7"),
                    name="Loss along ray"
                ))
                fig_ray.update_layout(
                    title="1D Loss Landscape along Gradient Ray: clamp(x + α · sign(∇_x L))",
                    xaxis_title="Step Size α",
                    yaxis_title="Loss",
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_ray, use_container_width=True)
            else:
                st.info("Loss ray data not available.")

        with vcol2:
            strength_info = signals.get("attack_strength_consistency", {})
            steps_list = strength_info.get("steps_evaluated", [])
            asrs_list = strength_info.get("asrs", [])
            if steps_list and asrs_list:
                fig_str = go.Figure()
                fig_str.add_trace(go.Scatter(
                    x=[f"{s} steps" for s in steps_list],
                    y=asrs_list,
                    mode="lines+markers",
                    line=dict(color="#60A5FA", width=3),
                    marker=dict(size=8, color="#2563EB"),
                    name="PGD ASR"
                ))
                fig_str.update_layout(
                    title="Attack Strength Trajectory (PGD Step Count)",
                    xaxis_title="PGD Iterations",
                    yaxis_title="Attack Success Rate (%)",
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_str, use_container_width=True)
            else:
                st.info("Attack trajectory data not available.")

        # ----------------- DETAILED SIGNAL BREAKDOWN ACCORDION -----------------
        st.subheader("Detailed Signal Breakdown")
        for sig_idx, (sig_key, sig_title) in enumerate([
            ("gradient_health", "Signal 1: Input Gradient Health & Numerical Sanity"),
            ("loss_landscape_ray", "Signal 2: 1D Loss Landscape Directional Monotonicity"),
            ("pgd_restart_sensitivity", "Signal 3: PGD Random Restart Sensitivity"),
            ("attack_strength_consistency", "Signal 4: Attack Strength Consistency (Steps & Epsilon)"),
            ("fgsm_vs_pgd", "Signal 5: FGSM vs Iterative PGD Comparison"),
            ("gradient_free_baseline", "Signal 6: Zero-Order (Gradient-Free) Random Search Baseline"),
        ], start=1):
            s_data = signals.get(sig_key, {})
            s_status = s_data.get("status", "N/A")
            badge = "✅" if s_status == "OK" else ("⚠️" if s_status == "SUSPICIOUS" else "ℹ️")
            with st.expander(f"{badge} {sig_title} — [{s_status}]"):
                st.write(f"**Status:** {s_status}")
                if "details" in s_data:
                    st.write(f"**Details:** {s_data['details']}")
                st.json(s_data)

        st.caption(
            "This is a heuristic diagnostic and does not prove the presence or absence of gradient masking. "
            "Results apply strictly to the evaluated sample batch, threat model (Linf), and attack configurations."
        )

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "Download Diagnostic Report (JSON)",
                data=json.dumps(diag, indent=2),
                file_name="ram_plus_gradient_masking_report.json",
                mime="application/json",
            )
        with dcol2:
            summary_df = pd.DataFrame([{
                "Status": diag.get("status"),
                "Clean Accuracy": f"{res.get('clean_accuracy', 0):.2f}%",
                "FGSM Accuracy": f"{res.get('fgsm_accuracy', 0):.2f}%",
                "PGD Accuracy": f"{res.get('pgd_accuracy', 0):.2f}%",
                "Worst-Case ASR": f"{res.get('worst_case_attack_success_rate', 0):.2f}%",
            }])
            st.download_button(
                "Download Diagnostic Summary (CSV)",
                data=summary_df.to_csv(index=False),
                file_name="ram_plus_diagnostic_summary.csv",
                mime="text/csv",
            )


# -------------------- OPTION 4: GRADIENT MASKING LAB --------------------
elif page == "Gradient Masking Lab":
    st.header("4. 🔬 Gradient Masking Lab (Controlled Simulation)")
    st.write(
        "Evaluate the controlled input-quantization pipeline with RAM++ multi-label tagging, "
        "comparing standard PGD against BPDA-assisted PGD."
    )

    st.subheader("Input Test Images")
    st.write("Upload up to 4 custom images to audit, or leave empty to automatically load the 4 project demo photos (`demo1.jpg` – `demo4.jpg`).")
    sim_uploaded_files = st.file_uploader(
        "Upload images for gradient-masking audit (up to 4)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="sim_image_upload",
    )

    with st.sidebar:
        st.subheader("Simulation Options")
        model_mode = st.radio(
            "Evaluation Model Mode",
            ["Full RAM++ Swin-Large Backbone (3.0 GB)", "Fast Diagnostic Simulation (Offline Mock)"],
            index=0,
            help="Full RAM++ Swin-Large evaluates the real 4,585 open-vocabulary tag list from the pretrained 3.01 GB model weights.",
        )
        sim_levels = st.selectbox("Quantization levels", [4, 8, 16, 32], index=1)
        sim_eps_pixels = st.slider("Attack epsilon / 255", min_value=1, max_value=16, value=8)
        sim_steps = st.slider("PGD steps", min_value=2, max_value=10, value=3)

    if st.button("Run RAM++ Simulation", type="primary"):
        status_placeholder = st.empty()
        with st.spinner("Processing test images and running controlled simulation..."):
            if "Fast Diagnostic" in model_mode:
                from model.load_model import load_model
                model, device, tag_list, class_threshold, transform = load_model(checkpoint_path="")
            else:
                model, device, tag_list, class_threshold, transform = get_model()

            from torchvision import transforms
            to_tensor_384 = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
            ])

            # Prepare test image batch (from upload or fallback demo photos)
            batch_tensors = []
            preview_pils = []

            if sim_uploaded_files and len(sim_uploaded_files) > 0:
                for uf in sim_uploaded_files[:4]:
                    pil_img = Image.open(uf).convert("RGB")
                    preview_pils.append(pil_img)
                    batch_tensors.append(to_tensor_384(pil_img))
            else:
                demo_paths = [
                    Path("C:/projects/main el sem 3/RAM++/images/demo/demo1.jpg"),
                    Path("C:/projects/main el sem 3/RAM++/images/demo/demo2.jpg"),
                    Path("C:/projects/main el sem 3/RAM++/images/demo/demo3.jpg"),
                    Path("C:/projects/main el sem 3/RAM++/images/demo/demo4.jpg"),
                ]
                for dp in demo_paths:
                    if dp.is_file():
                        pil_img = Image.open(dp).convert("RGB")
                        preview_pils.append(pil_img)
                        batch_tensors.append(to_tensor_384(pil_img))

            if len(batch_tensors) == 0:
                # Synthetic fallback if demo files missing
                batch_imgs = torch.rand(4, 3, 384, 384).to(device)
            else:
                batch_imgs = torch.stack(batch_tensors).to(device)

            sim_sample_count = batch_imgs.shape[0]
            sim_pipeline = _HardQuantizedPipeline(model, levels=sim_levels)
            with torch.no_grad():
                thresh = class_threshold.to(device) if isinstance(class_threshold, torch.Tensor) else class_threshold
                test_labels = (sim_pipeline(batch_imgs).sigmoid() > thresh).float()

            sim_out, tb = run_masking_simulation(
                model=model,
                images=batch_imgs,
                labels=test_labels,
                quantization_levels=sim_levels,
                epsilon=sim_eps_pixels / 255.0,
                alpha=min(1 / 255, sim_eps_pixels / 255.0),
                pgd_steps=sim_steps,
                sample_cap=sim_sample_count,
                min_eligible=1,
                multi_label=True,
                threshold=class_threshold,
                return_tensors=True,
            )
            
            # Retrieve attack tensors computed directly during simulation (avoids duplicate PGD/BPDA loops!)
            pgd_adv_imgs = tb.get("pgd_images", batch_imgs.detach().cpu())
            bpda_adv_imgs = tb.get("bpda_images", batch_imgs.detach().cpu())

            # Cache images and attack outputs for interactive visual inspection
            st.session_state["ram_sim_out"] = sim_out
            st.session_state["sim_model_mode_used"] = model_mode
            st.session_state["sim_tag_list"] = tag_list
            st.session_state["sim_raw_images"] = batch_imgs.detach().cpu()
            st.session_state["sim_pgd_images"] = pgd_adv_imgs
            st.session_state["sim_bpda_images"] = bpda_adv_imgs

    if st.session_state.get("ram_sim_out") is not None:
        so = st.session_state["ram_sim_out"]
        val = so.get("simulation_validation", {})
        det = so.get("detector_result", {})
        meas = so.get("measurements", {})
        atks = meas.get("attacks", {})

        p1, p2 = st.columns(2)
        with p1:
            st.subheader("Simulation Validation")
            st.info(f"Status: **{val.get('status')}** — {val.get('summary')}")
        with p2:
            st.subheader("Detector Verdict")
            st.warning(f"Status: **{det.get('status')}** — {det.get('summary')}")

        st.subheader("Attack Success Rates (Standard vs BPDA)")
        b1, b2, b3 = st.columns(3)
        b1.metric("Standard FGSM ASR", f"{atks.get('fgsm', {}).get('asr')}%")
        b2.metric("Standard PGD ASR", f"{atks.get('pgd', {}).get('asr')}%")
        b3.metric("BPDA PGD ASR", f"{atks.get('bpda_pgd', {}).get('asr')}%")

        # ----------------- SIMULATION VISUAL CHARTS -----------------
        st.subheader("Visual Simulation Analytics & Loss Landscape")
        scol1, scol2 = st.columns(2)

        with scol1:
            # Bar chart comparing attacks
            attacks_names = ["Standard FGSM", "Standard PGD", "BPDA-Assisted PGD"]
            asr_values = [
                float(atks.get('fgsm', {}).get('asr') or 0),
                float(atks.get('pgd', {}).get('asr') or 0),
                float(atks.get('bpda_pgd', {}).get('asr') or 0),
            ]
            fig_attacks = go.Figure(data=[
                go.Bar(
                    x=attacks_names,
                    y=asr_values,
                    marker_color=["#38BDF8", "#F59E0B", "#EF4444"],
                    text=[f"{v:.1f}%" for v in asr_values],
                    textposition="auto",
                )
            ])
            fig_attacks.update_layout(
                title="ASR: Gradient-Masked PGD vs BPDA Bypass",
                yaxis_title="Attack Success Rate (%)",
                template="plotly_dark",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_attacks, use_container_width=True)

        with scol2:
            # 1D Loss Landscape Ray (Gold standard for Gradient Masking detection)
            sim_signals = det.get("signals", {})
            ray_data = sim_signals.get("loss_landscape_ray", {})
            alphas = ray_data.get("alphas", [])
            masked_losses = ray_data.get("losses", [])
            bpda_losses = ray_data.get("bpda_losses", [])

            fig_ray = go.Figure()
            if alphas and masked_losses:
                # Alpha formatted as epsilon steps
                alpha_labels = [f"{a * 255:.1f}/255" for a in alphas]
                
                # Masked Loss Line (Flat or erratic)
                fig_ray.add_trace(go.Scatter(
                    x=alpha_labels,
                    y=masked_losses,
                    mode="lines+markers",
                    name="Masked Defense (Ordinary Grad)",
                    line=dict(color="#EF4444", width=3, dash="dash"),
                    marker=dict(size=8, symbol="circle"),
                ))
                
                # BPDA Loss Line (Climbing curve if available)
                if bpda_losses and len(bpda_losses) == len(alphas):
                    fig_ray.add_trace(go.Scatter(
                        x=alpha_labels,
                        y=bpda_losses,
                        mode="lines+markers",
                        name="BPDA Adaptive Grad (Bypass)",
                        line=dict(color="#10B981", width=3),
                        marker=dict(size=8, symbol="diamond"),
                    ))

                fig_ray.update_layout(
                    title="1D Loss Landscape Ray: Masked (Flat) vs Adaptive",
                    xaxis_title="Step Distance along Gradient Direction (α)",
                    yaxis_title="Cross-Entropy / BCE Loss",
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_ray, use_container_width=True)
            else:
                # Fallback to gradient norm comparison chart
                g_checks = meas.get("gradient_checks", {})
                ord_norm = g_checks.get("ordinary_gradient_norm") or 0.0
                bpda_norm = g_checks.get("bpda_gradient_norm") or 0.0
                
                fig_grad = go.Figure(data=[
                    go.Bar(
                        x=["Masked (Quantized) Grad L2", "BPDA Straight-Through Grad L2"],
                        y=[ord_norm, bpda_norm],
                        marker_color=["#64748B", "#10B981"],
                        text=[f"{ord_norm:.2e}", f"{bpda_norm:.2e}"],
                        textposition="auto",
                    )
                ])
                fig_grad.update_layout(
                    title="Input Gradient Magnitude Comparison (L2 Norm)",
                    yaxis_title="Gradient L2 Norm",
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_grad, use_container_width=True)

        # ----------------- LAYER 4: AUTOMATED TRUST & INTEGRITY ATTESTATION -----------------
        st.subheader("🛡️ Automated Mathematical Integrity & Sanity Attestation")
        attestation = meas.get("trust_attestation", {})
        attest_status = attestation.get("status", "NOT_AVAILABLE")
        if attest_status == "ATTESTED":
            st.success("✅ **MATHEMATICAL INTEGRITY ATTESTED:** All adversary perturbation bounds, valid pixel domain constraints [0, 1], and tensor sanity invariants strictly verified.")
        else:
            st.warning(f"⚠️ Attestation status: {attest_status}")

        attest_cols = st.columns(3)
        checks = attestation.get("checks", [])
        for i, chk in enumerate(checks):
            with attest_cols[i % 3]:
                st.markdown(f"**{'✅' if chk['passed'] else '❌'} {chk['name']}**")
                st.caption(chk['description'])

        # ----------------- LAYER 1: PER-SAMPLE VISUAL TAG AUDIT -----------------
        st.subheader("🔍 Sample-by-Sample Tag & Adversarial Inspection")
        st.write(
            "Direct empirical verification: Inspect exactly which tags were preserved by the defense vs. "
            "destroyed by the adaptive BPDA attack for each sample."
        )

        sample_audits = meas.get("sample_audits", [])
        sim_tags = st.session_state.get("sim_tag_list", [f"tag_{k}" for k in range(20)])
        raw_images = st.session_state.get("sim_raw_images")
        pgd_images = st.session_state.get("sim_pgd_images")
        bpda_images = st.session_state.get("sim_bpda_images")

        if sample_audits:
            sample_tabs = st.tabs([f"Sample #{s['sample_index'] + 1}" for s in sample_audits])
            for s_idx, tab in enumerate(sample_tabs):
                with tab:
                    sa = sample_audits[s_idx]
                    t_labels = sa.get("target_labels_bin", [])
                    c_preds = sa.get("clean_preds_bin", [])
                    p_preds = sa.get("pgd_preds_bin", [])
                    b_preds = sa.get("bpda_preds_bin", [])

                    clean_tags_list = [sim_tags[k] for k, v in enumerate(c_preds) if v == 1]
                    pgd_tags_list = [sim_tags[k] for k, v in enumerate(p_preds) if v == 1]
                    bpda_tags_list = [sim_tags[k] for k, v in enumerate(b_preds) if v == 1]

                    # Status badges
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.info(f"**Clean Baseline:** {'Matches Target' if sa.get('clean_correct') else 'Partial Target'}")
                    with col_b2:
                        pgd_fooled = sa.get("pgd_fooled", False)
                        if pgd_fooled:
                            st.warning("⚠️ **Standard PGD:** Tags Altered")
                        else:
                            st.success("🛡️ **Standard PGD:** Defense Held (0 Tags Flipped)")
                    with col_b3:
                        bpda_fooled = sa.get("bpda_fooled", False)
                        if bpda_fooled:
                            st.error("🚨 **BPDA Adaptive:** Defense Bypassed (Tags Wiped)")
                        else:
                            st.success("🛡️ **BPDA Adaptive:** Robust")

                    # 3-Way Side-by-Side Images & Perturbation Noise Heatmap
                    if raw_images is not None and s_idx < raw_images.shape[0]:
                        st.markdown("**Visual Adversarial Comparison (384 × 384):**")
                        im_c1, im_c2, im_c3, im_c4 = st.columns(4)
                        
                        clean_np = raw_images[s_idx].permute(1, 2, 0).clamp(0, 1).numpy()
                        with im_c1:
                            st.image(clean_np, caption="1. Original Clean Image", width=180)

                        if pgd_images is not None and s_idx < pgd_images.shape[0]:
                            pgd_np = pgd_images[s_idx].permute(1, 2, 0).clamp(0, 1).numpy()
                            with im_c2:
                                st.image(pgd_np, caption="2. Standard PGD (Masked)", width=180)

                        if bpda_images is not None and s_idx < bpda_images.shape[0]:
                            bpda_np = bpda_images[s_idx].permute(1, 2, 0).clamp(0, 1).numpy()
                            with im_c3:
                                st.image(bpda_np, caption="3. BPDA Image (Adaptive)", width=180)

                            # Amplified perturbation heatmap (10x noise)
                            noise_np = np.abs(bpda_np - clean_np)
                            noise_amplified = np.clip(noise_np * 15.0, 0, 1)
                            with im_c4:
                                st.image(noise_amplified, caption="4. BPDA Perturbation Heatmap (15×)", width=180)

                    # Tag inspection table with Color-Coded Diffs
                    c_col1, c_col2, c_col3 = st.columns(3)
                    with c_col1:
                        st.markdown("**Clean Tags Recognized:**")
                        if clean_tags_list:
                            clean_html = " ".join([f"<span style='background:#1E293B; color:#38BDF8; padding:3px 8px; border-radius:6px; margin:2px; display:inline-block; font-weight:600;'>{t}</span>" for t in clean_tags_list])
                            st.markdown(clean_html, unsafe_allow_html=True)
                        else:
                            st.caption("No positive tags")

                    with c_col2:
                        st.markdown("**Tags Survived Standard PGD:**")
                        if pgd_tags_list:
                            pgd_html = " ".join([f"<span style='background:#064E3B; color:#34D399; padding:3px 8px; border-radius:6px; margin:2px; display:inline-block; font-weight:600;'>{t}</span>" for t in pgd_tags_list])
                            st.markdown(pgd_html, unsafe_allow_html=True)
                        else:
                            st.caption("No positive tags")

                    with c_col3:
                        st.markdown("**Tags Survived BPDA Attack:**")
                        # Highlight preserved vs dropped/hallucinated
                        bpda_badges = []
                        for t in bpda_tags_list:
                            if t in clean_tags_list:
                                bpda_badges.append(f"<span style='background:#1E293B; color:#94A3B8; padding:3px 8px; border-radius:6px; margin:2px; display:inline-block;'>{t}</span>")
                            else:
                                bpda_badges.append(f"<span style='background:#78350F; color:#FBBF24; padding:3px 8px; border-radius:6px; margin:2px; display:inline-block; font-weight:600;'>+{t}</span>")
                        
                        # Show dropped tags in strikethrough red
                        for t in clean_tags_list:
                            if t not in bpda_tags_list:
                                bpda_badges.append(f"<span style='background:#7F1D1D; color:#FCA5A5; text-decoration:line-through; padding:3px 8px; border-radius:6px; margin:2px; display:inline-block;'>{t}</span>")
                        
                        if bpda_badges:
                            st.markdown(" ".join(bpda_badges), unsafe_allow_html=True)
                        else:
                            st.caption("All tags wiped!")

                    st.caption(f"Verified L-infinity perturbation on this sample: **{sa.get('max_linf', 0.0):.5f}** (Imperceptible)")

        # Detector Signals Accordion in Simulation
        sim_signals = det.get("signals", {})
        if sim_signals:
            st.subheader("Simulated Defense Diagnostic Signals")
            for sig_key, sig_title in [
                ("gradient_health", "Signal 1: Input Gradient Health & Numerical Sanity"),
                ("loss_landscape_ray", "Signal 2: 1D Loss Landscape Directional Monotonicity"),
                ("pgd_restart_sensitivity", "Signal 3: PGD Random Restart Sensitivity"),
                ("attack_strength_consistency", "Signal 4: Attack Strength Consistency (Steps & Epsilon)"),
                ("fgsm_vs_pgd", "Signal 5: FGSM vs Iterative PGD Comparison"),
            ]:
                if sig_key in sim_signals:
                    s_data = sim_signals.get(sig_key, {})
                    s_status = s_data.get("status", "N/A")
                    badge = "✅" if s_status == "OK" else ("⚠️" if s_status == "SUSPICIOUS" else "ℹ️")
                    with st.expander(f"{badge} {sig_title} — [{s_status}]"):
                        st.write(f"**Status:** {s_status}")
                        if "details" in s_data:
                            st.write(f"**Details:** {s_data['details']}")
                        st.json(s_data)

        st.caption(
            "Controlled simulation demonstrates the hallmark of gradient masking: standard gradient attacks fail due to "
            "zeroed/shattered gradients across the quantization barrier, whereas BPDA successfully bypasses the defense."
        )
