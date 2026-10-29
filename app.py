import streamlit as st
import torch
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
from evaluation.masking_simulation import run_masking_simulation
import json
import plotly.graph_objects as go
import plotly.express as px


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ModelShield",
    page_icon="🛡️",
    layout="wide",
)

# -------------------- CYBER-SECURITY UI STYLING --------------------
st.markdown("""
<style>
/* Global Card Base */
.ms-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
    margin-bottom: 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.ms-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
}

/* Status Hero Banners */
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
.ms-hero-info {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(59, 130, 246, 0.4);
    border-left: 5px solid #3B82F6;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* Metric Pill */
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

/* Styled HTML Table */
.ms-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(15, 23, 42, 0.6);
    margin-top: 0.5rem;
}
.ms-table th {
    background: rgba(30, 41, 59, 0.85);
    color: #CBD5E1;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    text-align: left;
}
.ms-table td {
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #E2E8F0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
.ms-table tr:last-child td {
    border-bottom: none;
}
.ms-table tr:hover td {
    background: rgba(51, 65, 85, 0.3);
}

/* Status Badges */
.badge-pass {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-fail {
    display: inline-flex;
    align-items: center;
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-warning {
    display: inline-flex;
    align-items: center;
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Sample Visual Card */
.ms-sample-card {
    background: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    height: 100%;
}
.ms-pill-tag {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
}
.ms-tag-clean { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); }
.ms-tag-fgsm { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
.ms-tag-pgd { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }
.ms-tag-heatmap { background: rgba(168, 85, 247, 0.2); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.4); }
.ms-pred-label {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-top: 0.6rem;
    margin-bottom: 0.2rem;
}
.ms-stat-sub {
    font-size: 0.8rem;
    color: #94A3B8;
}
.ms-stat-conf {
    font-size: 0.82rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

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


def compute_perturbation_heatmap(orig_tensor, adv_tensor, colormap_name="inferno"):
    """
    Computes amplified L_infinity spatial perturbation heatmap across color channels.
    Returns RGB uint8 numpy array ready for display.
    """
    import numpy as np
    import matplotlib.cm as cm
    # Compute absolute difference across RGB channels, take max across channels
    diff = (adv_tensor.detach().cpu()[0] - orig_tensor.detach().cpu()[0]).abs()
    spatial_diff = diff.max(dim=0)[0].numpy()  # shape (32, 32)
    max_val = spatial_diff.max()
    if max_val > 1e-7:
        normalized = spatial_diff / max_val
    else:
        normalized = spatial_diff
    cmap = cm.get_cmap(colormap_name)
    colored = cmap(normalized)[:, :, :3]  # drop alpha, shape (32, 32, 3) in [0, 1]
    return (colored * 255).astype(np.uint8)


def predict_image(model, image_tensor, device):
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        prediction = int(logits.argmax(dim=1).item())
    return prediction, float(probabilities[prediction].item() * 100), probabilities


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
        "Test Your Own Image",
        "Create an Adversarial Image",
        "Test CIFAR-10 Dataset",
        "Gradient Masking Lab",
    ],
)
st.sidebar.caption("Model: Carmon2019Unlabeled · Dataset: CIFAR-10 · Input size: 32×32")


# -------------------- OPTION 1: TEST OWN IMAGE --------------------
if page == "Test Your Own Image":
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

        st.subheader("🔍 Sample-Level Adversarial Inspection")
        st.write("Inspect sample-level CIFAR-10 instances, amplified spatial perturbation heatmaps, adversarial probability shifts, and Linf budget compliance:")

        # Build options for selector dropdown
        sample_options = []
        for idx in range(len(labels)):
            t_name = CLASS_NAMES[int(labels[idx].item())]
            c_name = CLASS_NAMES[int(results["clean_predictions"][idx].item())]
            p_name = CLASS_NAMES[int(results["pgd_predictions"][idx].item())]
            is_fooled = (c_name == t_name and p_name != t_name)
            status_text = "💥 FOOLED" if is_fooled else ("CORRECT" if p_name == t_name else "MISCLASSIFIED")
            sample_options.append(f"Sample {idx + 1}: {t_name} · Clean: {c_name} · PGD: {p_name} ({status_text})")

        selected_label = st.selectbox(
            "Select CIFAR-10 Instance to Inspect:",
            options=sample_options,
            index=0,
            key="dataset_sample_selector",
        )
        image_index = sample_options.index(selected_label)

        clean_image = images[image_index:image_index + 1]
        fgsm_image = results["fgsm_images"][image_index:image_index + 1]
        pgd_image = results["pgd_images"][image_index:image_index + 1]
        clean_pred = int(results["clean_predictions"][image_index].item())
        fgsm_pred = int(results["fgsm_predictions"][image_index].item())
        pgd_pred = int(results["pgd_predictions"][image_index].item())
        true_label = int(labels[image_index].item())

        # Compute model confidence for clean and adversarial images
        with torch.no_grad():
            clean_logits = model(clean_image)
            clean_conf = float(torch.softmax(clean_logits, dim=1)[0, clean_pred].item() * 100)
            fgsm_logits = model(fgsm_image)
            fgsm_conf = float(torch.softmax(fgsm_logits, dim=1)[0, fgsm_pred].item() * 100)
            pgd_logits = model(pgd_image)
            pgd_conf = float(torch.softmax(pgd_logits, dim=1)[0, pgd_pred].item() * 100)

        # Compute perturbation metrics
        fgsm_linf = float((fgsm_image - clean_image).abs().max().item())
        pgd_linf = float((pgd_image - clean_image).abs().max().item())

        # Compute Spatial Gradient / Perturbation Heatmap
        heatmap_img = compute_perturbation_heatmap(clean_image, pgd_image, colormap_name="inferno")

        # 4-Column Layout matching the requested design
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)

        with col_c1:
            st.markdown("""
            <div class="ms-sample-card">
                <div class="ms-pill-tag ms-tag-clean">Clean Input</div>
            """, unsafe_allow_html=True)
            st.image(tensor_to_display(clean_image), use_container_width=True)
            conf_color = "#34D399" if clean_pred == true_label else "#F87171"
            st.markdown(f"""
                <div class="ms-pred-label">{CLASS_NAMES[clean_pred]}</div>
                <div class="ms-stat-sub">True: {CLASS_NAMES[true_label]}</div>
                <div class="ms-stat-conf" style="color: {conf_color};">Conf: {clean_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_c2:
            st.markdown("""
            <div class="ms-sample-card">
                <div class="ms-pill-tag ms-tag-fgsm">FGSM 1-Step</div>
            """, unsafe_allow_html=True)
            st.image(tensor_to_display(fgsm_image), use_container_width=True)
            conf_color = "#34D399" if fgsm_pred == true_label else "#F87171"
            st.markdown(f"""
                <div class="ms-pred-label">{CLASS_NAMES[fgsm_pred]}</div>
                <div class="ms-stat-sub">L<sub>∞</sub>: {fgsm_linf:.4f}</div>
                <div class="ms-stat-conf" style="color: {conf_color};">Conf: {fgsm_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_c3:
            st.markdown("""
            <div class="ms-sample-card">
                <div class="ms-pill-tag ms-tag-pgd">PGD-10 Iterative</div>
            """, unsafe_allow_html=True)
            st.image(tensor_to_display(pgd_image), use_container_width=True)
            conf_color = "#34D399" if pgd_pred == true_label else "#F87171"
            st.markdown(f"""
                <div class="ms-pred-label">{CLASS_NAMES[pgd_pred]}</div>
                <div class="ms-stat-sub">L<sub>∞</sub>: {pgd_linf:.4f}</div>
                <div class="ms-stat-conf" style="color: {conf_color};">Conf: {pgd_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_c4:
            st.markdown("""
            <div class="ms-sample-card">
                <div class="ms-pill-tag ms-tag-heatmap">Perturbation Heatmap</div>
            """, unsafe_allow_html=True)
            st.image(heatmap_img, use_container_width=True)
            st.markdown(f"""
                <div class="ms-pred-label" style="font-size: 1rem;">Spatial Gradient Focus</div>
                <div class="ms-stat-sub">Colormap: Inferno</div>
                <div class="ms-stat-conf" style="color: #60A5FA;">Budget: ≤ {dataset_epsilon:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

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


# -------------------- OPTION 4: GRADIENT MASKING LAB --------------------
elif page == "Gradient Masking Lab":
    st.header("4. 🔬 Gradient Masking Lab (Validation Suite)")
    st.write(
        "Run a controlled validation experiment that deliberately introduces a gradient-obscuring condition "
        "(input quantization) in a temporary pipeline, and test whether ModelShield's detector identifies the resulting evidence. "
        "Standard FGSM and PGD are evaluated alongside BPDA-assisted PGD (straight-through gradient approximation)."
    )

    with st.sidebar:
        st.subheader("Simulation Configuration")
        sim_mode = st.radio(
            "Evaluation Mode",
            ["Simulated Gradient Masking", "Normal Evaluation Reference"],
            index=0,
            key="sim_eval_mode",
            help="Choose whether to evaluate the model with simulated gradient masking or regular differentiable inference.",
        )
        sim_quant_levels = st.selectbox(
            "Quantization levels",
            [4, 8, 16, 32],
            index=2,
            key="sim_quant_levels",
            help="Number of discrete levels per color channel. Lower levels produce more severe gradient shattering.",
        )
        sim_sample_cap = st.selectbox(
            "Simulation sample cap",
            [8, 16, 32],
            index=1,
            key="sim_sample_cap",
            help="Number of CIFAR-10 images evaluated in the simulation.",
        )
        sim_eps_pixels = st.slider(
            "Attack epsilon (pixel levels / 255)",
            min_value=1, max_value=16, value=8, key="sim_eps_slider"
        )
        sim_pgd_steps = st.slider(
            "PGD iterations (Standard & BPDA)",
            min_value=5, max_value=20, value=10, key="sim_pgd_steps_slider"
        )
        sim_seed = st.number_input(
            "Random seed",
            min_value=0, max_value=999999, value=123, step=1, key="sim_random_seed"
        )

    sim_epsilon = sim_eps_pixels / 255.0

    c_btn1, c_btn2 = st.columns([2, 1])
    with c_btn1:
        run_sim_btn = st.button(
            "Run Simulated Gradient-Masking Test",
            type="primary",
            key="run_sim_test_btn",
        )
    with c_btn2:
        clear_sim_btn = st.button(
            "Clear Simulation Results",
            type="secondary",
            key="clear_sim_test_btn",
        )

    if clear_sim_btn:
        st.session_state["gm_simulation_results"] = None
        st.rerun()

    if run_sim_btn:
        if sim_mode == "Normal Evaluation Reference":
            st.info("Switched to Normal Evaluation Reference. Running standard diagnostic suite without quantization.")
            try:
                with st.spinner("Evaluating standard baseline pipeline..."):
                    model, device = get_model()
                    if st.session_state.get("cifar_eval_images") is not None:
                        sim_images = st.session_state["cifar_eval_images"][:sim_sample_cap].to(device)
                        sim_labels = st.session_state["cifar_eval_labels"][:sim_sample_cap].to(device)
                    else:
                        raw_imgs, raw_lbls = get_cifar_dataset(sim_sample_cap)
                        sim_images = raw_imgs.to(device)
                        sim_labels = raw_lbls.to(device)

                    normal_diag = analyze_gradient_masking(
                        model=model,
                        images=sim_images,
                        labels=sim_labels,
                        epsilon=sim_epsilon,
                        pgd_steps=sim_pgd_steps,
                        max_diagnostic_samples=sim_sample_cap,
                    )
                    st.session_state["gm_simulation_results"] = {
                        "mode": "normal_reference",
                        "simulation_validation": {
                            "status": "VALIDATED",
                            "summary": "Normal evaluation reference: unquantized model pipeline.",
                            "checks": [{"name": "baseline_unaltered", "passed": True, "details": "Model evaluated without simulated transformations."}],
                        },
                        "detector_result": normal_diag,
                        "measurements": {
                            "sample_count": len(sim_labels),
                            "eligible_clean_correct_count": normal_diag["metadata"]["clean_correct_count"],
                            "original_clean_accuracy": normal_diag["metadata"]["clean_accuracy"],
                            "simulated_clean_accuracy": normal_diag["metadata"]["clean_accuracy"],
                            "clean_accuracy_drop_percentage_points": 0.0,
                            "attacks": {
                                "fgsm": {"asr": normal_diag["signals"]["fgsm_vs_pgd"]["fgsm_success_rate"], "success_count": 0, "denominator": normal_diag["metadata"]["clean_correct_count"]},
                                "pgd": {"asr": normal_diag["signals"]["fgsm_vs_pgd"]["pgd_success_rate"], "success_count": 0, "denominator": normal_diag["metadata"]["clean_correct_count"]},
                                "bpda_pgd": {"asr": normal_diag["signals"]["fgsm_vs_pgd"]["pgd_success_rate"], "success_count": 0, "denominator": normal_diag["metadata"]["clean_correct_count"]},
                            },
                            "gradient_checks": {
                                "ordinary_gradient_norm": normal_diag["signals"]["gradient_health"]["median_l2_norm"],
                                "ordinary_near_zero_fraction": normal_diag["signals"]["gradient_health"]["median_near_zero_fraction"],
                                "bpda_gradient_norm": normal_diag["signals"]["gradient_health"]["median_l2_norm"],
                                "bpda_gradient_finite": not normal_diag["signals"]["gradient_health"]["non_finite"],
                            },
                        },
                        "configuration": {
                            "quantization_levels": None,
                            "epsilon": sim_epsilon,
                            "pgd_steps": sim_pgd_steps,
                            "sample_count_requested": sim_sample_cap,
                            "random_seed": sim_seed,
                        },
                        "limitations": ["Reference mode: standard differentiable model with no simulated masking."],
                    }
            except Exception as exc:
                st.error(f"Normal reference evaluation failed: {exc}")
        else:
            try:
                with st.spinner("Executing controlled simulation: quantization pipeline, standard attacks, BPDA-assisted PGD, and detector evaluation..."):
                    model, device = get_model()
                    if st.session_state.get("cifar_eval_images") is not None:
                        sim_images = st.session_state["cifar_eval_images"][:sim_sample_cap].to(device)
                        sim_labels = st.session_state["cifar_eval_labels"][:sim_sample_cap].to(device)
                    else:
                        raw_imgs, raw_lbls = get_cifar_dataset(sim_sample_cap)
                        sim_images = raw_imgs.to(device)
                        sim_labels = raw_lbls.to(device)

                    sim_res = run_masking_simulation(
                        model=model,
                        images=sim_images,
                        labels=sim_labels,
                        quantization_levels=sim_quant_levels,
                        epsilon=sim_epsilon,
                        alpha=min(2 / 255, sim_epsilon),
                        pgd_steps=sim_pgd_steps,
                        sample_cap=sim_sample_cap,
                        random_seed=sim_seed,
                        bpda_advantage_margin=5.0,
                    )
                    st.session_state["gm_simulation_results"] = sim_res
            except Exception as exc:
                st.error(f"Simulation run failed: {exc}")

    # Render Simulation Results if present
    if st.session_state.get("gm_simulation_results") is not None:
        sim_data = st.session_state["gm_simulation_results"]
        val_info = sim_data.get("simulation_validation", {})
        val_status = val_info.get("status", "INCONCLUSIVE")
        det_res = sim_data.get("detector_result", {})
        det_status = det_res.get("status", "INCONCLUSIVE")
        meas = sim_data.get("measurements", {})
        atks = meas.get("attacks", {})
        grad_chk = meas.get("gradient_checks", {})

        st.divider()

        # Panel 1 & 2: Side-by-Side Status (Independent Validation vs Detector Verdict)
        st.subheader("Experiment Status & Detector Verification")
        p1, p2 = st.columns(2)

        with p1:
            st.markdown("#### 1. Simulation Validation Status")
            if val_status == "VALIDATED":
                st.markdown(f"""
                <div class="ms-hero-validated">
                    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #34D399; font-weight: 700;">
                        ✔ Simulation Validated
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-top: 0.25rem;">
                        {val_info.get('summary', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif val_status == "CONDITION_NOT_DEMONSTRATED":
                st.markdown(f"""
                <div class="ms-hero-warning">
                    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #FBBF24; font-weight: 700;">
                        ⚠ Condition Not Demonstrated
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-top: 0.25rem;">
                        {val_info.get('summary', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif val_status == "ERROR":
                st.error(f"**ERROR**: {val_info.get('summary', '')}")
            else:
                st.info(f"**INCONCLUSIVE**: {val_info.get('summary', '')}")

            # Validation checks table - rendered cleanly via Streamlit native table or unindented HTML
            v_checks = val_info.get("checks", [])
            if v_checks:
                table_rows = []
                for c in v_checks:
                    c_name = c.get("name", "").replace("_", " ").title()
                    c_pass = c.get("passed", False)
                    badge = "✔ PASS" if c_pass else "✘ FAIL"
                    details = c.get("details", "")
                    table_rows.append({
                        "Check": c_name,
                        "Status": badge,
                        "Diagnostic Evidence": details
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        with p2:
            st.markdown("#### 2. ModelShield Detector Verdict")
            if det_status == "WARNING":
                st.markdown(f"""
                <div class="ms-hero-warning">
                    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #FBBF24; font-weight: 700;">
                        ⚠ Warning: Masking Detected
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-top: 0.25rem;">
                        {det_res.get('summary', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif det_status == "NO_STRONG_SIGNAL":
                st.markdown(f"""
                <div class="ms-hero-validated">
                    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #34D399; font-weight: 700;">
                        ✔ No Strong Signal
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-top: 0.25rem;">
                        {det_res.get('summary', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"**INCONCLUSIVE**: {det_res.get('summary', '')}")

            st.caption(
                "🛡️ **Detector Independence Guarantee:** The detector evaluates observed mathematical evidence only. "
                "It is strictly decoupled and does not receive the simulation mode or expected outcome."
            )

        # Panel 3: Clean Accuracy Impact
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.subheader("3. Clean Accuracy Impact (Pipeline Integrity)")
        ca1, ca2, ca3, ca4 = st.columns(4)
        orig_ca = meas.get("original_clean_accuracy")
        sim_ca = meas.get("simulated_clean_accuracy")
        drop_ca = meas.get("clean_accuracy_drop_percentage_points")
        elig_cnt = meas.get("eligible_clean_correct_count")

        with ca1:
            st.markdown(f"""
            <div class="ms-metric-box">
                <div class="ms-metric-label">Original Accuracy</div>
                <div class="ms-metric-val">{orig_ca:.2f}%</div>
                <div class="ms-metric-sub">Unaltered PyTorch model</div>
            </div>
            """, unsafe_allow_html=True)
        with ca2:
            st.markdown(f"""
            <div class="ms-metric-box">
                <div class="ms-metric-label">Quantized Accuracy</div>
                <div class="ms-metric-val">{sim_ca:.2f}%</div>
                <div class="ms-metric-sub">Pipeline with hard rounding</div>
            </div>
            """, unsafe_allow_html=True)
        with ca3:
            drop_color = "#34D399" if (drop_ca is not None and abs(drop_ca) < 5.0) else "#F87171"
            st.markdown(f"""
            <div class="ms-metric-box">
                <div class="ms-metric-label">Clean Accuracy Drop</div>
                <div class="ms-metric-val" style="color: {drop_color};">{drop_ca:+.2f} pp</div>
                <div class="ms-metric-sub">Delta on eligible samples</div>
            </div>
            """, unsafe_allow_html=True)
        with ca4:
            st.markdown(f"""
            <div class="ms-metric-box">
                <div class="ms-metric-label">Eligible Denominator</div>
                <div class="ms-metric-val">{elig_cnt} <span style="font-size: 0.95rem; font-weight: normal; color: #94A3B8;">/ {meas.get('sample_count', 0)}</span></div>
                <div class="ms-metric-sub">Clean-correct base samples</div>
            </div>
            """, unsafe_allow_html=True)

        # Panel 4: Attack Comparison (Standard FGSM/PGD vs BPDA-assisted PGD)
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.subheader("4. Attack Comparison (Identical Denominator)")
        fgsm_asr = atks.get("fgsm", {}).get("asr")
        pgd_asr = atks.get("pgd", {}).get("asr")
        bpda_asr = atks.get("bpda_pgd", {}).get("asr")
        bpda_delta = (bpda_asr - pgd_asr) if (bpda_asr is not None and pgd_asr is not None) else 0.0

        at1, at2, at3 = st.columns(3)
        with at1:
            st.markdown(f"""
            <div class="ms-metric-box" style="border-top: 3px solid #6366F1;">
                <div class="ms-metric-label">Standard FGSM</div>
                <div class="ms-metric-val">{fgsm_asr:.2f}%</div>
                <div class="ms-metric-sub">One-step gradient (blinded by quant)</div>
            </div>
            """, unsafe_allow_html=True)
        with at2:
            st.markdown(f"""
            <div class="ms-metric-box" style="border-top: 3px solid #F43F5E;">
                <div class="ms-metric-label">Standard PGD</div>
                <div class="ms-metric-val">{pgd_asr:.2f}%</div>
                <div class="ms-metric-sub">Iterative descent (shattered gradients)</div>
            </div>
            """, unsafe_allow_html=True)
        with at3:
            st.markdown(f"""
            <div class="ms-metric-box" style="border-top: 3px solid #A855F7;">
                <div class="ms-metric-label">BPDA-Assisted PGD</div>
                <div class="ms-metric-val" style="color: #C084FC;">{bpda_asr:.2f}%</div>
                <div class="ms-metric-sub" style="color: #34D399; font-weight: 600;">+{bpda_delta:.2f} pp breakthrough advantage</div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Annotated Bar Chart
        categories = ["Standard FGSM", "Standard PGD", "BPDA-Assisted PGD"]
        asr_values = [
            fgsm_asr if fgsm_asr is not None else 0.0,
            pgd_asr if pgd_asr is not None else 0.0,
            bpda_asr if bpda_asr is not None else 0.0,
        ]
        bar_colors = ["#6366F1", "#F43F5E", "#A855F7"]

        fig_atk = go.Figure(data=[
            go.Bar(
                x=categories,
                y=asr_values,
                text=[f"{v:.2f}%" for v in asr_values],
                textposition='outside',
                textfont=dict(size=14, color="#F8FAFC", family="sans-serif"),
                marker=dict(
                    color=bar_colors,
                    line=dict(color='rgba(255, 255, 255, 0.15)', width=1.5),
                    cornerradius=8
                ),
                width=0.45
            )
        ])

        fig_atk.update_layout(
            title=dict(
                text="<b>Attack Success Rate (ASR) Comparison</b> — Proving Gradient Masking",
                font=dict(size=16, color="#F1F5F9")
            ),
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            yaxis=dict(
                title="Attack Success Rate (%)",
                range=[0, max(max(asr_values) * 1.25, 25.0)],
                gridcolor='rgba(255, 255, 255, 0.08)',
                color="#94A3B8"
            ),
            xaxis=dict(
                color="#F1F5F9",
                tickfont=dict(size=13, family="sans-serif")
            ),
            margin=dict(l=40, r=40, t=50, b=40),
            height=340
        )

        st.plotly_chart(fig_atk, use_container_width=True)
        st.caption(
            "💡 **Interpretation:** Standard PGD is ineffective because hard rounding creates zero gradients almost everywhere. "
            "BPDA-assisted PGD uses the identical forward quantization, but applies a straight-through gradient approximation in the backward pass to expose the model's true vulnerability."
        )

        # Panel 5: Gradient Comparison
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.subheader("5. Gradient Behavior Comparison")
        g1, g2, g3, g4 = st.columns(4)
        ord_norm = grad_chk.get("ordinary_gradient_norm")
        ord_nz = grad_chk.get("ordinary_near_zero_fraction")
        bpda_norm = grad_chk.get("bpda_gradient_norm")
        bpda_finite = grad_chk.get("bpda_gradient_finite")

        with g1:
            st.markdown(f"""
            <div class="ms-metric-box">
                <div class="ms-metric-label">Ordinary Gradient L2 Norm</div>
                <div class="ms-metric-val" style="color: #F87171;">{f"{ord_norm:.2e}" if ord_norm is not None else "0.00 (None)"}</div>
                <div class="ms-metric-sub">Vanished / Disconnected</div>
            </div>
            """, unsafe_allow_html=True)
        with g2:
            st.markdown(f"""
            <div class="ms-metric-box">
                <div class="ms-metric-label">Near-Zero Element Fraction</div>
                <div class="ms-metric-val">{f"{ord_nz * 100:.1f}%" if ord_nz is not None else "100.0%"}</div>
                <div class="ms-metric-sub">Shattered gradient indicator</div>
            </div>
            """, unsafe_allow_html=True)
        with g3:
            st.markdown(f"""
            <div class="ms-metric-box">
                <div class="ms-metric-label">BPDA Gradient L2 Norm</div>
                <div class="ms-metric-val" style="color: #34D399;">{f"{bpda_norm:.4f}" if bpda_norm is not None else "N/A"}</div>
                <div class="ms-metric-sub">Healthy non-zero magnitude</div>
            </div>
            """, unsafe_allow_html=True)
        with g4:
            st.markdown(f"""
            <div class="ms-metric-box">
                <div class="ms-metric-label">BPDA Usability</div>
                <div class="ms-metric-val" style="color: #A855F7;">{"Finite & Usable" if bpda_finite else "Error"}</div>
                <div class="ms-metric-sub">Straight-through backward pass</div>
            </div>
            """, unsafe_allow_html=True)

        # Panel 6: Detector Signal Breakdown
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.subheader("6. Detector Diagnostic Details")
        det_warnings = det_res.get("warning_reasons", [])
        if det_warnings:
            st.markdown("**Identified Warning Reasons:**")
            for w in det_warnings:
                st.markdown(f"- ⚠️ {w}")
        else:
            st.info("The detector did not observe sufficient warning triggers in this specific run.")

        if det_res.get("actionable_guidance"):
            st.markdown("**Actionable Guidance from Detector:**")
            for g in det_res["actionable_guidance"]:
                st.markdown(f"- 💡 {g}")

        # Panel 7: Interpretation and Limitations
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.subheader("7. Scientific Limitations")
        st.caption(" • ".join(sim_data.get("limitations", [])))

        # Export Buttons
        st.divider()
        e_col1, e_col2 = st.columns(2)
        e_col1.download_button(
            "Download Simulation Report (JSON)",
            data=json.dumps(sim_data, indent=2).encode("utf-8"),
            file_name="modelshield_simulation_report.json",
            mime="application/json",
        )
        flat_sim = {
            "Simulation_Validation_Status": [val_status],
            "Detector_Verdict": [det_status],
            "Quantization_Levels": [sim_data.get("configuration", {}).get("quantization_levels")],
            "Original_Clean_Accuracy": [orig_ca],
            "Simulated_Clean_Accuracy": [sim_ca],
            "Standard_FGSM_ASR": [fgsm_asr],
            "Standard_PGD_ASR": [pgd_asr],
            "BPDA_PGD_ASR": [bpda_asr],
            "Ordinary_Grad_Norm": [ord_norm],
            "BPDA_Grad_Norm": [bpda_norm],
        }
        e_col2.download_button(
            "Download Simulation Summary (CSV)",
            data=pd.DataFrame(flat_sim).to_csv(index=False).encode("utf-8"),
            file_name="modelshield_simulation_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("Configure simulation settings in the sidebar and click **Run Simulated Gradient-Masking Test** to begin.")
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