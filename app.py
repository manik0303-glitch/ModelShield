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
import json


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
else:
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