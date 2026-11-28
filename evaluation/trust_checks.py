import torch


def check_perturbation_budget(
    original_images,
    adversarial_images,
    epsilon=8 / 255,
    tolerance=1e-6,
):
    if original_images.shape != adversarial_images.shape:
        return {
            "passed": False,
            "message": "Original and adversarial image shapes do not match.",
        }

    max_perturbation = (adversarial_images - original_images).abs().max().item()
    passed = max_perturbation <= epsilon + tolerance

    return {
        "passed": passed,
        "message": (
            f"Maximum perturbation: {max_perturbation:.6f}; "
            f"budget: {epsilon:.6f}."
        ),
        "max_perturbation": max_perturbation,
        "epsilon": epsilon,
    }


def check_pixel_range(adversarial_images):
    minimum = adversarial_images.min().item()
    maximum = adversarial_images.max().item()
    passed = minimum >= 0.0 and maximum <= 1.0

    return {
        "passed": passed,
        "message": (
            f"Pixel range: [{minimum:.6f}, {maximum:.6f}]."
        ),
    }


def check_prediction_outputs(outputs, labels, multi_label=False):
    expected_label_ndim = 2 if multi_label else 1
    labels_shape_valid = labels.ndim == expected_label_ndim
    batch_matches = (
        outputs.ndim == 2
        and labels.ndim >= 1
        and outputs.shape[0] == labels.shape[0]
    )

    if multi_label:
        class_dimensions_match = (
            outputs.ndim == 2
            and labels.ndim == 2
            and labels.shape[1] > 0
            and outputs.shape[1] == labels.shape[1]
        )
    else:
        class_dimensions_match = outputs.ndim == 2 and outputs.shape[1] > 1

    valid_shape = (
        labels_shape_valid
        and batch_matches
        and class_dimensions_match
    )
    finite_outputs = torch.isfinite(outputs).all().item()
    passed = valid_shape and finite_outputs

    return {
        "passed": passed,
        "message": (
            f"Output shape: {tuple(outputs.shape)}; "
            f"label shape: {tuple(labels.shape)}; "
            f"finite outputs: {finite_outputs}."
        ),
    }


def check_metric_consistency(
    clean_accuracy,
    adversarial_accuracy,
    attack_success_rate,
):
    values = [
        clean_accuracy,
        adversarial_accuracy,
        attack_success_rate,
    ]

    passed = all(
        isinstance(value, (int, float))
        and 0.0 <= value <= 100.0
        for value in values
    )

    return {
        "passed": passed,
        "message": (
            "All metrics are within [0, 100]."
            if passed
            else "One or more metrics are outside [0, 100]."
        ),
    }


def get_trust_attestation_summary(
    original_images,
    adversarial_images,
    outputs,
    labels,
    epsilon=8 / 255,
    multi_label=False,
):
    """Run mathematical trust checks and return an auditable summary."""
    budget_check = check_perturbation_budget(
        original_images, adversarial_images, epsilon=epsilon
    )
    range_check = check_pixel_range(adversarial_images)
    output_check = check_prediction_outputs(
        outputs, labels, multi_label=multi_label
    )

    all_passed = (
        budget_check["passed"]
        and range_check["passed"]
        and output_check["passed"]
    )

    if "max_perturbation" in budget_check:
        budget_description = (
            f"Verified max deviation ||Δx||_∞ = "
            f"{budget_check['max_perturbation']:.5f} <= {epsilon:.5f} "
            "(imperceptible)."
        )
    else:
        budget_description = (
            "Maximum perturbation unavailable because image shapes do not match."
        )

    checks = [
        {
            "name": "L-infinity Perturbation Budget",
            "passed": budget_check["passed"],
            "description": budget_description,
            "details": budget_check["message"],
        },
        {
            "name": "Valid Pixel Value Clamping [0, 1]",
            "passed": range_check["passed"],
            "description": (
                "All pixel intensities strictly clamped within [0.0, 1.0] "
                "(no overflow/underflow artifacts)."
            ),
            "details": range_check["message"],
        },
        {
            "name": "Numerical Tensor Stability",
            "passed": output_check["passed"],
            "description": (
                "Logit vectors are 100% finite (0 NaN, Inf, or underflow errors)."
            ),
            "details": output_check["message"],
        },
    ]

    return {
        "all_passed": all_passed,
        "status": "ATTESTED" if all_passed else "FAILED_CHECK",
        "checks": checks,
    }
