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

 max_perturbation = (
    adversarial_images - original_images
).abs().max().item()

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

def check_prediction_outputs(outputs, labels):

 valid_shape = (
    outputs.ndim == 2
    and outputs.shape[0] == labels.shape[0]
    and outputs.shape[1] > 1
)

 finite_outputs = torch.isfinite(outputs).all().item()

 passed = valid_shape and finite_outputs

 return {
    "passed": passed,
    "message": (
        f"Output shape: {tuple(outputs.shape)}; "
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

