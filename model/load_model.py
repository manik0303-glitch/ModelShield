import torch
from robustbench.utils import load_model as robustbench_load_model


def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = robustbench_load_model(
        model_name="Carmon2019Unlabeled",
        dataset="cifar10",
        threat_model="Linf"
    )

    model = model.to(device)
    model.eval()

    return model, device