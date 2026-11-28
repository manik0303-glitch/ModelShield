import torch
from robustbench.utils import load_model

print("Downloading Sehwag2020Hydra...")

model = load_model(
    model_name="Sehwag2020Hydra",
    dataset="cifar10",
    threat_model="Linf"
)

torch.save(
    model.state_dict(),
    "candidate_sehwag2020hydra.pth"
)

print("Candidate checkpoint saved!")
