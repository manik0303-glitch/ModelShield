import os
import pickle

import torch

from config import get_ram_plus_config
from model.ram_adapter import get_ram_model, is_ram_available
from model.ram_wrapper import RAMPlusWrapper


class ModelConfigurationError(RuntimeError):
    """Raised when a requested model or verified weights cannot be loaded."""


def _convert_to_rgb(image):
    return image.convert("RGB")


def _ram_pixel_transform(image_size):
    from torchvision import transforms

    return transforms.Compose([
        _convert_to_rgb,
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_ram_plus_model():
    """Load the shared RAM++ checkpoint as differentiable tag logits."""
    try:
        settings = get_ram_plus_config()
    except FileNotFoundError as exc:
        raise ModelConfigurationError(f"RAM++ configuration is unavailable: {exc}") from exc

    available, reason = is_ram_available()
    if not available:
        raise ModelConfigurationError(f"RAM++ is unavailable: {reason}")

    base_model, device = get_ram_model()
    model = RAMPlusWrapper(base_model, image_size=settings["image_size"]).to(device)
    model.eval()

    tag_list = base_model.tag_list
    class_threshold = base_model.class_threshold
    if tag_list is None or class_threshold is None:
        raise ModelConfigurationError("RAM++ checkpoint is missing tag vocabulary or thresholds.")
    if len(tag_list) != class_threshold.numel():
        raise ModelConfigurationError(
            "RAM++ tag vocabulary and class thresholds have different lengths."
        )

    transform = _ram_pixel_transform(settings["image_size"])
    return model, device, tag_list, class_threshold, transform


def load_ram_plus_mock_model(image_size=384):
    """Create the explicit offline diagnostic model; never used as a real-model fallback."""
    semantic_tag_list = [
        "dog", "animal", "mammal", "pet", "canine",
        "outdoor", "grass", "field", "ears", "fur",
        "standing", "coat", "park", "nature", "ground",
        "snout", "eyes", "paws", "alert", "tail",
    ]

    class MockRAMPlus(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_class = len(semantic_tag_list)
            self.tag_list = semantic_tag_list
            self.class_threshold = torch.ones(self.num_class) * 0.5
            self.conv = torch.nn.Conv2d(3, 8, kernel_size=16, stride=16)
            self.head = torch.nn.Linear(8 * 24 * 24, self.num_class)
            torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

        def forward(self, image):
            features = torch.relu(self.conv(image))
            return self.head(features.flatten(1))

    transform = _ram_pixel_transform(image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MockRAMPlus().to(device)
    model.eval()
    return model, device, model.tag_list, model.class_threshold, transform


def load_model():
    """Load RAM++ for the primary ModelShield runtime."""
    return load_ram_plus_model()


def _try_load_cifar_checkpoint(path, device):
    if not os.path.exists(path):
        return None

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = None
        metadata = {}

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
            metadata = getattr(checkpoint, "model_metadata", {})
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint)
            if isinstance(state_dict, dict) and state_dict:
                import torchvision.models as models

                model = models.resnet18(num_classes=10)
                model.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
                model.maxpool = torch.nn.Identity()
                missing, _ = model.load_state_dict(state_dict, strict=False)
                if missing:
                    return None
                metadata = checkpoint.get("metadata", {})

        if not isinstance(model, torch.nn.Module):
            return None

        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 32, 32, device=device))

        if output.ndim != 2 or output.shape != (1, 10) or not torch.isfinite(output).all():
            return None

        if not any(
            parameter.numel() > 0 and parameter.std().item() > 1e-6
            for parameter in model.parameters()
        ):
            return None

        model.model_metadata = {
            "architecture": metadata.get(
                "architecture", model.__class__.__name__
            ),
            "dataset": "cifar10",
            "threat_model": "Linf",
            "checkpoint_source": f"Local Verified Checkpoint ({path})",
            "is_pretrained": metadata.get("is_pretrained", False) is True,
            "verified_compatible": True,
        }
        return model
    except pickle.UnpicklingError as exc:
        raise ModelConfigurationError(
            f"Checkpoint '{path}' could not be loaded in weights-only mode. "
            "Use a trusted CIFAR-10 ResNet-18 state_dict checkpoint instead "
            "of a serialized model object."
        ) from exc
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _load_robustbench_cifar_model():
    from robustbench.utils import load_model as robustbench_load_model

    return robustbench_load_model(
        model_name="Carmon2019Unlabeled",
        dataset="cifar10",
        threat_model="Linf",
    )


def load_cifar10_model(checkpoint_path=None, *, trust_local_checkpoint=False):
    """Load the isolated legacy CIFAR-10 RobustBench model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is not None:
        if not trust_local_checkpoint:
            raise ModelConfigurationError(
                "Local CIFAR-10 checkpoints require explicit trust_local_checkpoint=True "
                "after independently verifying their provenance."
            )
        model = _try_load_cifar_checkpoint(checkpoint_path, device)
        if model is not None:
            model.model_metadata["is_pretrained"] = True
            model.model_metadata["provenance_verified_by_caller"] = True
            return model, device
        raise ModelConfigurationError(
            f"Specified checkpoint '{checkpoint_path}' does not exist or failed "
            "architectural verification. Only compatible pretrained CIFAR-10 "
            "checkpoints are accepted."
        )

    robustbench_error = None
    try:
        model = _load_robustbench_cifar_model()
        model = model.to(device)
        model.eval()
        model.model_metadata = {
            "architecture": "Carmon2019Unlabeled",
            "dataset": "cifar10",
            "threat_model": "Linf",
            "checkpoint_source": "RobustBench",
            "is_pretrained": True,
        }
        return model, device
    except ImportError as exc:
        robustbench_error = str(exc)
    except (OSError, RuntimeError, ValueError) as exc:
        robustbench_error = str(exc)

    raise ModelConfigurationError(
        "Unable to load the legacy CIFAR-10 RobustBench model "
        "'Carmon2019Unlabeled'.\n"
        f"Failure Reason: {robustbench_error}\n\n"
        "Security Policy Enforcement:\n"
        "ModelShield refuses to substitute random weights for CIFAR-10 evaluation.\n\n"
        "Actionable Resolution:\n"
        "1. Install the optional CIFAR dependencies from requirements-cifar.txt.\n"
        "2. Alternatively, provide a locally stored checkpoint only after independently "
        "verifying its provenance and setting trust_local_checkpoint=True."
    )
