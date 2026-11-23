"""
Model and Checkpoint Loader for ModelShield.

Provides safe, modular loading and validation of PyTorch model checkpoints (.pt, .pth),
architecture detection and instantiation for CIFAR-10 robustness evaluation,
and verification that loaded models can execute valid forward passes.
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class SupportedArchitecture:
    WIDERESNET_28_10_CARMON = "WideResNet-28-10 (Carmon2019 / RobustBench)"
    WIDERESNET_28_10 = "WideResNet-28-10 (Standard)"
    WIDERESNET_34_10 = "WideResNet-34-10"
    RESNET18_CIFAR = "ResNet-18 (CIFAR-10)"
    RESNET18_TORCHVISION = "ResNet-18 (Standard TorchVision)"
    RESNET50_CIFAR = "ResNet-50 (CIFAR-10)"
    CUSTOM_MODULE = "Custom nn.Module / TorchScript"

    ALL = [
        WIDERESNET_28_10_CARMON,
        WIDERESNET_28_10,
        WIDERESNET_34_10,
        RESNET18_CIFAR,
        RESNET18_TORCHVISION,
        RESNET50_CIFAR,
        CUSTOM_MODULE,
    ]


class CifarResNetBasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class CifarResNet18(nn.Module):
    """Standard ResNet-18 adapted for CIFAR-10 (3x3 conv1, stride 1, no initial maxpool)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(CifarResNetBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.linear(out)


def create_model_architecture(architecture_name: str, num_classes: int = 10) -> nn.Module:
    """Instantiates a PyTorch model corresponding to the requested architecture."""
    if architecture_name == SupportedArchitecture.WIDERESNET_28_10_CARMON:
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet
        return WideResNet(depth=28, widen_factor=10, sub_block1=True, num_classes=num_classes)

    elif architecture_name == SupportedArchitecture.WIDERESNET_28_10:
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet
        return WideResNet(depth=28, widen_factor=10, sub_block1=False, num_classes=num_classes)

    elif architecture_name == SupportedArchitecture.WIDERESNET_34_10:
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet
        return WideResNet(depth=34, widen_factor=10, sub_block1=False, num_classes=num_classes)

    elif architecture_name == SupportedArchitecture.RESNET18_CIFAR:
        return CifarResNet18(num_classes=num_classes)

    elif architecture_name == SupportedArchitecture.RESNET18_TORCHVISION:
        import torchvision.models as tv_models
        return tv_models.resnet18(num_classes=num_classes)

    elif architecture_name == SupportedArchitecture.RESNET50_CIFAR:
        import torchvision.models as tv_models
        return tv_models.resnet50(num_classes=num_classes)

    elif architecture_name == SupportedArchitecture.CUSTOM_MODULE:
        raise ValueError(
            "Custom nn.Module / TorchScript requires the checkpoint to contain the complete serialized model object."
        )

    else:
        raise ValueError(f"Unsupported model architecture: {architecture_name!r}")


def clean_state_dict(raw_dict: Any) -> Dict[str, torch.Tensor]:
    """
    Extracts and standardizes the tensor state dict from checkpoint structures,
    stripping common wrapper prefixes ('module.', 'model.', '_orig_mod.').
    """
    if not isinstance(raw_dict, dict):
        raise ValueError(f"State dict must be a dictionary, got {type(raw_dict).__name__}")

    # Check for nested state dict containers
    for key in ("state_dict", "model_state_dict", "model", "net"):
        if key in raw_dict and isinstance(raw_dict[key], dict):
            raw_dict = raw_dict[key]
            break

    cleaned: Dict[str, torch.Tensor] = {}
    for key, val in raw_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        cleaned_key = key
        for prefix in ("module.", "model.", "_orig_mod."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]
        cleaned[cleaned_key] = val

    if not cleaned:
        raise ValueError("No tensor weights found in checkpoint state dictionary.")

    return cleaned


def infer_architecture_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    """
    Heuristically infers the most likely model architecture from state dict keys and tensor shapes.
    Returns None if the architecture cannot be reliably determined.
    """
    keys = set(state_dict.keys())

    # WideResNet with Carmon sub_block1
    if any("sub_block1" in k for k in keys):
        return SupportedArchitecture.WIDERESNET_28_10_CARMON

    # Standard WideResNet
    if any("block1.layer" in k for k in keys):
        # Check depth
        if any("block3.layer.4" in k for k in keys):
            return SupportedArchitecture.WIDERESNET_34_10
        elif any("block3.layer.3" in k for k in keys):
            return SupportedArchitecture.WIDERESNET_28_10

    # ResNet variants
    if any("layer1.0.conv1.weight" in k for k in keys):
        if "conv1.weight" in state_dict:
            w = state_dict["conv1.weight"]
            if len(w.shape) == 4:
                if w.shape[2] == 3 and w.shape[3] == 3:
                    return SupportedArchitecture.RESNET18_CIFAR
                elif w.shape[2] == 7 and w.shape[3] == 7:
                    return SupportedArchitecture.RESNET18_TORCHVISION

        if any("layer1.0.conv3.weight" in k for k in keys):
            return SupportedArchitecture.RESNET50_CIFAR
        return SupportedArchitecture.RESNET18_CIFAR

    return None


def load_checkpoint_safely(
    checkpoint_source: Union[str, Path, BinaryIO, bytes],
    architecture: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_classes: int = 10,
) -> Tuple[nn.Module, str, Dict[str, Any]]:
    """
    Safely loads a candidate model checkpoint.

    Parameters:
        checkpoint_source: File path, file-like buffer (e.g. from st.file_uploader), or raw bytes.
        architecture: Explicit architecture choice, or None/'auto' to trigger auto-detection.
        device: Target device (defaults to CUDA if available, else CPU).
        num_classes: Number of classification outputs (defaults to 10 for CIFAR-10).

    Returns:
        (model, resolved_architecture_name, checkpoint_info)
    """
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If filename is a string or Path, validate file extension
    source_name = "uploaded_checkpoint"
    if isinstance(checkpoint_source, (str, Path)):
        source_path = Path(checkpoint_source)
        source_name = source_path.name
        if not source_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_source}")
        ext = source_path.suffix.lower()
        if ext not in (".pt", ".pth", ".bin"):
            raise ValueError(f"Invalid checkpoint extension '{ext}'. Expected .pt or .pth.")
    elif hasattr(checkpoint_source, "name"):
        source_name = getattr(checkpoint_source, "name", "uploaded_checkpoint")
        ext = Path(source_name).suffix.lower()
        if ext and ext not in (".pt", ".pth", ".bin"):
            raise ValueError(f"Invalid uploaded file '{source_name}'. Expected .pt or .pth checkpoint.")

    # Wrap raw bytes in BytesIO
    if isinstance(checkpoint_source, bytes):
        checkpoint_source = io.BytesIO(checkpoint_source)

    # Attempt safe deserialization
    raw_loaded = None
    load_errors = []

    # First attempt: safe weights-only loading (avoids arbitrary code execution)
    try:
        raw_loaded = torch.load(checkpoint_source, map_location="cpu", weights_only=True)
    except Exception as exc_weights_only:
        load_errors.append(f"weights_only=True: {exc_weights_only}")
        # Second attempt: standard torch.load if weights_only failed (e.g. for full module instances)
        try:
            if hasattr(checkpoint_source, "seek"):
                checkpoint_source.seek(0)
            raw_loaded = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
        except Exception as exc_fallback:
            load_errors.append(f"standard load: {exc_fallback}")
            raise ValueError(
                f"Failed to load PyTorch checkpoint '{source_name}'. Details: {'; '.join(load_errors)}"
            ) from exc_fallback

    # Case 1: Checkpoint is already an instantiated nn.Module or ScriptModule
    if isinstance(raw_loaded, (nn.Module, torch.jit.ScriptModule)):
        model = raw_loaded.to(target_device)
        model.eval()
        resolved_arch = architecture or SupportedArchitecture.CUSTOM_MODULE
        _validate_model_forward(model, target_device, num_classes)
        return model, resolved_arch, {"source": source_name, "type": "nn.Module", "architecture": resolved_arch}

    # Case 2: Checkpoint is a dictionary containing state_dict weights
    if not isinstance(raw_loaded, dict):
        raise ValueError(
            f"Checkpoint '{source_name}' contains an unsupported payload type: {type(raw_loaded).__name__}. "
            "Expected a state_dict dictionary or PyTorch nn.Module."
        )

    clean_sd = clean_state_dict(raw_loaded)

    # Resolve architecture
    resolved_arch = architecture
    if not resolved_arch or resolved_arch == "auto":
        inferred = infer_architecture_from_state_dict(clean_sd)
        if inferred is None:
            raise ValueError(
                f"Could not automatically determine the model architecture for checkpoint '{source_name}'. "
                f"Please manually select an architecture from: {', '.join(SupportedArchitecture.ALL[:-1])}."
            )
        resolved_arch = inferred

    model = create_model_architecture(resolved_arch, num_classes=num_classes)

    # Load weights with informative error handling
    try:
        incompatible = model.load_state_dict(clean_sd, strict=False)
        missing_keys = incompatible.missing_keys
        unexpected_keys = incompatible.unexpected_keys

        # If too many keys are missing, the architecture is incompatible
        total_model_keys = len(list(model.state_dict().keys()))
        if len(missing_keys) > 0 and (len(missing_keys) / total_model_keys) > 0.3:
            raise ValueError(
                f"Architecture mismatch: '{resolved_arch}' is incompatible with checkpoint weights. "
                f"{len(missing_keys)} of {total_model_keys} parameters could not be loaded."
            )
    except Exception as exc:
        raise ValueError(
            f"Failed to load weights into architecture '{resolved_arch}': {exc}"
        ) from exc

    model = model.to(target_device)
    model.eval()

    _validate_model_forward(model, target_device, num_classes)

    info = {
        "source": source_name,
        "type": "state_dict",
        "architecture": resolved_arch,
        "device": str(target_device),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
    }

    return model, resolved_arch, info


def _validate_model_forward(model: nn.Module, device: torch.device, expected_classes: int = 10) -> None:
    """Verifies that the model can run a single test inference without NaN or size mismatches."""
    dummy_input = torch.zeros(1, 3, 32, 32, device=device)
    with torch.no_grad():
        try:
            output = model(dummy_input)
        except Exception as exc:
            raise ValueError(
                f"Model forward pass verification failed for input shape (1, 3, 32, 32): {exc}"
            ) from exc

    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Model forward pass must return a torch.Tensor, got {type(output).__name__}.")

    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] != expected_classes:
        raise ValueError(
            f"Model output shape mismatch: expected (1, {expected_classes}), got {tuple(output.shape)}."
        )

    if torch.isnan(output).any() or torch.isinf(output).any():
        raise ValueError("Model output contains NaN or infinite values during validation forward pass.")


def load_baseline_model(
    checkpoint_path: str = "models/cifar10/Linf/Carmon2019Unlabeled.pt",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, str, Dict[str, Any]]:
    """
    Loads the canonical Carmon2019Unlabeled baseline model for CIFAR-10 Linf robustness.
    """
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.isfile(checkpoint_path):
        model, arch, info = load_checkpoint_safely(
            checkpoint_path,
            architecture=SupportedArchitecture.WIDERESNET_28_10_CARMON,
            device=target_device,
        )
        return model, "Carmon2019Unlabeled", info
    else:
        # Fallback to robustbench loader if local file path differs
        from model.load_model import load_model as rb_load_model
        model, dev = rb_load_model()
        model = model.to(target_device).eval()
        _validate_model_forward(model, target_device, 10)
        return model, "Carmon2019Unlabeled", {
            "source": "robustbench",
            "architecture": SupportedArchitecture.WIDERESNET_28_10_CARMON,
            "device": str(target_device),
        }

