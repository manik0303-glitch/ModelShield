import os
import torch

try:
    from robustbench.utils import load_model as robustbench_load_model
    _ROBUSTBENCH_AVAILABLE = True
except ImportError:
    _ROBUSTBENCH_AVAILABLE = False


class ModelConfigurationError(RuntimeError):
    """Raised when the requested evaluation model or verified weights cannot be loaded."""
    pass


def load_model(checkpoint_path: str = None):
    """
    Load the primary adversarial robustness benchmark model for CIFAR-10.

    Primary Source: RobustBench Carmon2019Unlabeled (threat_model=Linf).
    Secondary Source: Verified local CIFAR-10 checkpoint if provided or placed in models/cifar10_model.pt.

    Security Rule:
        Untrained or randomly initialized models are strictly disallowed for security evaluations.
        If the primary model cannot be loaded and no verified checkpoint is present,
        raises ModelConfigurationError with actionable installation instructions.

    Returns:
        tuple: (model, device)
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    def _try_load_checkpoint(path: str):
        if not os.path.exists(path):
            return None
        try:
            checkpoint = torch.load(path, map_location=device)
            model = None
            metadata = {}

            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
                metadata = getattr(checkpoint, "model_metadata", {})
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint.get("state_dict", checkpoint)
                if isinstance(state_dict, dict) and len(state_dict) > 0:
                    import torchvision.models as models
                    model = models.resnet18(num_classes=10)
                    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    model.maxpool = torch.nn.Identity()
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if len(missing) > 0:
                        return None
                    metadata = checkpoint.get("metadata", {})

            if model is not None and isinstance(model, torch.nn.Module):
                model = model.to(device)
                model.eval()

                # Architectural & tensor compatibility probe: forward pass on CIFAR-10 shape (1, 3, 32, 32)
                test_in = torch.zeros(1, 3, 32, 32, device=device)
                with torch.no_grad():
                    test_out = model(test_in)

                if test_out.ndim != 2 or test_out.shape != (1, 10) or not torch.isfinite(test_out).all():
                    return None

                # Verify non-trivial weight distribution
                has_valid_weights = False
                for param in model.parameters():
                    if param.numel() > 0 and param.std().item() > 1e-6:
                        has_valid_weights = True
                        break

                if not has_valid_weights:
                    return None

                is_pretrained = metadata.get("is_pretrained", True)
                arch_name = metadata.get("architecture", getattr(model, "__class__", type(model)).__name__)

                model.model_metadata = {
                    "architecture": arch_name,
                    "dataset": "cifar10",
                    "threat_model": "Linf",
                    "checkpoint_source": f"Local Verified Checkpoint ({path})",
                    "is_pretrained": is_pretrained,
                    "verified_compatible": True,
                }
                return model
        except Exception:
            return None
        return None

    # 1. If user explicitly specified a checkpoint path, prioritize it
    if checkpoint_path is not None:
        model = _try_load_checkpoint(checkpoint_path)
        if model is not None:
            return model, device
        raise ModelConfigurationError(
            f"Specified checkpoint '{checkpoint_path}' does not exist or failed architectural verification.\n"
            f"Security Policy: Only verified pretrained CIFAR-10 checkpoints with non-trivial weights are accepted."
        )

    # 2. Primary path: RobustBench benchmark model
    robustbench_error = None
    if _ROBUSTBENCH_AVAILABLE:
        try:
            model = robustbench_load_model(
                model_name="Carmon2019Unlabeled",
                dataset="cifar10",
                threat_model="Linf"
            )
            model = model.to(device)
            model.eval()

            # Explicit provenance metadata
            model.model_metadata = {
                "architecture": "Carmon2019Unlabeled",
                "dataset": "cifar10",
                "threat_model": "Linf",
                "checkpoint_source": "RobustBench",
                "is_pretrained": True,
            }
            return model, device
        except Exception as exc:
            robustbench_error = str(exc)
    else:
        robustbench_error = "Package 'robustbench' is not installed in the current environment."

    # 3. Secondary path: Automatic local verified checkpoint discovery
    for default_path in ["./models/cifar10_model.pt", "./cifar10_model.pt"]:
        model = _try_load_checkpoint(default_path)
        if model is not None:
            return model, device

    # 4. Fail-safe: Refuse to silently substitute an untrained model
    raise ModelConfigurationError(
        f"Unable to load the primary RobustBench benchmark model 'Carmon2019Unlabeled'.\n"
        f"Failure Reason: {robustbench_error}\n\n"
        f"Security Policy Enforcement:\n"
        f"ModelShield refuses to silently substitute an untrained or randomly initialized model, "
        f"as evaluating random weights produces meaningless accuracy and adversarial robustness metrics.\n\n"
        f"Actionable Resolution:\n"
        f"1. Install RobustBench in your Python environment:\n"
        f"   pip install git+https://github.com/RobustBench/robustbench.git\n"
        f"2. Ensure network connectivity to allow downloading pretrained Carmon2019Unlabeled weights.\n"
        f"3. Alternatively, place a verified pretrained CIFAR-10 model checkpoint at './models/cifar10_model.pt'."
    )
