import os
from pathlib import Path
from typing import Dict


CONFIG = {
    "dataset": "CIFAR-10",
    "batch_size": 16,
    "test_size": 500,
    "attacks": {
        "fgsm": {"epsilon": 0.03},
        "pgd": {"epsilon": 0.03, "step_size": 0.007, "steps": 10},
    },
}

RAM_PLUS_CONFIG = {
    "model_name": "RAM++",
    "dataset": "RAM++ Tags",
    "batch_size": 4,
    "test_size": 50,
    "image_size": 384,
    "tag_threshold": 0.68,
    "attacks": {
        "fgsm": {"epsilon": 4 / 255},
        "pgd": {"epsilon": 4 / 255, "step_size": 1 / 255, "steps": 10},
    },
}


def get_ram_plus_paths() -> Dict[str, Path]:
    """Resolve the shared RAM++ source and checkpoint paths without validating availability."""
    project_root = Path(__file__).resolve().parent
    ram_root = Path(
        os.environ.get(
            "MODELSHIELD_RAM_PLUS_ROOT",
            project_root / "ram-plus",
        )
    ).expanduser()
    checkpoint_path = Path(
        os.environ.get(
            "MODELSHIELD_RAM_PLUS_CHECKPOINT",
            ram_root / "pretrained" / "ram_plus_swin_large_14m.pth",
        )
    ).expanduser()

    return {
        "ram_root": ram_root,
        "checkpoint_path": checkpoint_path,
    }


def get_ram_plus_config() -> Dict[str, object]:
    """Return validated RAM++ settings without changing CIFAR-10 defaults."""
    paths = get_ram_plus_paths()
    ram_root = paths["ram_root"]
    checkpoint_path = paths["checkpoint_path"]

    if not ram_root.is_dir():
        raise FileNotFoundError(
            f"RAM++ source directory not found: {ram_root}. "
            "Set MODELSHIELD_RAM_PLUS_ROOT to the RAM++ source directory."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"RAM++ checkpoint not found: {checkpoint_path}. "
            "Set MODELSHIELD_RAM_PLUS_CHECKPOINT to a verified checkpoint."
        )

    return {
        **RAM_PLUS_CONFIG,
        "ram_root": str(ram_root),
        "checkpoint_path": str(checkpoint_path),
        "attacks": {
            name: settings.copy()
            for name, settings in RAM_PLUS_CONFIG["attacks"].items()
        },
    }
