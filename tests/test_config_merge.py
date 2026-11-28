import pytest

import config


def test_cifar_defaults_remain_unchanged():
    assert config.CONFIG == {
        "dataset": "CIFAR-10",
        "batch_size": 16,
        "test_size": 500,
        "attacks": {
            "fgsm": {"epsilon": 0.03},
            "pgd": {"epsilon": 0.03, "step_size": 0.007, "steps": 10},
        },
    }


def test_ram_plus_settings_are_separate_from_cifar_defaults():
    assert config.RAM_PLUS_CONFIG == {
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


def test_ram_plus_config_uses_explicit_existing_checkpoint(monkeypatch, tmp_path):
    root = tmp_path / "ram"
    default_checkpoint = root / "pretrained" / "ram_plus_swin_large_14m.pth"
    default_checkpoint.parent.mkdir(parents=True)
    default_checkpoint.touch()
    explicit_checkpoint = tmp_path / "verified-custom-checkpoint.pth"
    explicit_checkpoint.touch()

    monkeypatch.setenv("MODELSHIELD_RAM_PLUS_ROOT", str(root))
    monkeypatch.setenv("MODELSHIELD_RAM_PLUS_CHECKPOINT", str(explicit_checkpoint))

    resolved = config.get_ram_plus_config()

    assert resolved["ram_root"] == str(root)
    assert resolved["checkpoint_path"] == str(explicit_checkpoint)
    assert config.CONFIG["dataset"] == "CIFAR-10"


def test_ram_plus_config_defaults_checkpoint_from_root(monkeypatch, tmp_path):
    root = tmp_path / "ram"
    checkpoint = root / "pretrained" / "ram_plus_swin_large_14m.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    monkeypatch.setenv("MODELSHIELD_RAM_PLUS_ROOT", str(root))
    monkeypatch.delenv("MODELSHIELD_RAM_PLUS_CHECKPOINT", raising=False)

    resolved = config.get_ram_plus_config()

    assert resolved["checkpoint_path"] == str(checkpoint)


def test_ram_plus_config_rejects_missing_checkpoint(monkeypatch, tmp_path):
    root = tmp_path / "ram"
    root.mkdir()
    monkeypatch.setenv("MODELSHIELD_RAM_PLUS_ROOT", str(root))
    monkeypatch.delenv("MODELSHIELD_RAM_PLUS_CHECKPOINT", raising=False)

    with pytest.raises(FileNotFoundError, match="RAM\\+\\+ checkpoint"):
        config.get_ram_plus_config()
