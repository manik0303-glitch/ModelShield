import builtins
import io
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from attacks.pgd import pgd_attack
from evaluation.live_eval import evaluate_model_live, is_snapshot_compatible_with_config, load_cifar10_samples
from evaluation.regression_gate import (
    EvaluationMetadata,
    EvaluationSnapshot,
    RegressionThresholds,
    validate_evaluation_conditions,
)
from model.checkpoint_loader import CifarResNet18, SupportedArchitecture, load_baseline_model, load_checkpoint_safely
from model.ram_loader import load_ram_candidate_model


def test_untrusted_cifar_checkpoint_never_falls_back_to_pickle():
    with patch("model.checkpoint_loader.torch.load", side_effect=ValueError("unsafe")) as load:
        with pytest.raises(ValueError):
            load_checkpoint_safely(io.BytesIO(b"invalid"), device=torch.device("cpu"))

    assert load.call_count == 1
    assert load.call_args.kwargs["weights_only"] is True


def test_untrusted_ram_checkpoint_uses_weights_only():
    with patch("model.ram_loader.torch.load", side_effect=ValueError("unsafe")) as load:
        with pytest.raises(ValueError):
            load_ram_candidate_model(io.BytesIO(b"invalid"), device="cpu")

    assert load.call_args.kwargs["weights_only"] is True


def test_cifar_baseline_fallback_uses_cifar_loader(tmp_path):
    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    model = Model()
    with (
        patch("model.load_model.load_cifar10_model", return_value=(model, torch.device("cpu"))) as load,
        patch("model.checkpoint_loader._validate_model_forward"),
    ):
        actual, name, _ = load_baseline_model(
            checkpoint_path=str(tmp_path / "missing.pt"), device=torch.device("cpu")
        )

    load.assert_called_once_with()
    assert actual is model
    assert name == "Carmon2019Unlabeled"


def test_snapshot_random_start_matches_attack_configuration():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    images = torch.rand(2, 3, 32, 32)
    labels = model(images).argmax(dim=1)

    with patch("attacks.ensemble.pgd_attack", wraps=pgd_attack) as attack:
        snapshot, _ = evaluate_model_live(
            model,
            "candidate",
            n_samples=2,
            pgd_steps=1,
            random_start=False,
            device=torch.device("cpu"),
            preloaded_data=(images, labels),
        )

    assert attack.call_args.kwargs["random_start"] is False
    assert snapshot.metadata.random_start is False


def test_ram_gate_uses_shared_main_model_for_default_checkpoint():
    import model.ram_loader as loader

    model = object()
    transform = object()
    device = torch.device("cpu")
    loader.load_ram_model.clear()
    with (
        patch.object(loader, "get_ram_model", return_value=(model, device)) as load,
        patch.object(loader, "ram") as ram,
    ):
        ram.get_transform.return_value = transform
        actual_model, actual_transform, actual_device = loader.load_ram_model()
    loader.load_ram_model.clear()

    load.assert_called_once_with()
    assert (actual_model, actual_transform, actual_device) == (model, transform, device)


def test_cifar_checkpoint_rejects_missing_weights():
    state = CifarResNet18(num_classes=10).state_dict()
    state.pop("linear.weight")
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)

    with pytest.raises(ValueError, match="weights|incompatible|mismatch"):
        load_checkpoint_safely(buffer, architecture=SupportedArchitecture.RESNET18_CIFAR)


def test_cifar_checkpoint_accepts_complete_weights():
    state = CifarResNet18(num_classes=10).state_dict()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)

    model, _, _ = load_checkpoint_safely(
        buffer, architecture=SupportedArchitecture.RESNET18_CIFAR, device=torch.device("cpu")
    )

    assert torch.equal(model.state_dict()["linear.weight"], state["linear.weight"])


def test_snapshot_missing_requested_step_size_cannot_be_reused():
    snapshot = EvaluationSnapshot(
        metadata=EvaluationMetadata(
            dataset_name="CIFAR-10", num_samples=8, epsilon=0.1, pgd_steps=4, alpha=None
        ),
        metrics={},
    )

    assert not is_snapshot_compatible_with_config(
        snapshot, "CIFAR-10", n_samples=8, epsilon=0.1, pgd_steps=4, alpha=0.01
    )


def test_missing_cifar_dependency_has_actionable_error():
    original_import = builtins.__import__

    def missing_robustbench(name, *args, **kwargs):
        if name == "robustbench.data":
            raise ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=missing_robustbench):
        with pytest.raises(RuntimeError, match="requirements-cifar.txt"):
            load_cifar10_samples()


def test_ram_candidate_rejects_incomplete_state_dict():
    import model.ram_loader as loader

    model = MagicMock()
    model.load_state_dict.side_effect = RuntimeError("Missing weights")
    with (
        patch.object(loader, "ram_plus", return_value=model),
        patch.object(loader, "ram_original"),
        patch.object(loader, "validate_ram_state_dict", return_value=(True, "")),
        patch.object(loader.torch, "load", return_value={"_orig_mod.image_encoder.weight": torch.ones(1)}),
    ):
        with pytest.raises(ValueError, match="weights do not match"):
            loader.load_ram_candidate_model(io.BytesIO(b"checkpoint"), device="cpu")

    assert model.load_state_dict.call_args.kwargs["strict"] is True
    assert "image_encoder.weight" in model.load_state_dict.call_args.args[0]


def test_gate_rejects_different_pgd_random_start_conditions():
    baseline = EvaluationMetadata(
        dataset_name="CIFAR-10", num_samples=8, epsilon=0.1, pgd_steps=4, random_start=True
    )
    candidate = EvaluationMetadata(
        dataset_name="CIFAR-10", num_samples=8, epsilon=0.1, pgd_steps=4, random_start=False
    )

    compatible, reasons, _ = validate_evaluation_conditions(
        baseline, candidate, RegressionThresholds(), {"pgd_accuracy": 90.0}, {"pgd_accuracy": 90.0}
    )

    assert not compatible
    assert any("random start" in reason.lower() for reason in reasons)


def test_snapshot_reuse_rejects_different_random_start():
    snapshot = EvaluationSnapshot(
        metadata=EvaluationMetadata(
            dataset_name="CIFAR-10",
            num_samples=8,
            epsilon=0.1,
            pgd_steps=4,
            random_start=False,
        ),
        metrics={},
    )

    assert not is_snapshot_compatible_with_config(
        snapshot, "CIFAR-10", n_samples=8, epsilon=0.1, pgd_steps=4, random_start=True
    )


def test_shared_ram_loader_retries_on_cpu_after_cuda_memory_error():
    import model.ram_adapter as adapter

    model = MagicMock()
    model.to.side_effect = [torch.cuda.OutOfMemoryError("out of memory"), model]
    ram_models = types.ModuleType("ram.models")
    ram_models.ram_plus = MagicMock(return_value=model)
    with (
        patch.dict(sys.modules, {"ram.models": ram_models}),
        patch.object(adapter, "get_ram_plus_config", return_value={"image_size": 384, "tag_threshold": 0.68}),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "empty_cache"),
    ):
        with pytest.warns(RuntimeWarning, match="using CPU"):
            actual_model, device = adapter._load_ram_model()

    assert actual_model is model
    assert device.type == "cpu"
    assert [call.args[0].type for call in model.to.call_args_list] == ["cuda", "cpu"]
