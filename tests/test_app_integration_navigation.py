from pathlib import Path

import pytest
import torch
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_exposes_ram_and_cifar_workspaces():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    workspaces = app.sidebar.radio[0].options
    assert "Dataset Tagging Robustness" in workspaces
    assert "Gradient Masking Lab" in workspaces
    assert "Test CIFAR-10 Dataset" in workspaces
    assert "Model Security Regression Gate" in workspaces
    assert "Defense Verification Lab" in workspaces


@pytest.mark.parametrize(
    ("workspace", "heading"),
    [
        ("Test CIFAR-10 Dataset", "3. Test the CIFAR-10 Dataset"),
        ("Model Security Regression Gate", "Model Security Regression Gate"),
        ("Defense Verification Lab", "Defense Verification Lab"),
    ],
)
def test_integrated_workspace_can_open_without_loading_a_model(workspace, heading):
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    app.sidebar.radio[0].set_value(workspace).run(timeout=60)

    assert not app.exception
    assert any(heading in header.value for header in app.header)


def test_cifar_snapshot_remains_available_after_navigation():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    images = torch.zeros(2, 3, 32, 32)
    predictions = torch.tensor([0, 1])
    results = {
        "total_images": 2,
        "clean_accuracy": 100.0,
        "fgsm_accuracy": 100.0,
        "pgd_accuracy": 100.0,
        "fgsm_attack_success_rate": 0.0,
        "pgd_attack_success_rate": 0.0,
        "worst_case_attack_success_rate": 0.0,
        "initially_correct_count": 2,
        "worst_case_fooled_count": 0,
        "clean_predictions": predictions,
        "fgsm_predictions": predictions,
        "pgd_predictions": predictions,
        "fgsm_images": images,
        "pgd_images": images,
    }
    app.session_state["cifar_eval_results"] = results
    app.session_state["cifar_eval_checks"] = {}
    app.session_state["cifar_eval_trust_flag"] = {
        "status": "PASS", "passed": 0, "total": 0
    }
    app.session_state["cifar_eval_images"] = images
    app.session_state["cifar_eval_labels"] = predictions
    app.session_state["cifar_eval_epsilon"] = 4 / 255
    app.session_state["cifar_eval_pgd_steps"] = 3

    app.sidebar.radio[0].set_value("Test CIFAR-10 Dataset").run(timeout=60)
    assert not app.exception
    app.button(key="btn_save_cifar_baseline").click().run(timeout=60)
    assert not app.exception

    snapshot = app.session_state["gate_baseline_snapshot"]
    assert snapshot.metadata.dataset_name == "CIFAR-10"
    assert snapshot.metadata.num_samples == 2
    app.sidebar.radio[0].set_value("Model Security Regression Gate").run(timeout=60)
    assert not app.exception
    assert app.session_state["gate_baseline_snapshot"].metadata.num_samples == 2


def test_ram_gate_displays_configured_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "configured-ram-plus.pth"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setenv("MODELSHIELD_RAM_PLUS_CHECKPOINT", str(checkpoint))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    app.sidebar.radio[0].set_value("Model Security Regression Gate").run(timeout=60)
    app.radio(key="reg_gate_model_family").set_value("Image Tagging (RAM++)").run(timeout=60)

    assert not app.exception
    assert any(str(checkpoint) in item.value for item in app.success)
