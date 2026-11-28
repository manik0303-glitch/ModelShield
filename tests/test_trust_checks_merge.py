import torch

from evaluation.trust_checks import (
    check_prediction_outputs,
    get_trust_attestation_summary,
)


def test_prediction_outputs_keep_single_label_shape_contract():
    outputs = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    labels = torch.tensor([0, 1])

    assert check_prediction_outputs(outputs, labels)["passed"] is True
    assert check_prediction_outputs(outputs, labels[:, None])["passed"] is False


def test_multi_label_prediction_outputs_require_matching_tag_dimensions():
    outputs = torch.tensor([[1.0, -1.0, 0.5], [-1.0, 1.0, 0.5]])
    labels = torch.tensor([[1, 0, 1], [0, 1, 0]])
    wrong_width_labels = labels[:, :2]

    assert check_prediction_outputs(outputs, labels, multi_label=True)["passed"] is True
    assert (
        check_prediction_outputs(
            outputs, wrong_width_labels, multi_label=True
        )["passed"]
        is False
    )
    assert check_prediction_outputs(outputs, labels[:, 0], multi_label=True)["passed"] is False


def test_invalid_perturbation_shape_is_unavailable_not_zero():
    original = torch.full((1, 3, 2, 2), 0.5)
    adversarial = torch.full((1, 3, 2, 3), 0.5)
    outputs = torch.tensor([[1.0, -1.0]])
    labels = torch.tensor([1])

    summary = get_trust_attestation_summary(
        original, adversarial, outputs, labels
    )

    budget_check = summary["checks"][0]
    assert summary["all_passed"] is False
    assert "unavailable" in budget_check["description"].lower()
    assert "0.00000" not in budget_check["description"]


def test_attestation_accepts_explicit_multi_label_mode():
    original = torch.full((1, 3, 2, 2), 0.5)
    adversarial = original + 0.01
    outputs = torch.tensor([[1.0, -1.0]])
    labels = torch.tensor([[1, 0]])

    summary = get_trust_attestation_summary(
        original,
        adversarial,
        outputs,
        labels,
        multi_label=True,
    )

    assert summary["all_passed"] is True
    assert summary["status"] == "ATTESTED"
