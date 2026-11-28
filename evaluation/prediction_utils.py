import torch
from typing import Tuple, List, Optional, Union


def predict_multilabel(
    model: torch.nn.Module,
    images: torch.Tensor,
    threshold: Union[float, torch.Tensor] = 0.68,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs forward pass and produces multi-label predictions via sigmoid thresholding.

    Args:
        model: Differentiable PyTorch model returning (B, num_tags) logits.
        images: Input tensor (B, C, H, W).
        threshold: Decision threshold (scalar float or per-tag tensor). Default 0.68.
        device: Device to place tensors on, if specified.

    Returns:
        logits: Raw output logits (B, num_tags).
        pred_binary: Binary multi-hot predictions (B, num_tags) of dtype torch.long.
    """
    if device is not None:
        images = images.to(device)

    with torch.no_grad():
        logits = model(images)

    thresh = threshold
    if isinstance(thresh, torch.Tensor):
        thresh = thresh.to(logits.device)

    pred_binary = (logits.sigmoid() > thresh).long()
    return logits, pred_binary


def is_correct_multilabel(
    pred_binary: torch.Tensor,
    labels_binary: torch.Tensor,
    mode: str = "exact",
) -> torch.Tensor:
    """
    Determines whether multi-label predictions match ground-truth multi-hot labels.

    Args:
        pred_binary: (B, num_tags) binary prediction tensor {0, 1}.
        labels_binary: (B, num_tags) binary ground-truth label tensor {0, 1}.
        mode:
            "exact": Strict exact-match — all tags in the vector must match exactly.
            "subset": Ground-truth tags must all be present in predictions.
            "any": At least one ground-truth tag is correctly detected.

    Returns:
        (B,) boolean tensor indicating per-sample correctness.
    """
    labels_binary = labels_binary.long()
    pred_binary = pred_binary.long()

    if mode == "exact":
        return (pred_binary == labels_binary).all(dim=1)
    elif mode == "subset":
        # Check if all ground-truth 1s are predicted as 1
        gt_mask = labels_binary == 1
        # If no positive tags in ground truth, check if predictions are all 0
        no_gt = gt_mask.sum(dim=1) == 0
        correct_when_gt = (pred_binary[gt_mask] == 1) if gt_mask.any() else torch.tensor([], dtype=torch.bool)
        # Per sample:
        match_subset = ((pred_binary * labels_binary).sum(dim=1) == labels_binary.sum(dim=1))
        return match_subset | no_gt
    elif mode == "any":
        gt_present = (labels_binary == 1).any(dim=1)
        overlap = ((pred_binary == 1) & (labels_binary == 1)).any(dim=1)
        # If ground truth has no tags, sample is considered correct if predictions are also empty
        empty_match = (labels_binary.sum(dim=1) == 0) & (pred_binary.sum(dim=1) == 0)
        return overlap | empty_match
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'exact', 'subset', or 'any'.")


def is_fooled_multilabel(
    clean_pred_binary: torch.Tensor,
    adv_pred_binary: torch.Tensor,
    clean_correct_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Determines which initially correct samples were successfully fooled by an attack.

    An attack succeeds on sample i if:
      1. The clean prediction was correct (clean_correct_mask[i] is True).
      2. The adversarial prediction differs from the clean prediction on at least one tag.

    Args:
        clean_pred_binary: (B, num_tags) binary prediction on clean input.
        adv_pred_binary: (B, num_tags) binary prediction on adversarial input.
        clean_correct_mask: (B,) boolean mask of initially correct clean inputs.

    Returns:
        (B,) boolean tensor where True means the sample was successfully fooled.
    """
    prediction_changed = (clean_pred_binary != adv_pred_binary).any(dim=1)
    return clean_correct_mask & prediction_changed
