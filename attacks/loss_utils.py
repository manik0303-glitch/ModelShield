import torch
import torch.nn.functional as F


def compute_attack_loss(logits: torch.Tensor, labels: torch.Tensor, multi_label: bool = True) -> torch.Tensor:
    """
    Unified loss computation for adversarial attacks and gradient evaluation.

    For multi-label recognition (e.g. RAM++):
        Computes binary cross entropy with logits against multi-hot binary labels.
    For single-label classification (e.g. CIFAR-10):
        Computes standard cross-entropy loss against integer class labels.

    Args:
        logits: Model output logits tensor (B, num_classes)
        labels: Ground-truth labels tensor.
                For multi_label=True, shape is (B, num_classes) with float/long {0, 1}.
                For multi_label=False, shape is (B,) with integer class indices.
        multi_label: Whether to use BCE with logits (True) or Cross Entropy (False).

    Returns:
        Scalar loss tensor.
    """
    if multi_label:
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    return F.cross_entropy(logits, labels.long())
