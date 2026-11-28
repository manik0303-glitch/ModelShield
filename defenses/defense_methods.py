import torch


def identity_defense(images):
    """
    No defense.
    Used as the baseline for comparison.
    """
    return images


def quantization_defense(images, levels=16):
    """
    Reduce the number of discrete pixel levels.

    Input:
        images: Tensor with pixel values in [0, 1]
        levels: Number of discrete levels

    Output:
        Quantized image tensor in [0, 1]
    """

    if levels < 2:
        raise ValueError("levels must be at least 2")

    images = images.clamp(0, 1)

    defended_images = (
        torch.round(images * (levels - 1))
        / (levels - 1)
    )

    return defended_images