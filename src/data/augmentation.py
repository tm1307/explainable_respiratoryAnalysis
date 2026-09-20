"""
Data augmentation utilities for robust respiratory sound classification.

Implements:
- SpecAugment: Time and frequency masking on spectrograms (Park et al., 2019)
- Mixup: Linear interpolation between training samples (Zhang et al., 2018)

These are applied during noise-aware training to improve robustness.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def spec_augment(
    spectrogram: torch.Tensor,
    num_freq_masks: int = 2,
    freq_mask_width: int = 10,
    num_time_masks: int = 2,
    time_mask_width: int = 25,
) -> torch.Tensor:
    """
    Apply SpecAugment to a log-mel spectrogram.

    Randomly masks contiguous bands along the frequency and time axes.
    This forces the model to not rely on any single frequency band or
    time segment, improving generalization under noise.

    Args:
        spectrogram: Input spectrogram of shape (C, n_mels, time_frames)
                     or (n_mels, time_frames).
        num_freq_masks: Number of frequency masks to apply.
        freq_mask_width: Maximum width of each frequency mask (in mel bins).
        num_time_masks: Number of time masks to apply.
        time_mask_width: Maximum width of each time mask (in frames).

    Returns:
        Augmented spectrogram with the same shape as input.
    """
    augmented = spectrogram.clone()

    # Handle both (C, F, T) and (F, T) shapes
    if augmented.dim() == 2:
        n_mels, time_frames = augmented.shape
        freq_dim, time_dim = 0, 1
    elif augmented.dim() == 3:
        _, n_mels, time_frames = augmented.shape
        freq_dim, time_dim = 1, 2
    else:
        raise ValueError(
            f"Expected 2D or 3D spectrogram, got shape {spectrogram.shape}"
        )

    # Frequency masking
    for _ in range(num_freq_masks):
        f_width = torch.randint(0, min(freq_mask_width, n_mels) + 1, (1,)).item()
        if f_width == 0:
            continue
        f_start = torch.randint(0, max(n_mels - f_width, 1), (1,)).item()
        if augmented.dim() == 2:
            augmented[f_start : f_start + f_width, :] = 0.0
        else:
            augmented[:, f_start : f_start + f_width, :] = 0.0

    # Time masking
    for _ in range(num_time_masks):
        t_width = torch.randint(0, min(time_mask_width, time_frames) + 1, (1,)).item()
        if t_width == 0:
            continue
        t_start = torch.randint(0, max(time_frames - t_width, 1), (1,)).item()
        if augmented.dim() == 2:
            augmented[:, t_start : t_start + t_width] = 0.0
        else:
            augmented[:, :, t_start : t_start + t_width] = 0.0

    return augmented


def mixup(
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Mixup augmentation between two samples.

    Creates a convex combination of two input-label pairs using a mixing
    coefficient drawn from a Beta distribution. This regularizes the model
    by training on interpolated examples.

    Args:
        x1: First input tensor (any shape, typically a spectrogram).
        y1: First label as a one-hot or soft-label tensor of shape (num_classes,).
        x2: Second input tensor (same shape as x1).
        y2: Second label tensor (same shape as y1).
        alpha: Parameter for the Beta distribution. Higher values produce
               more uniform mixing; alpha=0 means no mixing.

    Returns:
        Tuple of (mixed_x, mixed_y) with the same shapes as inputs.
    """
    if alpha <= 0.0:
        return x1, y1

    # Sample mixing coefficient from Beta(alpha, alpha)
    lam = torch.distributions.Beta(alpha, alpha).sample()

    mixed_x = lam * x1 + (1.0 - lam) * x2
    mixed_y = lam * y1 + (1.0 - lam) * y2

    return mixed_x, mixed_y


def mixup_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Mixup augmentation to an entire batch by shuffling and interpolating.

    Shuffles the batch and creates pairwise mixtures between original and
    shuffled samples, using a single lambda drawn from Beta(alpha, alpha).

    Args:
        batch_x: Batch of inputs, shape (B, ...).
        batch_y: Batch of labels as one-hot vectors, shape (B, num_classes).
        alpha: Beta distribution parameter for mixing coefficient.

    Returns:
        Tuple of (mixed_batch_x, mixed_batch_y).
    """
    if alpha <= 0.0:
        return batch_x, batch_y

    batch_size = batch_x.size(0)

    # Sample a single lambda for the entire batch
    lam = torch.distributions.Beta(alpha, alpha).sample()

    # Random permutation for pairing
    perm = torch.randperm(batch_size)

    mixed_x = lam * batch_x + (1.0 - lam) * batch_x[perm]
    mixed_y = lam * batch_y + (1.0 - lam) * batch_y[perm]

    return mixed_x, mixed_y


def to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer class labels to one-hot encoded vectors.

    Args:
        labels: Integer labels of shape (B,) with values in [0, num_classes-1].
        num_classes: Total number of classes.

    Returns:
        One-hot tensor of shape (B, num_classes) with dtype float32.
    """
    return F.one_hot(labels.long(), num_classes).float()
