"""
Baseline CNN classifier with attention pooling for respiratory sound classification.

Architecture:
- 4 convolutional blocks: Conv2d → BatchNorm2d → ReLU → MaxPool2d
- Channel progression: 32 → 64 → 128 → 256
- Attention pooling head: learns weighted average across time frames
- Output: logits for 4 classes (Normal, Crackle, Wheeze, Both)

Designed with named layers for Grad-CAM compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionPooling(nn.Module):
    """
    Attention-based temporal pooling.

    Learns a set of attention weights over the time dimension,
    then computes a weighted average of the feature vectors.
    This is more expressive than global average pooling because
    it can learn to focus on the most informative time segments
    (e.g., the part of the respiratory cycle containing the wheeze).
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.Tanh(),
            nn.Linear(in_features // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, C, Freq, T) from the CNN backbone.

        Returns:
            Pooled feature vector of shape (B, C*Freq).
        """
        B, C, Freq, T = x.shape

        # Reshape to (B, T, C*Freq) — treat each time frame as a feature vector
        x_permuted = x.permute(0, 3, 1, 2).reshape(B, T, C * Freq)

        # Compute attention weights: (B, T, 1)
        attn_weights = self.attention(x_permuted)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum: (B, C*Freq)
        pooled = (x_permuted * attn_weights).sum(dim=1)

        return pooled


class ConvBlock(nn.Module):
    """Single convolutional block: Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class BaselineCNN(nn.Module):
    """
    4-block CNN with attention pooling for respiratory sound classification.

    Input: Log-Mel spectrogram of shape (B, 1, n_mels, time_frames)
    Output: Logits of shape (B, num_classes)

    The feature extraction backbone is stored as `self.features` (a nn.Sequential),
    which is the standard hook point for Grad-CAM (specifically the last conv block).
    """

    def __init__(
        self,
        num_classes: int = 4,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        n_mels: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.channels = channels

        # Build the convolutional backbone as a Sequential for easy Grad-CAM hooking
        blocks = []
        in_ch = 1  # single-channel log-mel input
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)

        # After 4 max-pool operations with stride 2, the spatial dimensions are
        # reduced by a factor of 2^4 = 16 in each axis.
        # n_mels=128 → 128/16 = 8
        # time_frames varies but follows the same reduction
        self.reduced_freq = n_mels // (2 ** len(channels))

        # Attention pooling operates over the time dimension
        # Input features per time step: channels[-1] * reduced_freq
        pooling_features = channels[-1] * self.reduced_freq
        self.attention_pool = AttentionPooling(pooling_features)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooling_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Log-Mel spectrogram of shape (B, 1, n_mels, time_frames).

        Returns:
            Logits of shape (B, num_classes).
        """
        # Extract features: (B, 1, 128, T) → (B, 256, 8, T//16)
        features = self.features(x)

        # Attention pooling: (B, 256, 8, T//16) → (B, 256*8)
        pooled = self.attention_pool(features)

        # Classify: (B, 256*8) → (B, num_classes)
        logits = self.classifier(pooled)

        return logits

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the feature maps from the last conv block (for Grad-CAM).

        Args:
            x: Log-Mel spectrogram of shape (B, 1, n_mels, time_frames).

        Returns:
            Feature maps of shape (B, channels[-1], reduced_freq, reduced_time).
        """
        return self.features(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in respiratory sound classification.

    FocalLoss = -alpha_c * (1 - p_c)^gamma * log(p_c)

    When gamma=0, this reduces to standard cross-entropy.
    Higher gamma values down-weight easy examples and focus training
    on hard, misclassified samples — critical for the imbalanced ICBHI
    dataset where Normal cycles vastly outnumber Wheeze and Both.

    Args:
        alpha: Per-class weights (tensor of shape (num_classes,)) or scalar.
               If None, all classes are weighted equally.
        gamma: Focusing parameter. gamma=0 is standard CE; gamma=2 is typical.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model output of shape (B, num_classes).
            targets: Integer class labels of shape (B,).

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-sample loss (B,).
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=logits.size(1)).float()

        # Gather the predicted probability for the true class
        pt = (probs * targets_one_hot).sum(dim=1)  # shape: (B,)

        # Compute focal weight
        focal_weight = (1.0 - pt) ** self.gamma

        # Compute cross-entropy component: -log(pt)
        ce_loss = -torch.log(pt + 1e-8)

        # Apply class-specific alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.long())
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
