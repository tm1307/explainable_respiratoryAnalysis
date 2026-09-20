"""
Central configuration for the Explainable Respiratory Analysis pipeline.

All hyperparameters, paths, and experimental settings are defined here as
frozen dataclasses so they are immutable during a run and easy to serialize
for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class AudioConfig:
    """Audio processing parameters."""

    sample_rate: int = 16_000
    """Target sample rate in Hz. All audio is resampled to this rate."""

    duration_sec: float = 5.0
    """Fixed clip duration in seconds. Shorter clips are padded, longer are truncated."""

    n_samples: int = 80_000  # sample_rate * duration_sec
    """Number of samples per clip (derived from sample_rate * duration_sec)."""


@dataclass(frozen=True)
class MelConfig:
    """Log-Mel spectrogram parameters."""

    n_fft: int = 1024
    """FFT window size."""

    hop_length: int = 512
    """Hop length between STFT frames."""

    n_mels: int = 128
    """Number of Mel filterbank channels."""

    f_min: float = 50.0
    """Minimum frequency for Mel filterbank (Hz)."""

    f_max: float = 8000.0
    """Maximum frequency for Mel filterbank (Hz)."""

    log_offset: float = 1e-6
    """Small constant added before log to avoid log(0)."""


@dataclass(frozen=True)
class NoiseConfig:
    """Noise injection parameters for robustness evaluation."""

    snr_levels_db: Tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0, -5.0)
    """SNR levels in dB for the controlled noise sweep."""

    snr_train_range: Tuple[float, float] = (0.0, 20.0)
    """SNR range (min, max) in dB for randomized noise-aware training augmentation."""


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture parameters."""

    num_classes: int = 4
    """Number of output classes: Normal, Crackle, Wheeze, Both."""

    class_names: Tuple[str, ...] = ("Normal", "Crackle", "Wheeze", "Both")
    """Human-readable class labels."""

    cnn_channels: Tuple[int, ...] = (32, 64, 128, 256)
    """Channel sizes for each CNN block in the baseline model."""

    dropout: float = 0.3
    """Dropout probability."""

    focal_loss_alpha: float = 1.0
    """Alpha parameter for focal loss."""

    focal_loss_gamma: float = 2.0
    """Gamma (focusing) parameter for focal loss."""


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 10
    """Early stopping patience (epochs)."""

    k_folds: int = 5
    """Number of folds for patient-independent cross-validation."""

    seed: int = 42
    """Global random seed for reproducibility."""


@dataclass(frozen=True)
class AugmentConfig:
    """Data augmentation parameters."""

    spec_augment_freq_masks: int = 2
    spec_augment_freq_width: int = 10
    spec_augment_time_masks: int = 2
    spec_augment_time_width: int = 25

    mixup_alpha: float = 0.4
    """Beta distribution parameter for Mixup."""


@dataclass(frozen=True)
class ExplainConfig:
    """Explainability parameters."""

    gradcam_target_layer: str = "features.3"
    """Target convolutional layer name for Grad-CAM."""

    ig_n_steps: int = 50
    """Number of interpolation steps for Integrated Gradients."""

    top_k_percent: float = 0.1
    """Top-k fraction of attributed regions for IoU stability metric."""


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration metric parameters."""

    n_bins: int = 15
    """Number of bins for Expected Calibration Error."""


@dataclass(frozen=True)
class PipelineConfig:
    """Master configuration aggregating all sub-configs."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # Paths
    data_dir: str = "data/"
    icbhi_dir: str = "data/icbhi/"
    noise_dir: str = "data/noise/"
    output_dir: str = "outputs/"
    checkpoint_dir: str = "outputs/checkpoints/"


def get_default_config() -> PipelineConfig:
    """Return the default pipeline configuration."""
    return PipelineConfig()
