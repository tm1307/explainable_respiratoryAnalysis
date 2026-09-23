"""
SHAP Explainability for Respiratory Sound Classifier.

Uses shap.GradientExplainer to compute SHAP values over the Log-Mel spectrogram
input, identifying which time-frequency regions drove the model's prediction.

Key outputs:
  - 2D SHAP heatmap (same shape as input spectrogram) — overlay on spectrogram
  - Per-frequency-band mean |SHAP| — bar chart of low/mid/high band importance
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    Wrapper around shap.GradientExplainer for CNN-based spectrogram classifiers.

    Works by treating each pixel of the input mel spectrogram as a feature.
    GradientExplainer uses expected gradients (a smoothed version of integrated
    gradients) which is fast and works natively with PyTorch autograd.

    Usage:
        explainer = SHAPExplainer(model, background_tensor)
        shap_map, band_importance = explainer.explain(input_tensor, class_idx)
    """

    # Frequency band names and their mel-bin ranges (for 128 mel bins)
    BAND_DEFINITIONS = {
        "Low (50–500 Hz)":  (0,  32),   # bins 0–31
        "Mid (500–2k Hz)":  (32, 80),   # bins 32–79
        "High (2k–8k Hz)":  (80, 128),  # bins 80–127
    }

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        device: str = "cpu",
    ):
        """
        Args:
            model: Trained CNN (BaselineCNN or compatible).
            background: Reference background tensor of shape (N, 1, n_mels, T).
                        Typically 5–20 random/silence samples. Used as the
                        SHAP baseline distribution.
            device: 'cpu' or 'cuda'.
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "shap is not installed. Run: pip install shap"
            )

        # shap.GradientExplainer only supports CPU and CUDA — not MPS.
        self.device = torch.device("cpu")
        self.model = model.cpu().eval()
        self.background = background.cpu()

        # Build GradientExplainer on the background distribution
        self._explainer = shap.GradientExplainer(
            self.model, self.background
        )

    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute SHAP values for a single input spectrogram.

        Args:
            input_tensor: Shape (1, 1, n_mels, T) — single sample. Must be on CPU.
            class_idx: Target class index to explain.

        Returns:
            shap_map (np.ndarray): 2D array of shape (n_mels, T), normalised
                                   absolute SHAP values in [0, 1].
            band_importance (Dict[str, float]): Mean |SHAP| per frequency band.
        """
        input_t = input_tensor.cpu()

        # shap_values returns a list of arrays, one per output class.
        # Each array has shape (n_samples, 1, n_mels, T).
        # We index [class_idx] for the target class, [0] for batch dim, [0] for channel.
        shap_values = self._explainer.shap_values(input_t)

        if isinstance(shap_values, list):
            sv = shap_values[class_idx][0, 0]  # shape: (n_mels, T)
        else:
            # Newer shap returns (batch, channel, n_mels, T, n_classes)
            sv = shap_values[0, 0, :, :, class_idx]

        # Absolute SHAP → normalise to [0, 1]
        abs_sv = np.abs(sv)
        sv_min, sv_max = abs_sv.min(), abs_sv.max()
        if sv_max - sv_min > 1e-8:
            shap_map = (abs_sv - sv_min) / (sv_max - sv_min)
        else:
            shap_map = np.zeros_like(abs_sv)

        band_importance = self._compute_band_importance(abs_sv)

        return shap_map, band_importance

    def _compute_band_importance(self, abs_shap: np.ndarray) -> Dict[str, float]:
        """
        Compute mean |SHAP| per frequency band for the bar chart.

        Args:
            abs_shap: 2D array (n_mels, T) of absolute SHAP values.

        Returns:
            Dict mapping band name → mean absolute SHAP score (normalised).
        """
        n_mels = abs_shap.shape[0]
        total_mean = abs_shap.mean() + 1e-8

        result = {}
        for band_name, (lo, hi) in self.BAND_DEFINITIONS.items():
            lo_clipped = min(lo, n_mels)
            hi_clipped = min(hi, n_mels)
            if lo_clipped >= hi_clipped:
                result[band_name] = 0.0
            else:
                band_mean = abs_shap[lo_clipped:hi_clipped, :].mean()
                result[band_name] = float(band_mean / total_mean)

        return result

    @staticmethod
    def make_background_from_silence(
        n_samples: int = 10,
        n_mels: int = 128,
        time_frames: int = 157,
        noise_level: float = 0.01,
    ) -> torch.Tensor:
        """
        Generate a simple background distribution from near-silence + tiny noise.

        This is the recommended fallback when no real training data is available.
        Each background sample is close to zero (silence baseline) with a small
        amount of Gaussian noise to prevent degenerate gradient estimates.

        Args:
            n_samples: Number of background reference samples.
            n_mels: Mel bin count (must match model input).
            time_frames: Time frame count (must match model input).
            noise_level: Std-dev of additive Gaussian noise.

        Returns:
            Tensor of shape (n_samples, 1, n_mels, time_frames).
        """
        background = torch.randn(n_samples, 1, n_mels, time_frames) * noise_level
        return background

    @staticmethod
    def make_background_from_batch(
        mel_batch: torch.Tensor, max_samples: int = 20
    ) -> torch.Tensor:
        """
        Use a subset of real (or synthetic) mel spectrograms as background.

        Using real data as background gives more meaningful SHAP baselines
        because the expected-gradient estimate reflects the actual data manifold
        rather than silence.

        Args:
            mel_batch: Tensor of shape (N, 1, n_mels, T).
            max_samples: Cap on number of background samples (for speed).

        Returns:
            Tensor of shape (min(N, max_samples), 1, n_mels, T).
        """
        n = min(mel_batch.shape[0], max_samples)
        indices = torch.randperm(mel_batch.shape[0])[:n]
        return mel_batch[indices].detach()


def compute_shap_heatmap_fast(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    n_background: int = 8,
    device: str = "cpu",
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """
    Convenience function: build a background from near-silence and compute SHAP.

    Designed for the Streamlit dashboard where no pre-built background exists.
    Uses silence-based background for speed (< 2s on CPU for a 128×157 input).

    Args:
        model: Trained CNN model.
        input_tensor: Shape (1, 1, n_mels, T).
        class_idx: Class to explain.
        n_background: Number of silence background samples.
        device: 'cpu' or 'cuda'.

    Returns:
        (shap_map, band_importance) or (None, None) if shap is not installed.
    """
    if not SHAP_AVAILABLE:
        return None, None

    input_cpu = input_tensor.cpu()
    _, _, n_mels, T = input_cpu.shape
    background = SHAPExplainer.make_background_from_silence(
        n_samples=n_background, n_mels=n_mels, time_frames=T
    )

    try:
        explainer = SHAPExplainer(model, background, device="cpu")
        return explainer.explain(input_cpu, class_idx)
    except Exception as e:
        print(f"[shap_explainer] SHAP computation failed: {e}", flush=True)
        return None, None
