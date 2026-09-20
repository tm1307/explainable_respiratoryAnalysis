"""
End-to-end inference pipeline for respiratory sound classification.

Pipeline stages:
1. Audio quality check (accept/reject with reasons)
2. Feature extraction (Log-Mel spectrogram)
3. Classification (BaselineCNN with softmax)
4. Explainability (Grad-CAM heatmap)
5. Reliability assessment (JRI-style flag)

Output schema: {label, confidence, class_probabilities, heatmap, reliability_flag}
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from src.pipeline.quality_check import run_quality_check, QualityReport
from src.features.mel_features import MelSpectrogramExtractor, pad_or_truncate
from src.explainability.gradcam import GradCAM
from src.config import PipelineConfig, get_default_config


@dataclass
class PredictionResult:
    """Complete output from the inference pipeline."""

    label: str
    """Predicted class name (Normal, Crackle, Wheeze, Both)."""

    label_index: int
    """Predicted class index (0-3)."""

    confidence: float
    """Predicted probability for the top class."""

    class_probabilities: Dict[str, float] = field(default_factory=dict)
    """Probabilities for all classes."""

    heatmap: Optional[np.ndarray] = None
    """Grad-CAM explanation heatmap (2D numpy array), or None if not computed."""

    reliability_flag: str = "unknown"
    """Reliability assessment: 'high', 'medium', 'low', or 'unknown'."""

    quality_report: Optional[QualityReport] = None
    """Audio quality diagnostics."""


class InferencePipeline:
    """
    End-to-end inference pipeline wiring together all modules.

    Usage:
        pipeline = InferencePipeline(model, config)
        result = pipeline.predict(waveform)
    """

    CLASS_NAMES = ("Normal", "Crackle", "Wheeze", "Both")

    def __init__(
        self,
        model: nn.Module,
        config: Optional[PipelineConfig] = None,
        device: str = "cpu",
        enable_explainability: bool = True,
    ):
        """
        Args:
            model: Trained BaselineCNN (or compatible) model.
            config: Pipeline configuration. Uses defaults if None.
            device: Inference device ('cpu' or 'cuda').
            enable_explainability: Whether to compute Grad-CAM heatmaps.
        """
        self.config = config or get_default_config()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=self.config.audio.sample_rate,
            n_fft=self.config.mel.n_fft,
            hop_length=self.config.mel.hop_length,
            n_mels=self.config.mel.n_mels,
            f_min=self.config.mel.f_min,
            f_max=self.config.mel.f_max,
            log_offset=self.config.mel.log_offset,
        )

        self.enable_explainability = enable_explainability
        self.gradcam = None
        if enable_explainability and hasattr(model, "features"):
            # Hook Grad-CAM onto the last convolutional block
            target_layer = model.features[-1]
            self.gradcam = GradCAM(model, target_layer)

    def predict(
        self,
        waveform: torch.Tensor,
        run_quality_checks: bool = True,
    ) -> PredictionResult:
        """
        Run the full inference pipeline on a single audio waveform.

        Args:
            waveform: 1D audio tensor (raw waveform at the expected sample rate).
            run_quality_checks: Whether to perform audio quality validation.

        Returns:
            PredictionResult with label, confidence, heatmap, and reliability flag.
        """
        # Step 1: Quality check
        quality_report = None
        if run_quality_checks:
            quality_report = run_quality_check(
                waveform, self.config.audio.sample_rate
            )
            if not quality_report.passed:
                return PredictionResult(
                    label="REJECTED",
                    label_index=-1,
                    confidence=0.0,
                    reliability_flag="rejected",
                    quality_report=quality_report,
                )

        # Step 2: Pad/truncate to fixed duration
        target_length = int(
            self.config.audio.sample_rate * self.config.audio.duration_sec
        )
        waveform = pad_or_truncate(waveform.squeeze(), target_length)

        # Step 3: Extract Log-Mel spectrogram
        mel_spec = self.mel_extractor.extract(waveform)  # (1, n_mels, T)
        input_tensor = mel_spec.unsqueeze(0).to(self.device)  # (1, 1, n_mels, T)

        # Step 4: Classification
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        class_probabilities = {
            name: probs[i].item() for i, name in enumerate(self.CLASS_NAMES)
        }

        # Step 5: Explainability (Grad-CAM)
        heatmap = None
        if self.gradcam is not None:
            try:
                # Grad-CAM needs gradients, so we re-run with grad enabled
                heatmap_tensor = self.gradcam.generate(
                    input_tensor.clone().requires_grad_(False),
                    target_class=pred_idx,
                )
                heatmap = heatmap_tensor.cpu().numpy()
            except Exception:
                heatmap = None

        # Step 6: Reliability flag based on confidence
        reliability_flag = self._assess_reliability(confidence)

        return PredictionResult(
            label=self.CLASS_NAMES[pred_idx],
            label_index=pred_idx,
            confidence=confidence,
            class_probabilities=class_probabilities,
            heatmap=heatmap,
            reliability_flag=reliability_flag,
            quality_report=quality_report,
        )

    def _assess_reliability(self, confidence: float) -> str:
        """
        Assess prediction reliability based on confidence.

        This is a simplified heuristic. In the full system, this would
        incorporate the JRI score from the reliability metrics module.

        Args:
            confidence: Top-class probability.

        Returns:
            'high', 'medium', or 'low'.
        """
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"

    def cleanup(self):
        """Remove hooks and free resources."""
        if self.gradcam is not None:
            self.gradcam.remove_hooks()
