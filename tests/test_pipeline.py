"""
Tests for the quality check and inference pipeline modules.
Uses synthetic data only.
"""

import torch
import pytest
from src.pipeline.quality_check import (
    run_quality_check,
    check_duration,
    check_amplitude,
    check_clipping,
    check_silence,
    QualityReport,
)
from src.pipeline.inference import InferencePipeline, PredictionResult
from src.models.baseline_cnn import BaselineCNN


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


# ===== Quality Check Tests =====

class TestQualityCheck:

    def test_normal_audio_passes(self):
        """Normal audio with reasonable amplitude passes all checks."""
        waveform = torch.randn(16000 * 5) * 0.3  # 5 seconds, moderate amplitude
        report = run_quality_check(waveform, sample_rate=16000)
        assert report.passed
        assert len(report.reasons) == 0

    def test_too_short_fails(self):
        """Audio shorter than minimum duration is rejected."""
        waveform = torch.randn(100) * 0.3  # Very short
        report = run_quality_check(waveform, sample_rate=16000)
        assert not report.passed
        assert any("TOO_SHORT" in r for r in report.reasons)

    def test_silent_audio_fails(self):
        """Near-silent audio is rejected."""
        waveform = torch.zeros(16000 * 5)  # Pure silence
        report = run_quality_check(waveform, sample_rate=16000)
        assert not report.passed
        assert any("TOO_QUIET" in r for r in report.reasons)

    def test_clipped_audio_fails(self):
        """Heavily clipped audio is rejected."""
        waveform = torch.ones(16000 * 5)  # All samples at max
        report = run_quality_check(waveform, sample_rate=16000)
        assert not report.passed
        assert any("CLIPPED" in r for r in report.reasons)

    def test_diagnostics_populated(self):
        """Quality report contains all diagnostic fields."""
        waveform = torch.randn(16000 * 3) * 0.3
        report = run_quality_check(waveform, sample_rate=16000)
        assert report.duration_sec > 0
        assert report.peak_amplitude > 0
        assert report.rms_energy > 0
        assert 0 <= report.silence_fraction <= 1
        assert 0 <= report.clipping_fraction <= 1

    def test_check_duration_exact_bounds(self):
        """Duration check with exact boundary values."""
        # Exactly at min
        ok, _ = check_duration(torch.randn(8000), 16000, min_duration_sec=0.5)
        assert ok
        # Below min
        ok, _ = check_duration(torch.randn(7999), 16000, min_duration_sec=0.5)
        assert not ok

    def test_2d_waveform_handled(self):
        """Quality check handles (1, N) shaped tensors."""
        waveform = torch.randn(1, 16000 * 3) * 0.3
        report = run_quality_check(waveform, sample_rate=16000)
        assert isinstance(report, QualityReport)


# ===== Inference Pipeline Tests =====

class TestInferencePipeline:

    def _get_pipeline(self):
        model = BaselineCNN(num_classes=4, n_mels=128)
        pipeline = InferencePipeline(
            model, device="cpu", enable_explainability=True
        )
        return pipeline

    def test_predict_returns_result(self):
        """Pipeline returns a PredictionResult for valid audio."""
        pipeline = self._get_pipeline()
        waveform = torch.randn(16000 * 5) * 0.3
        result = pipeline.predict(waveform)

        assert isinstance(result, PredictionResult)
        assert result.label in ("Normal", "Crackle", "Wheeze", "Both")
        assert 0 <= result.confidence <= 1
        assert result.label_index in range(4)
        pipeline.cleanup()

    def test_predict_has_probabilities(self):
        """Result contains probabilities for all 4 classes."""
        pipeline = self._get_pipeline()
        waveform = torch.randn(16000 * 5) * 0.3
        result = pipeline.predict(waveform)

        assert len(result.class_probabilities) == 4
        assert all(0 <= p <= 1 for p in result.class_probabilities.values())
        total = sum(result.class_probabilities.values())
        assert abs(total - 1.0) < 0.01  # Probabilities sum to ~1
        pipeline.cleanup()

    def test_predict_has_heatmap(self):
        """Result contains a Grad-CAM heatmap when explainability is enabled."""
        pipeline = self._get_pipeline()
        waveform = torch.randn(16000 * 5) * 0.3
        result = pipeline.predict(waveform)

        assert result.heatmap is not None
        assert result.heatmap.ndim == 2  # 2D heatmap
        pipeline.cleanup()

    def test_predict_without_explainability(self):
        """Pipeline works without Grad-CAM."""
        model = BaselineCNN(num_classes=4, n_mels=128)
        pipeline = InferencePipeline(
            model, device="cpu", enable_explainability=False
        )
        waveform = torch.randn(16000 * 5) * 0.3
        result = pipeline.predict(waveform)

        assert result.heatmap is None
        assert result.label in ("Normal", "Crackle", "Wheeze", "Both")
        pipeline.cleanup()

    def test_reject_bad_audio(self):
        """Pipeline rejects audio that fails quality checks."""
        pipeline = self._get_pipeline()
        waveform = torch.zeros(16000 * 5)  # Silent audio
        result = pipeline.predict(waveform)

        assert result.label == "REJECTED"
        assert result.reliability_flag == "rejected"
        assert result.quality_report is not None
        assert not result.quality_report.passed
        pipeline.cleanup()

    def test_skip_quality_check(self):
        """Pipeline processes audio without quality checks when disabled."""
        pipeline = self._get_pipeline()
        waveform = torch.zeros(16000 * 5) + 0.001  # Very quiet but not zero
        result = pipeline.predict(waveform, run_quality_checks=False)

        # Should still produce a prediction (not rejected)
        assert result.label != "REJECTED"
        pipeline.cleanup()

    def test_reliability_flag_values(self):
        """Reliability flag is one of the expected values."""
        pipeline = self._get_pipeline()
        waveform = torch.randn(16000 * 5) * 0.3
        result = pipeline.predict(waveform)

        assert result.reliability_flag in ("high", "medium", "low", "unknown", "rejected")
        pipeline.cleanup()
