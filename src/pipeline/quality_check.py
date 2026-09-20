"""
Audio quality validation module.

Checks incoming audio for common issues before it enters the
classification pipeline. Implements accept/reject logic with
reason codes for transparency.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class QualityReport:
    """Result of a quality check on an audio clip."""

    passed: bool
    """Whether the audio passed all quality checks."""

    reasons: List[str] = field(default_factory=list)
    """List of reason codes for any failed checks."""

    duration_sec: float = 0.0
    """Duration of the audio in seconds."""

    peak_amplitude: float = 0.0
    """Maximum absolute amplitude in the waveform."""

    rms_energy: float = 0.0
    """Root mean square energy of the waveform."""

    silence_fraction: float = 0.0
    """Fraction of the waveform that is near-silent."""

    clipping_fraction: float = 0.0
    """Fraction of samples near the clipping threshold."""


def check_duration(
    waveform: torch.Tensor,
    sample_rate: int,
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 30.0,
) -> Tuple[bool, str]:
    """
    Check if the audio duration is within acceptable bounds.

    Args:
        waveform: 1D audio tensor.
        sample_rate: Sample rate in Hz.
        min_duration_sec: Minimum acceptable duration.
        max_duration_sec: Maximum acceptable duration.

    Returns:
        Tuple of (passed, reason_string).
    """
    duration = waveform.shape[-1] / sample_rate
    if duration < min_duration_sec:
        return False, f"TOO_SHORT: {duration:.2f}s < {min_duration_sec}s"
    if duration > max_duration_sec:
        return False, f"TOO_LONG: {duration:.2f}s > {max_duration_sec}s"
    return True, ""


def check_amplitude(
    waveform: torch.Tensor,
    min_peak: float = 0.01,
) -> Tuple[bool, str]:
    """
    Check if the audio has sufficient amplitude (not near-silent).

    Args:
        waveform: 1D audio tensor.
        min_peak: Minimum peak amplitude to accept.

    Returns:
        Tuple of (passed, reason_string).
    """
    peak = waveform.abs().max().item()
    if peak < min_peak:
        return False, f"TOO_QUIET: peak={peak:.6f} < {min_peak}"
    return True, ""


def check_clipping(
    waveform: torch.Tensor,
    clipping_threshold: float = 0.99,
    max_clipping_fraction: float = 0.01,
) -> Tuple[bool, str]:
    """
    Detect excessive clipping (samples at or near maximum amplitude).

    Args:
        waveform: 1D audio tensor.
        clipping_threshold: Amplitude threshold to consider a sample clipped.
        max_clipping_fraction: Maximum fraction of clipped samples to accept.

    Returns:
        Tuple of (passed, reason_string).
    """
    clipped = (waveform.abs() >= clipping_threshold).float().mean().item()
    if clipped > max_clipping_fraction:
        return False, f"CLIPPED: {clipped:.4f} > {max_clipping_fraction}"
    return True, ""


def check_silence(
    waveform: torch.Tensor,
    silence_threshold: float = 0.005,
    max_silence_fraction: float = 0.9,
    frame_length: int = 1024,
) -> Tuple[bool, str]:
    """
    Detect excessive silence (large fraction of near-zero frames).

    Args:
        waveform: 1D audio tensor.
        silence_threshold: RMS threshold below which a frame is considered silent.
        max_silence_fraction: Maximum fraction of silent frames to accept.
        frame_length: Number of samples per analysis frame.

    Returns:
        Tuple of (passed, reason_string).
    """
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    num_frames = waveform.shape[0] // frame_length
    if num_frames == 0:
        return True, ""

    frames = waveform[: num_frames * frame_length].reshape(num_frames, frame_length)
    rms_per_frame = frames.pow(2).mean(dim=1).sqrt()
    silent_frames = (rms_per_frame < silence_threshold).float().mean().item()

    if silent_frames > max_silence_fraction:
        return False, f"MOSTLY_SILENT: {silent_frames:.2f} > {max_silence_fraction}"
    return True, ""


def run_quality_check(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
) -> QualityReport:
    """
    Run all quality checks on an audio waveform.

    Args:
        waveform: 1D audio tensor.
        sample_rate: Sample rate in Hz.

    Returns:
        QualityReport with pass/fail status and diagnostics.
    """
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    reasons = []

    # Duration check
    ok, reason = check_duration(waveform, sample_rate)
    if not ok:
        reasons.append(reason)

    # Amplitude check
    ok, reason = check_amplitude(waveform)
    if not ok:
        reasons.append(reason)

    # Clipping check
    ok, reason = check_clipping(waveform)
    if not ok:
        reasons.append(reason)

    # Silence check
    ok, reason = check_silence(waveform)
    if not ok:
        reasons.append(reason)

    # Compute diagnostics
    duration_sec = waveform.shape[-1] / sample_rate
    peak_amplitude = waveform.abs().max().item()
    rms_energy = waveform.pow(2).mean().sqrt().item()

    # Silence fraction
    frame_length = 1024
    num_frames = waveform.shape[0] // frame_length
    silence_fraction = 0.0
    if num_frames > 0:
        frames = waveform[: num_frames * frame_length].reshape(num_frames, frame_length)
        rms_per_frame = frames.pow(2).mean(dim=1).sqrt()
        silence_fraction = (rms_per_frame < 0.005).float().mean().item()

    # Clipping fraction
    clipping_fraction = (waveform.abs() >= 0.99).float().mean().item()

    return QualityReport(
        passed=len(reasons) == 0,
        reasons=reasons,
        duration_sec=duration_sec,
        peak_amplitude=peak_amplitude,
        rms_energy=rms_energy,
        silence_fraction=silence_fraction,
        clipping_fraction=clipping_fraction,
    )
