"""
Tests for Mel spectrogram feature extraction module.
"""

import torch
import math
from src.features.mel_features import MelSpectrogramExtractor, pad_or_truncate

def test_extract_shape():
    torch.manual_seed(42)
    extractor = MelSpectrogramExtractor(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128)
    waveform = torch.randn(80000)
    mel = extractor.extract(waveform)
    
    expected_time_frames = math.floor((80000 - 1024) / 512) + 1
    assert mel.shape == (1, 128, expected_time_frames)

def test_extract_batch_shape():
    torch.manual_seed(42)
    extractor = MelSpectrogramExtractor(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128)
    waveforms = torch.randn(4, 80000)
    mel = extractor.extract_batch(waveforms)
    
    expected_time_frames = math.floor((80000 - 1024) / 512) + 1
    assert mel.shape == (4, 1, 128, expected_time_frames)

def test_extract_values_finite():
    torch.manual_seed(42)
    extractor = MelSpectrogramExtractor()
    waveform = torch.randn(80000)
    mel = extractor.extract(waveform)
    
    assert torch.isfinite(mel).all()

def test_extract_deterministic():
    torch.manual_seed(42)
    extractor = MelSpectrogramExtractor()
    waveform = torch.randn(80000)
    
    mel1 = extractor.extract(waveform)
    mel2 = extractor.extract(waveform)
    
    assert torch.allclose(mel1, mel2)

def test_pad_or_truncate_padding():
    waveform = torch.ones(5000)
    padded = pad_or_truncate(waveform, 8000)
    
    assert padded.shape[0] == 8000
    assert torch.all(padded[:5000] == 1.0)
    assert torch.all(padded[5000:] == 0.0)

def test_pad_or_truncate_truncation():
    waveform = torch.ones(10000)
    truncated = pad_or_truncate(waveform, 8000)
    
    assert truncated.shape[0] == 8000
    assert torch.all(truncated == 1.0)

def test_pad_or_truncate_exact():
    waveform = torch.ones(8000)
    exact = pad_or_truncate(waveform, 8000)
    
    assert exact.shape[0] == 8000
    assert torch.all(exact == 1.0)

def test_log_offset_prevents_log_zero():
    extractor = MelSpectrogramExtractor(log_offset=1e-6)
    waveform = torch.zeros(80000)
    mel = extractor.extract(waveform)
    
    assert torch.isfinite(mel).all()
    assert not torch.isinf(mel).any()
