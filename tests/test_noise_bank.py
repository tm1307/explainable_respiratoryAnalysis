import pytest
import torch
import math
import torchaudio
import os
from src.data.noise_bank import (
    compute_signal_power,
    mix_at_snr,
    measure_snr,
    NoiseBank
)

def setup_module():
    torch.manual_seed(42)

def test_compute_signal_power():
    # sine wave with amplitude A has power A^2/2
    A = 2.0
    t = torch.linspace(0, 10 * math.pi, 100000)
    signal = A * torch.sin(t)
    power = compute_signal_power(signal)
    assert abs(power.item() - (A**2 / 2)) < 1e-2

def test_mix_at_snr_accuracy():
    clean = torch.sin(torch.linspace(0, 10 * math.pi, 1000))
    noise = torch.randn(1000)
    
    noisy = mix_at_snr(clean, noise, 10.0)
    noise_component = noisy - clean
    actual_snr = measure_snr(clean, noise_component)
    
    assert abs(actual_snr - 10.0) < 0.5

def test_mix_at_snr_multiple_levels():
    clean = torch.sin(torch.linspace(0, 10 * math.pi, 1000))
    noise = torch.randn(1000)
    
    levels = [20.0, 10.0, 5.0, 0.0, -5.0]
    for lvl in levels:
        noisy = mix_at_snr(clean, noise, lvl)
        noise_comp = noisy - clean
        actual = measure_snr(clean, noise_comp)
        assert abs(actual - lvl) < 0.5

def test_noise_shorter_than_signal():
    clean = torch.randn(1000)
    noise = torch.randn(300)
    
    noisy = mix_at_snr(clean, noise, 10.0)
    assert noisy.shape == clean.shape
    
    noise_comp = noisy - clean
    actual = measure_snr(clean, noise_comp)
    assert abs(actual - 10.0) < 0.5

def test_noise_longer_than_signal():
    clean = torch.randn(500)
    noise = torch.randn(1000)
    
    noisy = mix_at_snr(clean, noise, 10.0)
    assert noisy.shape == clean.shape
    
    noise_comp = noisy - clean
    actual = measure_snr(clean, noise_comp)
    assert abs(actual - 10.0) < 0.5

def test_zero_power_edge_cases():
    clean = torch.randn(1000)
    noise = torch.zeros(1000)
    noisy = mix_at_snr(clean, noise, 10.0)
    assert torch.allclose(noisy, clean)
    
    clean_zero = torch.zeros(1000)
    noise_randn = torch.randn(1000)
    noisy_zero = mix_at_snr(clean_zero, noise_randn, 10.0)
    assert torch.allclose(noisy_zero, clean_zero)

def test_snr_sweep(tmp_path):
    noise_dir = tmp_path / "noises"
    noise_dir.mkdir()
    
    # create synthetic wav
    noise_wav = torch.randn(1, 16000)
    torchaudio.save(str(noise_dir / "noise1.wav"), noise_wav, 16000)
    
    bank = NoiseBank(str(noise_dir), sample_rate=16000)
    clean = torch.randn(16000)
    
    levels = [10.0, 0.0]
    results = bank.get_snr_sweep(clean, levels)
    
    assert list(results.keys()) == levels
    for lvl in levels:
        noisy = results[lvl]
        noise_comp = noisy - clean
        actual = measure_snr(clean, noise_comp)
        assert abs(actual - lvl) < 0.5
