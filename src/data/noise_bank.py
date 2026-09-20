import torch
import torchaudio
import os
import glob
import math
from typing import Optional, List, Dict

def compute_signal_power(signal: torch.Tensor) -> torch.Tensor:
    """Compute the mean squared power of a signal."""
    return torch.mean(signal ** 2)

def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, target_snr_db: float) -> torch.Tensor:
    """Mix clean audio with noise at an exact target SNR."""
    # Ensure 1D
    clean = clean.squeeze()
    noise = noise.squeeze()
    
    P_x = compute_signal_power(clean)
    P_n = compute_signal_power(noise)
    
    if P_n.item() == 0.0:
        return clean
    if P_x.item() == 0.0:
        return clean
        
    if len(noise) < len(clean):
        # Tile
        repeats = (len(clean) // len(noise)) + 1
        noise = noise.repeat(repeats)
    
    if len(noise) > len(clean):
        # Truncate
        noise = noise[:len(clean)]
        
    noise_scaled = noise * torch.sqrt(P_x / (P_n * (10 ** (target_snr_db / 10))))
    x_noisy = clean + noise_scaled
    return x_noisy

def measure_snr(clean: torch.Tensor, noise_component: torch.Tensor) -> float:
    """Measure the actual SNR between signal and noise in dB."""
    P_x = compute_signal_power(clean)
    P_n = compute_signal_power(noise_component)
    if P_n.item() == 0.0:
        return float('inf')
    return 10 * math.log10(P_x.item() / P_n.item())

class NoiseBank:
    """Loads noise clips from a directory and mixes them at target SNRs."""
    def __init__(self, noise_dir: str, sample_rate: int = 16000):
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.noises = []
        
        # load all .wav files from noise_dir using torchaudio
        wav_files = glob.glob(os.path.join(noise_dir, "*.wav"))
        for f in wav_files:
            waveform, sr = torchaudio.load(f)
            self.noises.append(waveform.squeeze())
            
    def add_noise_at_snr(self, clean: torch.Tensor, target_snr_db: float, noise_index: Optional[int] = None) -> torch.Tensor:
        """Pick a random (or specified) noise clip and mix at target SNR."""
        if not self.noises:
            return clean
            
        if noise_index is None:
            noise_index = torch.randint(0, len(self.noises), (1,)).item()
            
        noise = self.noises[noise_index]
        return mix_at_snr(clean, noise, target_snr_db)
        
    def get_snr_sweep(self, clean: torch.Tensor, snr_levels: List[float]) -> Dict[float, torch.Tensor]:
        """Return dict mapping each SNR level to the noisy version."""
        results = {}
        for snr in snr_levels:
            results[snr] = self.add_noise_at_snr(clean, snr)
        return results
