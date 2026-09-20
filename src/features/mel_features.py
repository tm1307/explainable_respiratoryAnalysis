"""
Mel spectrogram feature extraction module.
"""

import torch
import torchaudio
import math

class MelSpectrogramExtractor:
    """
    Extracts Log-Mel spectrograms from raw audio waveforms.
    """
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 512, 
                 n_mels: int = 128, f_min: float = 50.0, f_max: float = 8000.0, log_offset: float = 1e-6):
        """
        Initialize the extractor.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.log_offset = log_offset
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            center=False,
            power=2.0
        )

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract Log-Mel spectrogram from a 1D waveform tensor.
        
        The time_frames dimension is calculated as: floor((num_samples - n_fft) / hop_length) + 1.
        
        Args:
            waveform (torch.Tensor): 1D waveform tensor (num_samples).
            
        Returns:
            torch.Tensor: Log-Mel spectrogram of shape (1, n_mels, time_frames).
        """
        if waveform.dim() != 1:
            raise ValueError("Input waveform must be a 1D tensor.")
        
        # Add channel dimension
        waveform = waveform.unsqueeze(0)
        
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = torch.log(mel_spec + self.log_offset)
        return log_mel_spec

    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Extract Log-Mel spectrograms from a batch of waveforms.
        
        Args:
            waveforms (torch.Tensor): 2D waveform tensor (B, num_samples).
            
        Returns:
            torch.Tensor: Log-Mel spectrograms of shape (B, 1, n_mels, time_frames).
        """
        if waveforms.dim() != 2:
            raise ValueError("Input waveforms must be a 2D tensor (B, num_samples).")
            
        mel_spec = self.mel_transform(waveforms)
        log_mel_spec = torch.log(mel_spec + self.log_offset)
        
        # Add channel dimension -> (B, 1, n_mels, time_frames)
        return log_mel_spec.unsqueeze(1)


def pad_or_truncate(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pad with zeros if shorter, truncate if longer. Works on 1D tensors.
    
    Args:
        waveform (torch.Tensor): 1D waveform tensor.
        target_length (int): Desired number of samples.
        
    Returns:
        torch.Tensor: Waveform of length target_length.
    """
    if waveform.dim() != 1:
        raise ValueError("Input waveform must be a 1D tensor.")
        
    current_length = waveform.shape[0]
    if current_length == target_length:
        return waveform
    elif current_length > target_length:
        return waveform[:target_length]
    else:
        padding = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, padding), "constant", 0.0)


def load_and_preprocess(filepath: str, sample_rate: int = 16000, duration_sec: float = 5.0) -> torch.Tensor:
    """
    Load audio file with torchaudio, resample if needed, convert to mono if stereo, 
    pad/truncate to fixed length.
    
    Args:
        filepath (str): Path to audio file.
        sample_rate (int): Target sample rate.
        duration_sec (float): Target duration in seconds.
        
    Returns:
        torch.Tensor: 1D preprocessed waveform tensor.
    """
    waveform, sr = torchaudio.load(filepath)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
        
    # Squeeze to 1D
    waveform = waveform.squeeze(0)
    
    # Pad or truncate
    target_length = int(sample_rate * duration_sec)
    waveform = pad_or_truncate(waveform, target_length)
    
    return waveform
