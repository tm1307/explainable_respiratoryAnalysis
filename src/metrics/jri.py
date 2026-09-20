from typing import Dict, List, Any
import numpy as np

def normalized_robustness(f1_noisy: float, f1_clean: float) -> float:
    """
    Compute Normalized Robustness: R(SNR) = F1(SNR) / F1(clean), clamped to [0, 1].
    """
    if f1_clean == 0:
        return 0.0
    return float(max(0.0, min(1.0, f1_noisy / f1_clean)))

def joint_reliability_index(robustness: float, stability: float) -> float:
    """
    Compute Joint Reliability Index: harmonic mean of robustness and stability.
    """
    if robustness + stability == 0:
        return 0.0
    return float(2 * robustness * stability / (robustness + stability))

def area_under_reliability_curve(jri_values: List[float]) -> float:
    """
    Compute Area Under Reliability Curve (AURC): mean across all SNR levels.
    """
    if not jri_values:
        return 0.0
    return float(np.mean(jri_values))

def compute_full_reliability_profile(f1_clean: float, f1_per_snr: Dict[float, float], stability_per_snr: Dict[float, float]) -> Dict[str, Any]:
    """
    Compute JRI at each SNR level and summarize.
    """
    robustness_per_snr = {}
    jri_per_snr = {}
    
    snr_levels = sorted(list(f1_per_snr.keys()))
    
    for snr in snr_levels:
        f1_noisy = f1_per_snr[snr]
        stability = stability_per_snr[snr]
        
        robustness = normalized_robustness(f1_noisy, f1_clean)
        robustness_per_snr[snr] = robustness
        
        jri = joint_reliability_index(robustness, stability)
        jri_per_snr[snr] = jri
        
    jri_values = [jri_per_snr[snr] for snr in snr_levels]
    aurc = area_under_reliability_curve(jri_values)
    
    return {
        'jri_per_snr': jri_per_snr,
        'aurc': aurc,
        'robustness_per_snr': robustness_per_snr,
        'stability_per_snr': stability_per_snr
    }
