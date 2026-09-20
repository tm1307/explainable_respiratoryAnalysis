import numpy as np
from typing import List
from skimage.metrics import structural_similarity
from scipy.stats import spearmanr

def ssim_2d(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """
    Compute SSIM between two 2D explanation heatmaps.
    """
    drange = float(max(map_a.max() - map_a.min(), map_b.max() - map_b.min()))
    if drange == 0:
        drange = 1.0
    return float(structural_similarity(map_a, map_b, data_range=drange))

def spearman_rank_correlation(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """
    Compute Spearman rank correlation between flattened maps.
    """
    corr, _ = spearmanr(map_a.flatten(), map_b.flatten())
    if np.isnan(corr):
        return 0.0
    return float(corr)

def iou_top_k(map_a: np.ndarray, map_b: np.ndarray, k_fraction: float = 0.1) -> float:
    """
    Compute IoU of top-k attributed regions.
    """
    flat_a = map_a.flatten()
    flat_b = map_b.flatten()
    
    k = max(1, int(len(flat_a) * k_fraction))
    
    top_k_a = np.argsort(flat_a)[-k:]
    top_k_b = np.argsort(flat_b)[-k:]
    
    intersection = len(np.intersect1d(top_k_a, top_k_b))
    union = len(np.union1d(top_k_a, top_k_b))
    
    return float(intersection / union) if union > 0 else 0.0

def compute_stability(clean_maps: List[np.ndarray], noisy_maps: List[np.ndarray], metric: str = 'ssim') -> float:
    """
    Compute mean stability across paired clean/noisy explanation maps.
    """
    if not clean_maps or not noisy_maps or len(clean_maps) != len(noisy_maps):
        raise ValueError("Invalid input maps.")
        
    scores = []
    for m_c, m_n in zip(clean_maps, noisy_maps):
        if metric == 'ssim':
            scores.append(ssim_2d(m_c, m_n))
        elif metric == 'spearman':
            scores.append(spearman_rank_correlation(m_c, m_n))
        elif metric == 'iou':
            scores.append(iou_top_k(m_c, m_n))
        else:
            raise ValueError(f"Unknown metric {metric}")
            
    return float(np.mean(scores))
