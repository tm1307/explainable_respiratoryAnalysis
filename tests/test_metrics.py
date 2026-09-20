import pytest
import numpy as np
import torch
import torch.nn as nn
from src.metrics.classification import compute_classification_metrics
from src.metrics.calibration import expected_calibration_error, brier_score, reliability_diagram_data
from src.metrics.stability import ssim_2d, spearman_rank_correlation, iou_top_k, compute_stability
from src.metrics.faithfulness import insertion_auc, deletion_auc, aopc
from src.metrics.jri import normalized_robustness, joint_reliability_index, area_under_reliability_curve, compute_full_reliability_profile

class BaselineCNN(nn.Module):
    """Mock model for testing faithfulness metrics."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.fc = nn.Linear(4 * 6 * 6, 4)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def setup_module(module):
    np.random.seed(42)
    torch.manual_seed(42)

# --- Classification metrics tests ---
def test_perfect_classification():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]
    y_prob = np.eye(4)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert metrics['accuracy'] == 1.0
    assert metrics['f1_macro'] == 1.0
    assert metrics['auroc_macro'] == 1.0
    assert metrics['balanced_accuracy'] == 1.0

def test_all_wrong():
    y_true = [0, 1, 2, 3]
    y_pred = [1, 2, 3, 0]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics['f1_macro'] == 0.0
    assert metrics['accuracy'] == 0.0

def test_with_probabilities():
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 0, 1, 2, 3]
    y_prob = np.array([
        [0.9, 0.05, 0.03, 0.02],
        [0.05, 0.9, 0.03, 0.02],
        [0.03, 0.05, 0.9, 0.02],
        [0.02, 0.05, 0.03, 0.9],
        [0.85, 0.05, 0.05, 0.05],
        [0.05, 0.85, 0.05, 0.05],
        [0.05, 0.05, 0.85, 0.05],
        [0.05, 0.05, 0.05, 0.85],
    ])
    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob, num_classes=4)
    assert 'auroc_macro' in metrics
    assert 'auprc_macro' in metrics
    assert not np.isnan(metrics['auroc_macro'])

# --- Calibration tests ---
def test_ece_perfect_calibration():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    ece = expected_calibration_error(y_true, y_prob)
    assert np.isclose(ece, 0.0)

def test_brier_score_perfect():
    y_true = np.array([0, 1])
    y_prob = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    brier = brier_score(y_true, y_prob, num_classes=4)
    assert np.isclose(brier, 0.0)

def test_brier_score_worst():
    y_true = np.array([0, 1])
    y_prob = np.array([[0, 1.0, 0, 0], [1.0, 0, 0, 0]])
    brier = brier_score(y_true, y_prob, num_classes=4)
    assert brier > 1.0

def test_reliability_diagram_data():
    y_true = np.array([0, 1])
    y_prob = np.array([[0.9, 0.1, 0, 0], [0.2, 0.8, 0, 0]])
    conf, acc, counts = reliability_diagram_data(y_true, y_prob, n_bins=10)
    assert len(conf) == 10
    assert len(acc) == 10
    assert len(counts) == 10

# --- Stability tests ---
def test_ssim_identical_maps():
    map_a = np.random.rand(8, 8)
    assert np.isclose(ssim_2d(map_a, map_a), 1.0)

def test_ssim_different_maps():
    map_a = np.ones((8, 8))
    map_b = np.zeros((8, 8))
    assert ssim_2d(map_a, map_b) < 1.0

def test_spearman_identical():
    map_a = np.random.rand(8, 8)
    assert np.isclose(spearman_rank_correlation(map_a, map_a), 1.0)

def test_iou_identical():
    map_a = np.random.rand(8, 8)
    assert np.isclose(iou_top_k(map_a, map_a, k_fraction=0.1), 1.0)

def test_iou_disjoint():
    map_a = np.zeros(100)
    map_a[:10] = 1.0
    map_b = np.zeros(100)
    map_b[-10:] = 1.0
    assert np.isclose(iou_top_k(map_a, map_b, k_fraction=0.1), 0.0)

def test_compute_stability():
    maps = [np.random.rand(8, 8) for _ in range(3)]
    stability = compute_stability(maps, maps, metric='ssim')
    assert np.isclose(stability, 1.0)

# --- Faithfulness tests ---
def test_insertion_auc_returns_float():
    model = BaselineCNN()
    inp = torch.rand(1, 1, 8, 8)
    attr = np.random.rand(8, 8)
    res = insertion_auc(model, inp, attr, target_class=0, n_steps=10)
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0

def test_deletion_auc_returns_float():
    model = BaselineCNN()
    inp = torch.rand(1, 1, 8, 8)
    attr = np.random.rand(8, 8)
    res = deletion_auc(model, inp, attr, target_class=0, n_steps=10)
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0

def test_aopc_returns_float():
    model = BaselineCNN()
    inp = torch.rand(1, 1, 8, 8)
    attr = np.random.rand(8, 8)
    res = aopc(model, inp, attr, target_class=0, K=5)
    assert isinstance(res, float)
    assert res >= 0.0

# --- JRI tests ---
def test_normalized_robustness():
    assert np.isclose(normalized_robustness(f1_noisy=0.5, f1_clean=1.0), 0.5)

def test_jri_harmonic_mean():
    R, S = 0.8, 0.6
    expected = 2 * R * S / (R + S)
    assert np.isclose(joint_reliability_index(R, S), expected)

def test_jri_zero_when_either_zero():
    assert np.isclose(joint_reliability_index(0.0, 0.8), 0.0)
    assert np.isclose(joint_reliability_index(0.8, 0.0), 0.0)

def test_jri_one_when_both_perfect():
    assert np.isclose(joint_reliability_index(1.0, 1.0), 1.0)

def test_aurc_is_mean():
    vals = [0.2, 0.4, 0.6]
    assert np.isclose(area_under_reliability_curve(vals), 0.4)

def test_full_reliability_profile():
    f1_clean = 0.9
    f1_snr = {0.0: 0.45, 5.0: 0.9}
    stab_snr = {0.0: 0.5, 5.0: 1.0}
    prof = compute_full_reliability_profile(f1_clean, f1_snr, stab_snr)
    
    assert 'jri_per_snr' in prof
    assert prof['robustness_per_snr'][5.0] == 1.0
    assert prof['jri_per_snr'][5.0] == 1.0
    assert prof['robustness_per_snr'][0.0] == 0.5
    assert prof['jri_per_snr'][0.0] == 0.5
    assert prof['aurc'] == 0.75
