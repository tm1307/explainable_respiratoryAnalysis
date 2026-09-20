import numpy as np
from typing import Tuple

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE) across prediction bins.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin = in_bin | (confidences == bin_lower)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 4) -> float:
    """
    Compute Brier Score for multi-class classification.
    """
    y_one_hot = np.eye(num_classes)[y_true]
    brier = np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1))
    return float(brier)

def reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data (confidences, accuracies, counts) per bin for plotting a reliability diagram.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin = in_bin | (confidences == bin_lower)
            
        bin_counts[i] = np.sum(in_bin)
        
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(accuracies[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
            
    return bin_confidences, bin_accuracies, bin_counts
