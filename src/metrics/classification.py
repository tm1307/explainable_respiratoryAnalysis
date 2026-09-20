import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
from typing import Dict, List, Union, Optional

def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]], 
    y_pred: Union[np.ndarray, List[int]], 
    y_prob: Optional[np.ndarray] = None, 
    num_classes: int = 4
) -> Dict[str, float]:
    """
    Compute classification metrics including precision, recall, f1,
    specificity, accuracy, and balanced accuracy. If probabilities are given,
    computes AUROC and AUPRC.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    precisions = precision_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
        
    metrics['specificity_macro'] = float(np.mean(specificities))
    
    for i in range(num_classes):
        metrics[f'precision_class_{i}'] = float(precisions[i])
        metrics[f'recall_class_{i}'] = float(recalls[i])
        metrics[f'f1_class_{i}'] = float(f1s[i])
        metrics[f'specificity_class_{i}'] = float(specificities[i])
        
    if y_prob is not None:
        y_prob = np.array(y_prob)
        y_true_one_hot = np.eye(num_classes)[y_true]
        
        try:
            metrics['auroc_macro'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        except ValueError:
            metrics['auroc_macro'] = float('nan')
            
        try:
            metrics['auprc_macro'] = float(average_precision_score(y_true_one_hot, y_prob, average='macro'))
        except ValueError:
            metrics['auprc_macro'] = float('nan')
            
        for i in range(num_classes):
            try:
                auroc = roc_auc_score(y_true_one_hot[:, i], y_prob[:, i])
            except ValueError:
                auroc = float('nan')
            try:
                auprc = average_precision_score(y_true_one_hot[:, i], y_prob[:, i])
            except ValueError:
                auprc = float('nan')
            
            metrics[f'auroc_class_{i}'] = float(auroc)
            metrics[f'auprc_class_{i}'] = float(auprc)
            
    return metrics
