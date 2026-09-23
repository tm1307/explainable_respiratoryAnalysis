import torch
import numpy as np
from typing import Any

# NumPy 2.0 renamed np.trapz to np.trapezoid; support both versions.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

def insertion_auc(model: Any, input_tensor: torch.Tensor, attribution_map: np.ndarray, target_class: int, n_steps: int = 100, device: str = 'cpu') -> float:
    """
    Progressively insert top-attributed time-frequency regions into a blank (zero) input.
    """
    model.eval()
    model.to(device)
    
    flat_attr = attribution_map.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1].copy()
    
    total_features = len(flat_attr)
    step_size = max(1, total_features // n_steps)
    
    current_input = torch.zeros_like(input_tensor).to(device)
    input_tensor = input_tensor.to(device)
    
    probs = []
    
    with torch.no_grad():
        out = model(current_input)
        prob = torch.softmax(out, dim=1)[0, target_class].item()
        probs.append(prob)
        
        flat_input = input_tensor.flatten()
        flat_current = current_input.flatten()
        
        for step in range(1, n_steps + 1):
            start_idx = (step - 1) * step_size
            end_idx = min(step * step_size, total_features)
            
            if start_idx < total_features:
                indices_to_add = sorted_indices[start_idx:end_idx]
                flat_current[indices_to_add] = flat_input[indices_to_add]
                
                reshaped_current = flat_current.view(input_tensor.shape)
                out = model(reshaped_current)
                prob = torch.softmax(out, dim=1)[0, target_class].item()
                probs.append(prob)
            else:
                probs.append(probs[-1])
                
    return float(_trapezoid(probs, dx=1.0 / n_steps))

def deletion_auc(model: Any, input_tensor: torch.Tensor, attribution_map: np.ndarray, target_class: int, n_steps: int = 100, device: str = 'cpu') -> float:
    """
    Progressively delete (zero out) top-attributed regions from the input.
    """
    model.eval()
    model.to(device)
    
    flat_attr = attribution_map.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1].copy()
    
    total_features = len(flat_attr)
    step_size = max(1, total_features // n_steps)
    
    current_input = input_tensor.clone().to(device)
    
    probs = []
    
    with torch.no_grad():
        out = model(current_input)
        prob = torch.softmax(out, dim=1)[0, target_class].item()
        probs.append(prob)
        
        flat_current = current_input.flatten()
        
        for step in range(1, n_steps + 1):
            start_idx = (step - 1) * step_size
            end_idx = min(step * step_size, total_features)
            
            if start_idx < total_features:
                indices_to_del = sorted_indices[start_idx:end_idx]
                flat_current[indices_to_del] = 0.0
                
                reshaped_current = flat_current.view(input_tensor.shape)
                out = model(reshaped_current)
                prob = torch.softmax(out, dim=1)[0, target_class].item()
                probs.append(prob)
            else:
                probs.append(probs[-1])
                
    return float(_trapezoid(probs, dx=1.0 / n_steps))

def aopc(model: Any, input_tensor: torch.Tensor, attribution_map: np.ndarray, target_class: int, K: int = 10, device: str = 'cpu') -> float:
    """
    Compute Area Over the Perturbation Curve (AOPC).
    """
    model.eval()
    model.to(device)
    
    flat_attr = attribution_map.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1].copy()
    
    current_input = input_tensor.clone().to(device)
    
    with torch.no_grad():
        out_orig = model(current_input)
        f_x = torch.softmax(out_orig, dim=1)[0, target_class].item()
        
        drops = []
        flat_current = current_input.flatten()
        
        for k in range(1, K + 1):
            if k - 1 < len(sorted_indices):
                idx = sorted_indices[k - 1]
                flat_current[idx] = 0.0
                
                reshaped = flat_current.view(input_tensor.shape)
                out_pert = model(reshaped)
                f_x_pert = torch.softmax(out_pert, dim=1)[0, target_class].item()
                drops.append(f_x - f_x_pert)
            else:
                break
                
    return float(np.mean(drops)) if drops else 0.0
