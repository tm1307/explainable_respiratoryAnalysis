import torch
import torch.nn as nn
from typing import Optional

class IntegratedGradients:
    """
    Integrated Gradients (Sundararajan et al., 2017).
    """
    def __init__(self, model: nn.Module):
        self.model = model
        
    def attribute(self, input_tensor: torch.Tensor, target_class: Optional[int] = None, 
                  n_steps: int = 50, baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.model.eval()
        
        if target_class is None:
            with torch.no_grad():
                out = self.model(input_tensor)
                target_class = out.argmax(dim=1).item()
                
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
            
        gradients = []
        for i in range(1, n_steps + 1):
            x_i = baseline + (i / n_steps) * (input_tensor - baseline)
            x_i = x_i.detach().requires_grad_(True)
            
            output = self.model(x_i)
            score = output[0, target_class] if output.size(0) == 1 else output[:, target_class].sum()
            
            self.model.zero_grad()
            score.backward()
            gradients.append(x_i.grad.detach())
            
        # Average gradients (approximating integral)
        avg_grads = torch.stack(gradients).mean(dim=0)
        
        # Multiply by (input - baseline)
        attributions = avg_grads * (input_tensor - baseline)
        return attributions
