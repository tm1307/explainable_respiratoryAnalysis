import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GradCAM:
    """
    Grad-CAM for CNN models.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handlers = []
        
        # Register hooks
        self.handlers.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self.handlers.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )
        
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
        
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass for the target class
        score = output[0, target_class]
        score.backward()
        
        # Compute weights: global average pooling of gradients
        # gradients shape: (B, C, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Compute heatmap
        # activations shape: (B, C, H, W)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        # Upsample to input spatial dimensions
        # input_tensor shape: (B, C, H, W)
        _, _, H, W = input_tensor.size()
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        
        return cam.squeeze()
        
    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        for handle in self.handlers:
            handle.remove()
        self.handlers = []
        
    def __del__(self):
        self.remove_hooks()
