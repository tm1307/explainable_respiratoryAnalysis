import torch
import pytest
from src.models.baseline_cnn import BaselineCNN
from src.explainability.gradcam import GradCAM
from src.explainability.integrated_grad import IntegratedGradients

@pytest.fixture
def model_and_input():
    torch.manual_seed(42)
    model = BaselineCNN(num_classes=4, n_mels=128)
    model.eval()
    input_tensor = torch.randn(1, 1, 128, 160)
    return model, input_tensor

def get_target_layer(model):
    """Helper to find a Conv2d layer to target."""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return module
    return None

def test_gradcam_output_shape(model_and_input):
    model, input_tensor = model_and_input
    target_layer = get_target_layer(model)
    
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor)
    
    # Check output shape matches spatial dims (128, 160)
    assert heatmap.shape == (128, 160)
    
def test_gradcam_values_in_range(model_and_input):
    model, input_tensor = model_and_input
    target_layer = get_target_layer(model)
    
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor)
    
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

def test_gradcam_not_all_zeros(model_and_input):
    model, input_tensor = model_and_input
    target_layer = get_target_layer(model)
    
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor)
    
    assert torch.sum(heatmap) > 0.0

def test_gradcam_specific_class(model_and_input):
    model, input_tensor = model_and_input
    target_layer = get_target_layer(model)
    
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, target_class=2)
    
    assert heatmap.shape == (128, 160)

def test_gradcam_cleanup(model_and_input):
    model, input_tensor = model_and_input
    target_layer = get_target_layer(model)
    
    gradcam = GradCAM(model, target_layer)
    assert len(gradcam.handlers) == 2
    
    gradcam.remove_hooks()
    assert len(gradcam.handlers) == 0

def test_ig_output_shape(model_and_input):
    model, input_tensor = model_and_input
    ig = IntegratedGradients(model)
    attr = ig.attribute(input_tensor)
    
    assert attr.shape == input_tensor.shape

def test_ig_not_all_zeros(model_and_input):
    model, input_tensor = model_and_input
    ig = IntegratedGradients(model)
    attr = ig.attribute(input_tensor)
    
    assert torch.sum(torch.abs(attr)) > 0.0

def test_ig_with_custom_baseline(model_and_input):
    model, input_tensor = model_and_input
    ig = IntegratedGradients(model)
    
    baseline = torch.ones_like(input_tensor)
    attr = ig.attribute(input_tensor, baseline=baseline)
    
    assert attr.shape == input_tensor.shape
    assert torch.sum(torch.abs(attr)) > 0.0

def test_ig_different_n_steps(model_and_input):
    model, input_tensor = model_and_input
    ig = IntegratedGradients(model)
    
    attr = ig.attribute(input_tensor, n_steps=10)
    assert attr.shape == input_tensor.shape
