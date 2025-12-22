"""Model factory for creating and modifying models with PEFT methods."""

import torch
import torch.nn as nn
from torchvision import models
from .lora_linear import LoRALinear
from .geometric_linear import GeometricLinear


def create_model(architecture, num_classes=10, pretrained=True):
    """Create a base model.
    
    Args:
        architecture: Model architecture name (e.g., 'resnet18', 'resnet50')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    if architecture == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model


def apply_peft_method(model, method='lora', rank=8, target_modules=None):
    """Apply PEFT method to model.
    
    Args:
        model: Base model to modify
        method: PEFT method ('lora', 'gft', 'full_ft', 'adapter')
        rank: Rank for low-rank methods
        target_modules: List of module names to adapt (None = all Linear layers)
        
    Returns:
        Modified model with trainable/frozen parameters set appropriately
    """
    if method == 'full_ft':
        # Full fine-tuning - all parameters trainable
        for param in model.parameters():
            param.requires_grad = True
        return model
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    if method in ['lora', 'gft']:
        # Replace linear layers with LoRA or GFT
        _replace_linear_layers(model, method, rank, target_modules)
    elif method == 'adapter':
        # Add adapter layers (simplified version)
        _add_adapter_layers(model, rank)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return model


def _replace_linear_layers(model, method, rank, target_modules=None):
    """Replace linear layers with LoRA or GFT layers."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            # Recursively apply to child modules
            _replace_linear_layers(module, method, rank, target_modules)
        
        if isinstance(module, nn.Linear):
            # Check if this module should be replaced
            if target_modules is None or name in target_modules:
                # Skip the final classification layer
                if 'fc' not in name and 'classifier' not in name:
                    if method == 'lora':
                        setattr(model, name, LoRALinear(module, rank=rank))
                    elif method == 'gft':
                        setattr(model, name, GeometricLinear(module, rank=rank))


def _add_adapter_layers(model, rank):
    """Add adapter layers (bottleneck) after each layer.
    
    Note: This is a simplified implementation. Full adapter implementation
    would require more sophisticated integration.
    """
    # TODO: Implement adapter layers if needed for experiments
    raise NotImplementedError("Adapter layers not yet implemented")


def count_parameters(model):
    """Count total and trainable parameters.
    
    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_percent': 100.0 * trainable / total if total > 0 else 0
    }
