"""Utilities for applying PEFT methods to model layers and aggregating outputs."""

import torch
import torch.nn as nn
from typing import List, Dict
from models import LoRAConv2d, GeometricConv2d


def apply_peft_to_conv_layer(layer: nn.Module, method: str, rank: int = 8) -> nn.Module:
    """Apply PEFT method to a single convolutional layer.
    
    Args:
        layer: The layer to adapt (Conv2d)
        method: PEFT method ('lora' or 'gft')
        rank: Rank for low-rank adaptation
        
    Returns:
        Adapted layer
    """
    if not isinstance(layer, nn.Conv2d):
        return layer
    
    if method == 'lora':
        return LoRAConv2d(layer, rank=rank)
    elif method == 'gft':
        return GeometricConv2d(layer, rank=rank)
    else:
        raise ValueError(f"Unsupported PEFT method: {method}")


def apply_peft_to_sequential(module: nn.Sequential, method: str, rank: int = 8) -> nn.Sequential:
    """Apply PEFT to all Conv2d layers in a Sequential module.
    
    Args:
        module: Sequential module containing layers
        method: PEFT method ('lora' or 'gft')
        rank: Rank for low-rank adaptation
        
    Returns:
        Modified Sequential module
    """
    for name, layer in module.named_children():
        if isinstance(layer, nn.Conv2d):
            setattr(module, name, apply_peft_to_conv_layer(layer, method, rank))
        elif isinstance(layer, nn.Sequential):
            setattr(module, name, apply_peft_to_sequential(layer, method, rank))
        elif hasattr(layer, 'conv1'):  # BasicBlock or Bottleneck
            apply_peft_to_resnet_block(layer, method, rank)
    
    return module


def apply_peft_to_resnet_block(block: nn.Module, method: str, rank: int = 8):
    """Apply PEFT to a ResNet BasicBlock or Bottleneck.
    
    Args:
        block: ResNet block (BasicBlock or Bottleneck)
        method: PEFT method ('lora' or 'gft')
        rank: Rank for low-rank adaptation
    """
    # Apply to conv layers in the block
    if hasattr(block, 'conv1') and isinstance(block.conv1, nn.Conv2d):
        block.conv1 = apply_peft_to_conv_layer(block.conv1, method, rank)
    if hasattr(block, 'conv2') and isinstance(block.conv2, nn.Conv2d):
        block.conv2 = apply_peft_to_conv_layer(block.conv2, method, rank)
    if hasattr(block, 'conv3') and isinstance(block.conv3, nn.Conv2d):  # Bottleneck
        block.conv3 = apply_peft_to_conv_layer(block.conv3, method, rank)


def apply_peft_to_resnet_layer(layer: nn.Sequential, method: str, rank: int = 8) -> nn.Sequential:
    """Apply PEFT to all blocks in a ResNet layer (e.g., layer3, layer4).
    
    Args:
        layer: ResNet layer (Sequential of BasicBlocks or Bottlenecks)
        method: PEFT method ('lora' or 'gft')
        rank: Rank for low-rank adaptation
        
    Returns:
        Modified layer
    """
    for block in layer:
        apply_peft_to_resnet_block(block, method, rank)
    return layer


def get_class_aggregation_mapping(task_type: str, num_base_classes: int) -> Dict[int, List[int]]:
    """Get mapping from adapted task classes to base task classes.
    
    Args:
        task_type: Type of adaptation task
        num_base_classes: Number of classes in base task (10 for CIFAR-10, 100 for CIFAR-100)
        
    Returns:
        Dictionary mapping adapted class index to list of base class indices
        e.g., {0: [0,2,4,6,8], 1: [1,3,5,7,9]} for even_odd
    """
    if num_base_classes == 10:
        # CIFAR-10 tasks
        if task_type == 'even_odd':
            return {
                0: [0, 2, 4, 6, 8],  # Even classes
                1: [1, 3, 5, 7, 9]   # Odd classes
            }
        elif task_type == 'animal_vehicle':
            return {
                0: [2, 3, 4, 5, 6, 7],  # Animals
                1: [0, 1, 8, 9]          # Vehicles
            }
        elif task_type == 'living_nonliving':
            return {
                0: [2, 3, 4, 5, 6, 7],  # Living
                1: [0, 1, 8, 9]          # Non-living
            }
        elif task_type == 'gt5_lt5':
            return {
                0: [0, 1, 2, 3, 4],  # < 5
                1: [5, 6, 7, 8, 9]   # >= 5
            }
    
    elif num_base_classes == 100:
        # CIFAR-100 coarse task
        if task_type == 'coarse':
            # Map each of 20 coarse classes to their 5 fine classes
            coarse_to_fine = {}
            fine_to_coarse = {
                # aquatic mammals
                4: 0, 30: 0, 55: 0, 72: 0, 95: 0,
                # fish
                1: 1, 32: 1, 67: 1, 73: 1, 91: 1,
                # flowers
                54: 2, 62: 2, 70: 2, 82: 2, 92: 2,
                # food containers
                9: 3, 10: 3, 16: 3, 28: 3, 61: 3,
                # fruit and vegetables
                0: 4, 51: 4, 53: 4, 57: 4, 83: 4,
                # household electrical devices
                22: 5, 39: 5, 40: 5, 86: 5, 87: 5,
                # household furniture
                5: 6, 20: 6, 25: 6, 84: 6, 94: 6,
                # insects
                6: 7, 7: 7, 14: 7, 18: 7, 24: 7,
                # large carnivores
                3: 8, 42: 8, 43: 8, 88: 8, 97: 8,
                # large outdoor man-made objects
                12: 9, 17: 9, 37: 9, 68: 9, 76: 9,
                # large natural outdoor scenes
                23: 10, 33: 10, 49: 10, 60: 10, 71: 10,
                # large omnivores and herbivores
                15: 11, 19: 11, 21: 11, 31: 11, 38: 11,
                # medium-sized mammals
                34: 12, 63: 12, 64: 12, 66: 12, 75: 12,
                # non-insect invertebrates
                26: 13, 45: 13, 77: 13, 79: 13, 99: 13,
                # people
                2: 14, 11: 14, 35: 14, 46: 14, 98: 14,
                # reptiles
                27: 15, 29: 15, 44: 15, 78: 15, 93: 15,
                # small mammals
                36: 16, 50: 16, 65: 16, 74: 16, 80: 16,
                # trees
                47: 17, 52: 17, 56: 17, 59: 17, 96: 17,
                # vehicles 1
                8: 18, 13: 18, 48: 18, 58: 18, 90: 18,
                # vehicles 2
                41: 19, 69: 19, 81: 19, 85: 19, 89: 19
            }
            
            # Invert mapping
            for fine_class, coarse_class in fine_to_coarse.items():
                if coarse_class not in coarse_to_fine:
                    coarse_to_fine[coarse_class] = []
                coarse_to_fine[coarse_class].append(fine_class)
            
            return coarse_to_fine
    
    raise ValueError(f"Unsupported task_type '{task_type}' for {num_base_classes} classes")


def aggregate_logits(logits: torch.Tensor, aggregation_mapping: Dict[int, List[int]]) -> torch.Tensor:
    """Aggregate base task logits to adapted task logits using LogSumExp.
    
    This is mathematically equivalent to summing probabilities in the softmax space:
    log(sum(exp(logits_i)))
    
    Args:
        logits: Base task logits [batch_size, num_base_classes]
        aggregation_mapping: Mapping from adapted class to base classes
        
    Returns:
        Aggregated logits [batch_size, num_adapted_classes]
    """
    batch_size = logits.size(0)
    num_adapted_classes = len(aggregation_mapping)
    
    aggregated = torch.zeros(batch_size, num_adapted_classes, device=logits.device)
    
    for adapted_class, base_classes in aggregation_mapping.items():
        # Use torch.logsumexp for numerical stability
        aggregated[:, adapted_class] = torch.logsumexp(logits[:, base_classes], dim=1)
    
    return aggregated


class FeatureExtractor(nn.Module):
    """Utility class to extract features from specific layers using hooks.
    
    Args:
        model: PyTorch model
        layers: List of layer names to extract from (e.g., ['layer3', 'layer4'])
        pool: Whether to apply global average pooling to the extracted features
    """
    
    def __init__(self, model: nn.Module, layers: List[str], pool: bool = True):
        super().__init__()
        self.model = model
        self.layers = layers
        self.pool = pool
        self.features = {layer: None for layer in layers}
        self.hooks = []
        
        self._setup_hooks()
        
    def _setup_hooks(self):
        """Setup forward hooks."""
        for layer_name in self.layers:
            # Find the layer
            parts = layer_name.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
                
            # Define the hook
            def hook_fn(m, i, o, name=layer_name):
                if self.pool and len(o.shape) == 4:
                    # Global average pooling: [B, C, H, W] -> [B, C]
                    self.features[name] = torch.mean(o, dim=(2, 3))
                else:
                    self.features[name] = o
            
            self.hooks.append(module.register_forward_hook(hook_fn))
            
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
            
    def forward(self, x) -> Dict[str, torch.Tensor]:
        """Forward pass and return collected features."""
        # Reset features
        self.features = {layer: None for layer in self.layers}
        # Run model
        self.model(x)
        return self.features
