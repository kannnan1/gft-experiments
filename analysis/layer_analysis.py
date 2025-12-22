"""Analysis utilities for geometric metrics and layer-wise analysis."""

import torch
from typing import Dict, List, Tuple
from utils.metrics import compute_distance_preservation, compute_singular_value_divergence, compute_cka_similarity


def analyze_layer_wise_forgetting(
    model_base: torch.nn.Module,
    model_adapted: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    """Analyze layer-wise forgetting using CKA similarity.
    
    Args:
        model_base: Base model before adaptation
        model_adapted: Model after adaptation
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Dictionary with layer names and CKA scores
    """
    model_base.eval()
    model_adapted.eval()
    
    # Hook to capture activations
    activations_base = {}
    activations_adapted = {}
    
    def get_activation(name, storage):
        def hook(model, input, output):
            storage[name] = output.detach()
        return hook
    
    # Register hooks for all layers
    hooks_base = []
    hooks_adapted = []
    
    for name, module in model_base.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks_base.append(module.register_forward_hook(get_activation(name, activations_base)))
    
    for name, module in model_adapted.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks_adapted.append(module.register_forward_hook(get_activation(name, activations_adapted)))
    
    # Run forward pass
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            model_base(inputs)
            model_adapted(inputs)
            break  # Just one batch for analysis
    
    # Remove hooks
    for hook in hooks_base + hooks_adapted:
        hook.remove()
    
    # Compute CKA for each layer
    cka_scores = {}
    for name in activations_base.keys():
        if name in activations_adapted:
            act_base = activations_base[name].flatten(1)
            act_adapted = activations_adapted[name].flatten(1)
            cka = compute_cka_similarity(act_base, act_adapted)
            cka_scores[name] = cka
    
    return cka_scores


def analyze_weight_changes(
    model_base: torch.nn.Module,
    model_adapted: torch.nn.Module
) -> Dict[str, float]:
    """Analyze weight changes using Frobenius norm.
    
    Args:
        model_base: Base model
        model_adapted: Adapted model
        
    Returns:
        Dictionary with layer names and Frobenius norms
    """
    weight_changes = {}
    
    base_state = model_base.state_dict()
    adapted_state = model_adapted.state_dict()
    
    for name in base_state.keys():
        if 'weight' in name and name in adapted_state:
            W_base = base_state[name]
            W_adapted = adapted_state[name]
            
            # Compute Frobenius norm of difference
            frob_norm = torch.norm(W_adapted - W_base, p='fro').item()
            weight_changes[name] = frob_norm
    
    return weight_changes
