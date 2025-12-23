"""LoRA (Low-Rank Adaptation) for Conv2d layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """LoRA adaptation for Conv2d layers.
    
    Args:
        base_layer: The base Conv2d layer to adapt
        rank: Rank of the low-rank decomposition (default: 8)
    """
    
    def __init__(self, base_layer: nn.Conv2d, rank: int = 8):
        super().__init__()
        self.base = base_layer
        # Freeze base layer
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        
        # Get dimensions
        out_channels = base_layer.out_channels
        in_channels = base_layer.in_channels
        kernel_size = base_layer.kernel_size
        
        # LoRA parameters: A and B matrices
        # A: (rank, in_channels * kernel_h * kernel_w)
        # B: (out_channels, rank)
        self.rank = rank
        kernel_numel = in_channels * kernel_size[0] * kernel_size[1]
        
        self.A = nn.Parameter(torch.randn(rank, kernel_numel) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_channels, rank))
        
        # Store conv params
        self.stride = base_layer.stride
        self.padding = base_layer.padding
        self.dilation = base_layer.dilation
        self.groups = base_layer.groups
        
    def forward(self, x):
        """Forward pass with LoRA adaptation."""
        # Compute LoRA delta: B @ A
        delta = self.B @ self.A  # (out_channels, kernel_numel)
        
        # Reshape to conv weight shape
        delta = delta.view(
            self.base.out_channels,
            self.base.in_channels,
            self.base.kernel_size[0],
            self.base.kernel_size[1]
        )
        
        # Apply convolution with adapted weights
        adapted_weight = self.base.weight + delta
        return F.conv2d(
            x, adapted_weight, self.base.bias,
            self.stride, self.padding, self.dilation, self.groups
        )
    
    def get_trainable_params(self):
        """Return number of trainable parameters."""
        return self.A.numel() + self.B.numel()
