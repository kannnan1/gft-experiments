"""LoRA (Low-Rank Adaptation) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA linear layer with low-rank adaptation.
    
    Args:
        base_layer: The base linear layer to adapt
        rank: Rank of the low-rank decomposition (default: 8)
    """
    
    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base = base_layer
        self.base.weight.requires_grad = False
        self.base.bias.requires_grad = False

        d_out, d_in = base_layer.weight.shape
        self.in_features = d_in
        self.out_features = d_out
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        self.rank = rank

    def forward(self, x):
        """Forward pass with LoRA adaptation."""
        delta = self.B @ self.A
        return F.linear(x, self.base.weight + delta, self.base.bias)
    
    def merge_weights(self):
        """Merge LoRA weights into base layer for inference."""
        with torch.no_grad():
            delta = self.B @ self.A
            self.base.weight.data += delta
            self.A.zero_()
            self.B.zero_()
    
    def get_trainable_params(self):
        """Return number of trainable parameters."""
        return self.A.numel() + self.B.numel()
