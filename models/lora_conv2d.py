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
        self.rank = rank

        # Freeze base layer
        for p in self.base.parameters():
            p.requires_grad = False

        # Dimensions
        out_channels = base_layer.out_channels
        in_channels = base_layer.in_channels
        kh, kw = base_layer.kernel_size
        kernel_numel = in_channels * kh * kw

        # LoRA params
        self.A = nn.Parameter(torch.randn(rank, kernel_numel) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_channels, rank))

        # Cached delta weight
        self.register_buffer("_delta_weight", None)
        self._dirty = True  # marks when A or B changes

        # Hook to mark cache dirty after backward
        self.A.register_hook(self._mark_dirty)
        self.B.register_hook(self._mark_dirty)

    def _mark_dirty(self, grad):
        self._dirty = True
        return grad

    def _compute_delta_weight(self):
        delta = self.B @ self.A  # (out_channels, kernel_numel)
        delta = delta.view(
            self.base.out_channels,
            self.base.in_channels,
            *self.base.kernel_size
        )
        self._delta_weight = delta
        self._dirty = False

    def forward(self, x):
        if self._dirty or self._delta_weight is None:
            self._compute_delta_weight()

        weight = self.base.weight + self._delta_weight

        return F.conv2d(
            x,
            weight,
            self.base.bias,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
        )
    
    def get_trainable_params(self):
        """Return number of trainable parameters."""
        return self.A.numel() + self.B.numel()
