"""Geometric Fine-Tuning (GFT) for Conv2d layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricConv2d(nn.Module):
    """Geometric adaptation for Conv2d layers using orthogonal transformations.
    
    Args:
        base_layer: The base Conv2d layer to adapt
        rank: Rank of the low-rank parameterization (default: 8)
    """
    
    def __init__(self, base_layer: nn.Conv2d, rank: int = 8):
        super().__init__()
        
        # Get weight and perform SVD on CPU (MPS has limited SVD support)
        W = base_layer.weight.data.detach().cpu().float()
        
        # Reshape to 2D for SVD: (out_channels, in_channels * kernel_h * kernel_w)
        out_channels = W.shape[0]
        W_flat = W.view(out_channels, -1)
        
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
        
        # Compute P and Q matrices
        P = (Vh.T @ torch.diag(S) @ Vh).float()
        Q = (U @ Vh).float()
        
        self.register_buffer("P", P)
        self.register_buffer("Q", Q)
        self.register_buffer("bias", base_layer.bias.data.clone() if base_layer.bias is not None else None)
        
        # Store dimensions
        self.out_channels = out_channels
        self.in_channels = base_layer.in_channels
        self.kernel_size = base_layer.kernel_size
        self.stride = base_layer.stride
        self.padding = base_layer.padding
        self.dilation = base_layer.dilation
        self.groups = base_layer.groups
        
        # Rotor parameters
        d_in = P.shape[0]  # flattened input dimension
        self.rank = rank
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.V = nn.Parameter(torch.zeros(d_in, rank))
        
    def get_rotor(self):
        """Compute the orthogonal rotation matrix using Cayley transform.
        
        Returns:
            Orthogonal matrix R that preserves distances and angles
        """
        A = self.U @ self.V.T - self.V @ self.U.T
        I = torch.eye(A.size(0), device=A.device)
        
        # Cayley transform: R = (I - 0.5A)^-1 (I + 0.5A)
        M_left = I - 0.5 * A + 1e-4 * I
        M_right = I + 0.5 * A
        
        # MPS Fix: linalg.solve backward is not implemented on MPS
        if A.device.type == 'mps':
            return torch.linalg.solve(M_left.cpu(), M_right.cpu()).to(A.device)
        
        return torch.linalg.solve(M_left, M_right)
    
    def forward(self, x):
        """Forward pass with geometric adaptation."""
        R = self.get_rotor()
        Wp = self.Q @ R @ self.P  # (out_channels, d_in)
        
        # Reshape back to conv weight shape
        Wp = Wp.view(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1]
        )
        
        return F.conv2d(
            x, Wp, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )
    
    def get_trainable_params(self):
        """Return number of trainable parameters."""
        return self.U.numel() + self.V.numel()
    
    def check_orthogonality(self):
        """Check how orthogonal the rotor is (for debugging/analysis).
        
        Returns:
            Frobenius norm of (R^T R - I), should be close to 0
        """
        R = self.get_rotor()
        I = torch.eye(R.size(0), device=R.device)
        return torch.norm(R.T @ R - I, p='fro').item()
