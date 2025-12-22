"""Geometric Fine-Tuning (GFT) implementation using orthogonal transformations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricLinear(nn.Module):
    """Geometric linear layer using orthogonal (rotational) adaptations.
    
    Preserves geometric structure through metric-preserving transformations.
    
    Args:
        base_layer: The base linear layer to adapt
        rank: Rank of the low-rank parameterization (default: 8)
    """
    
    def __init__(self, base_layer, rank=8):
        super().__init__()
        W = base_layer.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        P = Vh.T @ torch.diag(S) @ Vh
        Q = U @ Vh

        self.register_buffer("P", P)
        self.register_buffer("Q", Q)
        self.register_buffer("bias", base_layer.bias.data if base_layer.bias is not None else None)

        d = W.shape[1]
        self.U = nn.Parameter(torch.randn(d, rank) * 0.01)
        self.V = nn.Parameter(torch.zeros(d, rank))
        self.rank = rank

    def get_rotor(self):
        """Compute the orthogonal rotation matrix using Cayley transform.
        
        Returns:
            Orthogonal matrix R that preserves distances and angles
        """
        A = self.U @ self.V.T - self.V @ self.U.T
        I = torch.eye(A.size(0), device=A.device)
        return torch.linalg.solve(I - 0.5*A + 1e-4*I, I + 0.5*A)

    def forward(self, x):
        """Forward pass with geometric adaptation."""
        R = self.get_rotor()
        Wp = self.Q @ R @ self.P
        return F.linear(x, Wp, self.bias)
    
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
