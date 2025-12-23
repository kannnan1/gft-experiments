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
        # Ensure SVD is done on CPU as MPS support is limited/unstable for SVD
        W = base_layer.weight.data.detach().cpu().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        P = (Vh.T @ torch.diag(S) @ Vh).float()
        Q = (U @ Vh).float()

        self.register_buffer("P", P)
        self.register_buffer("Q", Q)
        self.register_buffer("bias", base_layer.bias.data.clone() if base_layer.bias is not None else None)

        d_out, d_in = W.shape
        self.in_features = d_in
        self.out_features = d_out
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.V = nn.Parameter(torch.zeros(d_in, rank))
        self.rank = rank

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
        
        # MPS Fix: linalg.solve backward is not implemented on MPS.
        # We move compute to CPU only for this small dxd matrix solve.
        if A.device.type == 'mps':
            return torch.linalg.solve(M_left.cpu(), M_right.cpu()).to(A.device)
        
        return torch.linalg.solve(M_left, M_right)

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
