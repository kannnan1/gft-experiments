"""Metrics computation and tracking for experiments."""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, top_k_accuracy_score


class MetricsTracker:
    """Tracks and computes various metrics for experiments."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def update_train(self, loss: float, acc: float):
        """Update training metrics."""
        self.train_losses.append(loss)
        self.train_accs.append(acc)
    
    def update_val(self, loss: float, acc: float):
        """Update validation metrics."""
        self.val_losses.append(loss)
        self.val_accs.append(acc)
    
    def get_average_train(self) -> Tuple[float, float]:
        """Get average training loss and accuracy."""
        avg_loss = np.mean(self.train_losses) if self.train_losses else 0.0
        avg_acc = np.mean(self.train_accs) if self.train_accs else 0.0
        return avg_loss, avg_acc
    
    def get_average_val(self) -> Tuple[float, float]:
        """Get average validation loss and accuracy."""
        avg_loss = np.mean(self.val_losses) if self.val_losses else 0.0
        avg_acc = np.mean(self.val_accs) if self.val_accs else 0.0
        return avg_loss, avg_acc


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: int = 1) -> float:
    """Compute top-k accuracy.
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        topk: Top-k accuracy (1 for top-1, 5 for top-5)
        
    Returns:
        Accuracy as percentage
    """
    with torch.no_grad():
        if topk == 1:
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            total = targets.size(0)
            return 100.0 * correct / total
        else:
            # Top-k accuracy
            _, pred = outputs.topk(topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
            return 100.0 * correct_k.item() / targets.size(0)


def compute_forgetting_metrics(
    base_acc: float,
    retention_acc: float
) -> Dict[str, float]:
    """Compute catastrophic forgetting metrics.
    
    Args:
        base_acc: Original base task accuracy (before adaptation)
        retention_acc: Base task accuracy after adaptation
        
    Returns:
        Dictionary with forgetting metrics
    """
    forgetting_pct = ((base_acc - retention_acc) / base_acc) * 100 if base_acc > 0 else 0.0
    absolute_drop = base_acc - retention_acc
    
    return {
        'forgetting_percent': forgetting_pct,
        'absolute_drop': absolute_drop,
        'retention_rate': (retention_acc / base_acc) * 100 if base_acc > 0 else 0.0
    }


def compute_distance_preservation(
    W_base: torch.Tensor,
    W_adapted: torch.Tensor,
    n_samples: int = 1000
) -> float:
    """Compute distance preservation score using Spearman correlation.
    
    Measures how well the adaptation preserves pairwise distances in feature space.
    
    Args:
        W_base: Base weight matrix
        W_adapted: Adapted weight matrix
        n_samples: Number of random samples to use
        
    Returns:
        Spearman correlation coefficient (higher is better, ~1.0 is perfect)
    """
    device = W_base.device
    
    # Generate random input samples
    X = torch.randn(n_samples, W_base.shape[1], device=device)
    
    # Compute features in base and adapted space
    F_base = X @ W_base.T
    F_adapted = X @ W_adapted.T
    
    # Compute pairwise distances
    D_base = torch.cdist(F_base, F_base)
    D_adapted = torch.cdist(F_adapted, F_adapted)
    
    # Flatten and compute rank correlation
    d_base_flat = D_base.flatten().cpu().numpy()
    d_adapted_flat = D_adapted.flatten().cpu().numpy()
    
    rho, _ = spearmanr(d_base_flat, d_adapted_flat)
    
    return rho


def compute_singular_value_divergence(
    W_base: torch.Tensor,
    W_adapted: torch.Tensor
) -> float:
    """Compute KL divergence between singular value distributions.
    
    Args:
        W_base: Base weight matrix
        W_adapted: Adapted weight matrix
        
    Returns:
        KL divergence (lower is better, 0 is identical)
    """
    # Compute SVD
    _, S_base, _ = torch.linalg.svd(W_base, full_matrices=False)
    _, S_adapted, _ = torch.linalg.svd(W_adapted, full_matrices=False)
    
    # Normalize to probability distributions
    p = S_base / S_base.sum()
    q = S_adapted / S_adapted.sum()
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    
    # Compute KL divergence
    kl_div = (p * torch.log(p / q)).sum()
    
    return kl_div.item()


def compute_cka_similarity(
    X: torch.Tensor,
    Y: torch.Tensor
) -> float:
    """Compute Centered Kernel Alignment (CKA) similarity.
    
    Used for layer-wise feature similarity analysis.
    
    Args:
        X: First feature matrix [batch, features]
        Y: Second feature matrix [batch, features]
        
    Returns:
        CKA similarity score (0 to 1, higher is more similar)
    """
    def centering(K):
        """Center a kernel matrix."""
        n = K.shape[0]
        unit = torch.ones(n, n, device=K.device)
        I = torch.eye(n, device=K.device)
        H = I - unit / n
        return H @ K @ H
    
    # Compute Gram matrices
    K_X = X @ X.T
    K_Y = Y @ Y.T
    
    # Center the matrices
    K_X_centered = centering(K_X)
    K_Y_centered = centering(K_Y)
    
    # Compute CKA
    numerator = torch.norm(K_X_centered @ K_Y_centered, p='fro') ** 2
    denominator = torch.norm(K_X_centered, p='fro') * torch.norm(K_Y_centered, p='fro')
    
    cka = numerator / (denominator + 1e-10)
    
    return cka.item()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_percent': 100.0 * trainable / total if total > 0 else 0.0
    }
