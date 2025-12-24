"""Visualization tools for geometric analysis of feature manifolds."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict, List, Optional


def plot_pca_manifold_comparison(
    F_base: torch.Tensor,
    F_adapted: torch.Tensor,
    targets: torch.Tensor,
    layer_name: str,
    output_dir: Path,
    exp_id: str,
    class_names: Optional[List[str]] = None
):
    """Plot PCA scatter plots comparing base vs adapted features.
    
    Args:
        F_base: Base features [batch, dim]
        F_adapted: Adapted features [batch, dim]
        targets: Target labels [batch]
        layer_name: Name of the layer being analyzed
        output_dir: Directory to save plots
        exp_id: Experiment ID
        class_names: List of class names for legend
    """
    # Move to CPU and numpy
    f1 = F_base.detach().cpu().numpy()
    f2 = F_adapted.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    
    # Fit PCA on base features to establish the coordinate system
    pca = PCA(n_components=2)
    f1_pca = pca.fit_transform(f1)
    # Project adapted features onto the same base axes
    f2_pca = pca.transform(f2)
    
    # Setup plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.set_style("whitegrid")
    
    # Unique colors for classes
    n_classes = len(np.unique(y))
    palette = sns.color_palette("husl", n_classes)
    
    # Plot Base
    for i, class_idx in enumerate(np.unique(y)):
        mask = (y == class_idx)
        label = class_names[class_idx] if class_names else f"Class {class_idx}"
        ax1.scatter(f1_pca[mask, 0], f1_pca[mask, 1], color=palette[i], label=label, alpha=0.6, s=30)
    
    ax1.set_title(f"Base Feature Manifold ({layer_name})", fontsize=15)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    
    # Plot Adapted
    for i, class_idx in enumerate(np.unique(y)):
        mask = (y == class_idx)
        ax2.scatter(f2_pca[mask, 0], f2_pca[mask, 1], color=palette[i], alpha=0.6, s=30)
    
    ax2.set_title(f"Adapted Feature Manifold ({layer_name})", fontsize=15)
    ax2.set_xlabel("PC1 (Projected)")
    ax2.set_ylabel("PC2 (Projected)")
    
    # Global legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(n_classes, 5), bbox_to_row=(0, -0.05))
    
    plt.suptitle(f"Semantic Manifold Warp Analysis: {exp_id}", fontsize=20, y=1.02)
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / f"{exp_id}_{layer_name}_PCA.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_geometric_metrics_summary(
    metrics_dict: Dict[str, float],
    output_dir: Path,
    exp_id: str
):
    """Plot a bar chart of geometric distortion metrics.
    
    Args:
        metrics_dict: Dictionary of metrics (e.g., {'Subspace Drift': 12.5, ...})
        output_dir: Directory to save plots
        exp_id: Experiment ID
    """
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Split metrics by scale if necessary, or just plot all
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    ax = sns.barplot(x=labels, y=values, palette="magma")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Metric Value")
    plt.title(f"Geometric Distortion Summary: {exp_id}", fontsize=16)
    
    # Add labels on top
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plot_path = output_dir / f"{exp_id}_geo_summary.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
