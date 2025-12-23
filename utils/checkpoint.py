"""Checkpoint management for saving and loading model states."""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import glob


class CheckpointManager:
    """Manages model checkpoints with automatic saving and cleanup.
    
    Keeps track of best model and maintains last N checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        exp_id: str,
        name: str,
        seed: int,
        keep_last_n: int = 3,
        save_best: bool = True
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
            exp_id: Experiment ID
            name: Experiment name
            seed: Random seed
            keep_last_n: Number of recent checkpoints to keep
            save_best: Whether to save best model separately
        """
        self.checkpoint_dir = Path(checkpoint_dir) / f"{exp_id}_{name}_seed{seed}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.exp_id = exp_id
        self.seed = seed
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        
        self.best_metric = None
        self.best_epoch = None
    
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """Save a checkpoint.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler (optional)
            metrics: Metrics to save with checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
            'exp_id': self.exp_id,
            'seed': self.seed
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save as last checkpoint
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint if applicable
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.best_epoch = epoch
            if metrics and 'val_acc' in metrics:
                self.best_metric = metrics['val_acc']
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        # Get all epoch checkpoints
        epoch_checkpoints = sorted(
            glob.glob(str(self.checkpoint_dir / "epoch_*.pt")),
            key=os.path.getmtime
        )
        
        # Remove old ones
        if len(epoch_checkpoints) > self.keep_last_n:
            for checkpoint_path in epoch_checkpoints[:-self.keep_last_n]:
                os.remove(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            
        Returns:
            Dictionary with checkpoint metadata (epoch, metrics, etc.)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'exp_id': checkpoint.get('exp_id', ''),
            'seed': checkpoint.get('seed', 0)
        }
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint if it exists."""
        best_path = self.checkpoint_dir / "best.pt"
        return str(best_path) if best_path.exists() else None
    
    def get_last_checkpoint_path(self) -> Optional[str]:
        """Get path to last checkpoint if it exists."""
        last_path = self.checkpoint_dir / "last.pt"
        return str(last_path) if last_path.exists() else None
