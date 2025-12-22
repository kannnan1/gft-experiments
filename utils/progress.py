"""Progress tracking with rich progress bars and real-time metrics display."""

from typing import Optional, Dict
from tqdm import tqdm


class ProgressTracker:
    """Progress tracker with epoch and batch-level progress bars.
    
    Displays real-time metrics including loss, accuracy, learning rate, and ETA.
    """
    
    def __init__(self, total_epochs: int, batches_per_epoch: int):
        """Initialize progress tracker.
        
        Args:
            total_epochs: Total number of training epochs
            batches_per_epoch: Number of batches per epoch
        """
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        
        # Epoch-level progress bar
        self.epoch_pbar = None
        # Batch-level progress bar
        self.batch_pbar = None
    
    def start_epoch(self, epoch: int):
        """Start tracking a new epoch.
        
        Args:
            epoch: Current epoch number (1-indexed)
        """
        if self.epoch_pbar is None:
            self.epoch_pbar = tqdm(
                total=self.total_epochs,
                desc="Training Progress",
                position=0,
                leave=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]'
            )
        
        self.epoch_pbar.set_description(f"Epoch {epoch}/{self.total_epochs}")
        
        # Create batch progress bar for this epoch
        self.batch_pbar = tqdm(
            total=self.batches_per_epoch,
            desc=f"  Epoch {epoch}",
            position=1,
            leave=False,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} batches | {postfix}'
        )
    
    def update_batch(self, batch_idx: int, metrics: Optional[Dict[str, float]] = None):
        """Update batch progress.
        
        Args:
            batch_idx: Current batch index
            metrics: Dictionary of metrics to display (e.g., {'loss': 0.5, 'acc': 0.9})
        """
        if self.batch_pbar is not None:
            if metrics:
                # Format metrics for display
                postfix_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                self.batch_pbar.set_postfix_str(postfix_str)
            self.batch_pbar.update(1)
    
    def finish_epoch(self, epoch_metrics: Optional[Dict[str, float]] = None):
        """Finish current epoch.
        
        Args:
            epoch_metrics: Final metrics for the epoch
        """
        if self.batch_pbar is not None:
            self.batch_pbar.close()
            self.batch_pbar = None
        
        if self.epoch_pbar is not None:
            if epoch_metrics:
                # Format metrics for display
                postfix_str = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
                self.epoch_pbar.set_postfix_str(postfix_str)
            self.epoch_pbar.update(1)
    
    def finish(self):
        """Finish all progress tracking."""
        if self.batch_pbar is not None:
            self.batch_pbar.close()
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()


class SimpleProgressBar:
    """Simple progress bar for evaluation and other tasks."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        """Initialize simple progress bar.
        
        Args:
            total: Total number of items
            desc: Description to display
        """
        self.pbar = tqdm(total=total, desc=desc, leave=True)
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.pbar.update(n)
    
    def set_postfix(self, **kwargs):
        """Set postfix metrics."""
        self.pbar.set_postfix(**kwargs)
    
    def close(self):
        """Close progress bar."""
        self.pbar.close()
