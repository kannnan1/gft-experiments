"""Comprehensive experiment logging with wandb, file, and console output."""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class ExperimentLogger:
    """Multi-backend experiment logger with wandb, file, and console support.
    
    Logs training metrics, validation metrics, geometric metrics, and more.
    """
    
    def __init__(
        self,
        exp_name: str,
        exp_id: str,
        config: Dict[str, Any],
        log_dir: str = "./results/logs",
        use_wandb: bool = True,
        wandb_project: str = "gft_finetuning",
        resume: bool = False
    ):
        """Initialize experiment logger.
        
        Args:
            exp_name: Experiment name
            exp_id: Experiment ID (e.g., 'R1.2')
            config: Full experiment configuration
            log_dir: Directory for log files
            use_wandb: Whether to use wandb
            wandb_project: Wandb project name
            resume: Whether resuming from checkpoint
        """
        self.exp_name = exp_name
        self.exp_id = exp_id
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Create log directory
        self.log_dir = Path(log_dir) / exp_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self._setup_file_logger()
        
        # Setup wandb
        if self.use_wandb:
            self._setup_wandb(wandb_project, resume)
        
        # Metrics storage
        self.metrics_history = []
        
        self.info(f"Initialized logger for experiment: {exp_name} ({exp_id})")
        self.info(f"Log directory: {self.log_dir}")
    
    def _setup_file_logger(self):
        """Setup file-based logging."""
        log_file = self.log_dir / f"{self.exp_id}.log"
        
        # Create logger
        self.file_logger = logging.getLogger(f"{self.exp_id}")
        self.file_logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.file_logger.addHandler(fh)
        self.file_logger.addHandler(ch)
    
    def _setup_wandb(self, project: str, resume: bool):
        """Setup wandb logging."""
        try:
            wandb.init(
                project=project,
                name=self.exp_name,
                config=self.config,
                resume="allow" if resume else False,
                id=self.exp_id if resume else None
            )
            self.info("Wandb initialized successfully")
        except Exception as e:
            self.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to all backends.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/epoch number
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        # Log to file
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        step_str = f"Step {step} - " if step is not None else ""
        self.info(f"{step_str}{metrics_str}")
        
        # Store in history
        self.metrics_history.append({
            'step': step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        **kwargs
    ):
        """Log epoch-level metrics.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss (optional)
            val_acc: Validation accuracy (optional)
            lr: Learning rate (optional)
            **kwargs: Additional metrics to log
        """
        metrics = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
        }
        
        if val_loss is not None:
            metrics['val/loss'] = val_loss
        if val_acc is not None:
            metrics['val/accuracy'] = val_acc
        if lr is not None:
            metrics['lr'] = lr
        
        # Add any additional metrics
        metrics.update(kwargs)
        
        self.log_metrics(metrics, step=epoch)
    
    def log_forgetting_metrics(
        self,
        base_acc: float,
        adapt_acc: float,
        retention_acc: float,
        epoch: Optional[int] = None
    ):
        """Log catastrophic forgetting metrics.
        
        Args:
            base_acc: Original base task accuracy
            adapt_acc: Adaptation task accuracy
            retention_acc: Base task accuracy after adaptation
            epoch: Epoch number (optional)
        """
        forgetting_pct = ((base_acc - retention_acc) / base_acc) * 100 if base_acc > 0 else 0
        
        metrics = {
            'base_accuracy': base_acc,
            'adaptation_accuracy': adapt_acc,
            'retention_accuracy': retention_acc,
            'forgetting_percent': forgetting_pct
        }
        
        self.log_metrics(metrics, step=epoch, prefix="forgetting/")
    
    def log_geometric_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """Log geometric preservation metrics.
        
        Args:
            metrics: Dictionary of geometric metrics
            epoch: Epoch number (optional)
        """
        self.log_metrics(metrics, step=epoch, prefix="geometric/")
    
    def log_computational_metrics(self, metrics: Dict[str, Any]):
        """Log computational efficiency metrics.
        
        Args:
            metrics: Dictionary with training_time, gpu_memory, etc.
        """
        self.log_metrics(metrics, prefix="compute/")
    
    def info(self, message: str):
        """Log info message."""
        self.file_logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.file_logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.file_logger.error(message)
    
    def save_metrics_history(self):
        """Save all metrics history to JSON file."""
        # 1. Save historical metrics
        history_file = self.log_dir / f"{self.exp_id}_metrics.json"
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
        # 2. Save experiment metadata/config for easier aggregation
        config_file = self.log_dir / "config.json"
        metadata = {
            'exp_name': self.exp_name,
            'exp_id': self.exp_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        with open(config_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.info(f"Saved metrics history and metadata to {self.log_dir}")
    
    def finish(self):
        """Finish logging and cleanup."""
        self.save_metrics_history()
        
        if self.use_wandb:
            wandb.finish()
        
        self.info("Experiment logging finished")
