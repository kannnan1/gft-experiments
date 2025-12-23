"""Base experiment class with common functionality."""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from utils.logger import ExperimentLogger
from utils.progress import ProgressTracker
from utils.checkpoint import CheckpointManager
from utils.metrics import MetricsTracker, compute_accuracy, count_parameters


class BaseExperiment(ABC):
    """Abstract base class for all experiments.
    
    Provides common functionality for training, evaluation, logging, and checkpointing.
    """
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """Initialize base experiment.
        
        Args:
            config: Experiment configuration dictionary
            seed: Random seed
        """
        self.config = config
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Device configuration with priority: config > auto-detect
        self.device = self._setup_device()
        
        # Initialize components (to be set by subclasses)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Utilities
        self.logger = None
        self.progress = None
        self.checkpoint_manager = None
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
    
    def _setup_device(self) -> torch.device:
        """Setup device with support for cuda, mps, and cpu.
        
        Priority: config setting > auto-detect
        
        Returns:
            torch.device
        """
        # Check if device specified in config
        device_name = self.config.get('device', 'auto')
        
        if device_name == 'auto':
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                print("Using Apple MPS device")
            else:
                device = torch.device('cpu')
                print("Using CPU device")
        else:
            # Use specified device
            device = torch.device(device_name)
            print(f"Using specified device: {device_name}")
        
        return device
    
    @abstractmethod
    def setup_model(self):
        """Setup model architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def setup_data(self):
        """Setup data loaders. Must be implemented by subclasses."""
        pass
    
    def setup(self):
        """Setup all components for the experiment."""
        # 1. Setup logger first so subclasses can use it in setup_model/setup_data
        exp_name = f"{self.config['experiment']['exp_id']}_{self.config['model']['method']}_r{self.config['model'].get('rank', 0)}_seed{self.seed}"
        self.logger = ExperimentLogger(
            exp_name=exp_name,
            exp_id=self.config['experiment']['exp_id'],
            config=self.config,
            log_dir=self.config['paths']['log_dir'],
            use_wandb=self.config['logging']['use_wandb'],
            wandb_project=self.config['logging'].get('wandb_project', 'gft_experiments')
        )
        
        # 2. Setup model and data
        self.setup_model()
        self.setup_data()
        
        # 3. Setup optimizer
        self.setup_optimizer()
        
        # 4. Log model info and setup other utilities
        param_counts = count_parameters(self.model)
        self.logger.info(f"Model: {self.config['model']['architecture']}")
        self.logger.info(f"Method: {self.config['model']['method']}")
        self.logger.info(f"Total parameters: {param_counts['total']:,}")
        self.logger.info(f"Trainable parameters: {param_counts['trainable']:,} ({param_counts['trainable_percent']:.2f}%)")
        self.logger.log_computational_metrics(param_counts)
        
        # Setup progress tracker
        self.progress = ProgressTracker(
            total_epochs=self.config['training']['epochs'],
            batches_per_epoch=len(self.train_loader)
        )
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config['paths']['checkpoint_dir'],
            exp_id=self.config['experiment']['exp_id'],
            name=self.config['experiment']['name'],
            seed=self.seed
        )
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Optimizer
        if self.config['training']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
        
        # Scheduler
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif self.config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        self.metrics_tracker.reset()
        
        self.progress.start_epoch(epoch)
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            acc = compute_accuracy(outputs, targets)
            
            # Update metrics
            self.metrics_tracker.update_train(loss.item(), acc)
            
            # Update progress bar
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.progress.update_batch(
                    batch_idx,
                    {'loss': loss.item(), 'acc': acc}
                )
        
        avg_loss, avg_acc = self.metrics_tracker.get_average_train()
        self.progress.finish_epoch({'loss': avg_loss, 'acc': avg_acc})
        
        return avg_loss, avg_acc
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader to evaluate on
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                acc = compute_accuracy(outputs, targets)
                correct += acc * targets.size(0) / 100.0
                total += targets.size(0)
        
        avg_loss = total_loss / len(data_loader)
        avg_acc = 100.0 * correct / total
        
        return avg_loss, avg_acc
    
    def run(self, finish: bool = True):
        """Main training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        # Early stopping and LR check parameters
        patience = self.config['training'].get('early_stopping_patience', 10)
        min_delta = self.config['training'].get('early_stopping_delta', 0.01)
        lr_threshold = self.config['training'].get('lr_threshold', 1e-8)
        self.patience_counter = 0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr
            )
            
            # Save checkpoint
            is_best = val_acc > (self.best_val_acc + min_delta)
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if epoch % self.config['logging']['save_interval'] == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics={'val_acc': val_acc, 'val_loss': val_loss},
                    is_best=is_best
                )
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Early stopping check
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs)")
                break
                
            # LR threshold check
            if current_lr < lr_threshold:
                self.logger.info(f"Training stopped at epoch {epoch}: Learning rate {current_lr:.2e} below threshold {lr_threshold:.2e}")
                break
        
        # Finish progress tracking
        self.progress.finish()
        
        # Training complete
        total_time = time.time() - start_time
        self.logger.info(f"Training complete! Total time: {total_time/60:.2f} minutes")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Final evaluation and finish logging if requested
        if finish:
            self.final_evaluation()
            self.logger.finish()
    
    @abstractmethod
    def final_evaluation(self):
        """Final evaluation and metrics computation. Must be implemented by subclasses."""
        pass
