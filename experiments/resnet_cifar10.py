"""ResNet CIFAR-10 experiment implementation with proper two-stage training."""

import torch
import torch.nn as nn
from typing import Dict, Any
from torchvision import models

from experiments.base_experiment import BaseExperiment
from models import LoRALinear, GeometricLinear
from utils.data_utils import create_data_loaders, get_num_classes
from utils.metrics import compute_forgetting_metrics


class ResNetCIFAR10Experiment(BaseExperiment):
    """ResNet experiment on CIFAR-10 with proper two-stage adaptation.
    
    Stage 1: Train on base task (CIFAR-10 10-class)
    Stage 2: Adapt final layer with PEFT to binary task
    
    This properly measures catastrophic forgetting.
    """
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """Initialize ResNet CIFAR-10 experiment.
        
        Args:
            config: Experiment configuration
            seed: Random seed
        """
        super().__init__(config, seed)
        
        # Store base task accuracy for forgetting computation
        self.base_task_acc = None
        self.base_model_state = None
    
    def setup_model(self):
        """Setup ResNet model for base task (10-class CIFAR-10)."""
        # Create base model for 10-class CIFAR-10
        if self.config['model']['architecture'] == 'resnet18':
            self.model = models.resnet18(pretrained=self.config['model']['pretrained'])
            self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        elif self.config['model']['architecture'] == 'resnet50':
            self.model = models.resnet50(pretrained=self.config['model']['pretrained'])
            self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        else:
            raise ValueError(f"Unsupported architecture: {self.config['model']['architecture']}")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        self.logger.info(f"Model setup complete: {self.config['model']['architecture']}")
        self.logger.info("Stage 1: Training on base CIFAR-10 (10 classes)")
    
    def setup_data(self):
        """Setup CIFAR-10 data loaders for base task."""
        # Base task: 10-class CIFAR-10
        self.train_loader, self.val_loader = create_data_loaders(
            dataset_name='cifar10',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=None  # Original 10-class task
        )
        
        self.test_loader = self.val_loader
        
        self.logger.info(f"Data loaders created for CIFAR-10 (10 classes)")
        self.logger.info(f"Training batches: {len(self.train_loader)}")
    
    def adapt_to_binary_task(self):
        """Adapt model to binary task using PEFT.
        
        This is the key method that implements proper adaptation:
        1. Save base model state
        2. Replace final layer with binary classifier
        3. Apply PEFT method (LoRA/GFT) to final layer
        4. Train on binary task
        """
        self.logger.info("=" * 50)
        self.logger.info("STAGE 2: Adapting to binary task")
        self.logger.info("=" * 50)
        
        # Save base model state
        self.base_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Get adaptation task
        adapt_task = self.config['data']['adapt_task']
        method = self.config['model']['method']
        
        self.logger.info(f"Adaptation task: {adapt_task}")
        self.logger.info(f"PEFT method: {method}")
        
        # Replace final layer with binary classifier
        in_features = self.model.fc.in_features
        base_fc = nn.Linear(in_features, 2)  # Binary classification
        
        # Copy weights from original fc (first 2 classes)
        with torch.no_grad():
            base_fc.weight.data = self.model.fc.weight.data[:2].clone()
            base_fc.bias.data = self.model.fc.bias.data[:2].clone()
        
        # Apply PEFT method
        if method == 'lora':
            rank = self.config['model'].get('rank', 8)
            self.model.fc = LoRALinear(base_fc, rank=rank)
            self.logger.info(f"Applied LoRA with rank={rank}")
        
        elif method == 'gft':
            rank = self.config['model'].get('rank', 8)
            self.model.fc = GeometricLinear(base_fc, rank=rank)
            self.logger.info(f"Applied GFT with rank={rank}")
        
        elif method == 'full_ft':
            self.model.fc = base_fc
            # Unfreeze all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            self.logger.info("Using full fine-tuning")
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup new data loaders for binary task
        self.train_loader, self.val_loader = create_data_loaders(
            dataset_name='cifar10',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=adapt_task
        )
        self.test_loader = self.val_loader
        
        # Reset optimizer for adaptation
        self.setup_optimizer()
        
        # Reset training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
    
    def evaluate_base_task(self) -> float:
        """Evaluate on original CIFAR-10 (10-class) task.
        
        Returns:
            Accuracy on base task
        """
        self.logger.info("Evaluating retention on base CIFAR-10 task...")
        
        # Create base task data loader
        _, base_test_loader = create_data_loaders(
            dataset_name='cifar10',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=None  # Original 10-class
        )
        
        # Temporarily replace fc with 10-class classifier
        current_fc = self.model.fc
        in_features = current_fc.base.in_features if hasattr(current_fc, 'base') else current_fc.in_features
        
        # Create 10-class fc and load base weights
        temp_fc = nn.Linear(in_features, 10).to(self.device)
        
        # Load base model fc weights
        if self.base_model_state is not None:
            temp_fc.load_state_dict({
                'weight': self.base_model_state['fc.weight'],
                'bias': self.base_model_state['fc.bias']
            })
        
        self.model.fc = temp_fc
        
        # Evaluate
        _, base_acc = self.evaluate(base_test_loader)
        
        # Restore adapted fc
        self.model.fc = current_fc
        
        return base_acc
    
    def run(self):
        """Run two-stage experiment: base task → adaptation."""
        # Stage 1: Train on base task
        self.logger.info("=" * 50)
        self.logger.info("STAGE 1: Training on base CIFAR-10 (10 classes)")
        self.logger.info("=" * 50)
        
        super().run()
        
        # Save base task accuracy
        _, self.base_task_acc = self.evaluate(self.test_loader)
        self.logger.info(f"Base task accuracy: {self.base_task_acc:.2f}%")
        
        # Stage 2: Adapt to binary task
        self.adapt_to_binary_task()
        
        # Train on adaptation task
        super().run()
    
    def final_evaluation(self):
        """Final evaluation with catastrophic forgetting metrics."""
        self.logger.info("=" * 50)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("=" * 50)
        
        # Evaluate on adaptation task
        adapt_loss, adapt_acc = self.evaluate(self.test_loader)
        self.logger.info(f"Adaptation task accuracy: {adapt_acc:.2f}%")
        
        # Evaluate retention on base task
        retention_acc = self.evaluate_base_task()
        
        # Compute forgetting metrics
        forgetting_metrics = compute_forgetting_metrics(
            base_acc=self.base_task_acc,
            retention_acc=retention_acc
        )
        
        self.logger.info("=" * 50)
        self.logger.info("CATASTROPHIC FORGETTING ANALYSIS")
        self.logger.info("=" * 50)
        self.logger.info(f"Base task accuracy (before adaptation): {self.base_task_acc:.2f}%")
        self.logger.info(f"Adaptation task accuracy: {adapt_acc:.2f}%")
        self.logger.info(f"Base task accuracy (after adaptation): {retention_acc:.2f}%")
        self.logger.info(f"Forgetting percentage: {forgetting_metrics['forgetting_percent']:.2f}%")
        self.logger.info(f"Absolute accuracy drop: {forgetting_metrics['absolute_drop']:.2f}%")
        self.logger.info(f"Retention rate: {forgetting_metrics['retention_rate']:.2f}%")
        self.logger.info("=" * 50)
        
        # Log to wandb
        self.logger.log_forgetting_metrics(
            base_acc=self.base_task_acc,
            adapt_acc=adapt_acc,
            retention_acc=retention_acc
        )
        
        # Log final metrics
        self.logger.log_metrics({
            'final/adaptation_accuracy': adapt_acc,
            'final/adaptation_loss': adapt_loss
        })


class ResNetCIFAR100Experiment(ResNetCIFAR10Experiment):
    """ResNet experiment on CIFAR-100 with coarse label adaptation."""
    
    def setup_model(self):
        """Setup ResNet model for CIFAR-100 (100 classes)."""
        if self.config['model']['architecture'] == 'resnet18':
            self.model = models.resnet18(pretrained=self.config['model']['pretrained'])
            self.model.fc = nn.Linear(self.model.fc.in_features, 100)
        elif self.config['model']['architecture'] == 'resnet50':
            self.model = models.resnet50(pretrained=self.config['model']['pretrained'])
            self.model.fc = nn.Linear(self.model.fc.in_features, 100)
        else:
            raise ValueError(f"Unsupported architecture: {self.config['model']['architecture']}")
        
        self.model = self.model.to(self.device)
        self.logger.info(f"Model setup complete: {self.config['model']['architecture']}")
        self.logger.info("Stage 1: Training on CIFAR-100 (100 fine classes)")
    
    def setup_data(self):
        """Setup CIFAR-100 data loaders for base task."""
        self.train_loader, self.val_loader = create_data_loaders(
            dataset_name='cifar100',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=None  # Original 100-class task
        )
        
        self.test_loader = self.val_loader
        self.logger.info(f"Data loaders created for CIFAR-100 (100 classes)")
    
    def adapt_to_binary_task(self):
        """Adapt to 20 coarse classes."""
        self.logger.info("=" * 50)
        self.logger.info("STAGE 2: Adapting to 20 coarse classes")
        self.logger.info("=" * 50)
        
        # Save base model state
        self.base_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Replace with 20-class classifier
        in_features = self.model.fc.in_features
        base_fc = nn.Linear(in_features, 20)
        
        method = self.config['model']['method']
        
        if method == 'lora':
            rank = self.config['model'].get('rank', 16)
            self.model.fc = LoRALinear(base_fc, rank=rank)
        elif method == 'gft':
            rank = self.config['model'].get('rank', 16)
            self.model.fc = GeometricLinear(base_fc, rank=rank)
        elif method == 'full_ft':
            self.model.fc = base_fc
            for param in self.model.parameters():
                param.requires_grad = True
        
        self.model = self.model.to(self.device)
        
        # Setup coarse label data loaders
        # Note: This requires implementing coarse label mapping in data_utils
        self.logger.warning("CIFAR-100 coarse label mapping not yet implemented")
        self.logger.warning("Using fine labels for now - implement coarse mapping in data_utils.py")
        
        # Reset optimizer
        self.setup_optimizer()
        self.current_epoch = 0
        self.best_val_acc = 0.0
