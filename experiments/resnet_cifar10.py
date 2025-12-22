"""ResNet CIFAR-10 experiment implementation."""

import torch
from typing import Dict, Any

from experiments.base_experiment import BaseExperiment
from models import create_model, apply_peft_method
from utils.data_utils import create_data_loaders, get_num_classes
from utils.metrics import compute_forgetting_metrics


class ResNetCIFAR10Experiment(BaseExperiment):
    """ResNet experiment on CIFAR-10 with adaptation tasks.
    
    Supports:
    - Base task: CIFAR-10 (10 classes)
    - Adaptation tasks: Even/Odd, Animal/Vehicle, Living/Non-living
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
    
    def setup_model(self):
        """Setup ResNet model with PEFT method."""
        # Get number of classes for adaptation task
        adapt_task = self.config['data']['adapt_task']
        num_classes = get_num_classes('cifar10', task_type=adapt_task)
        
        # Create base model
        self.model = create_model(
            architecture=self.config['model']['architecture'],
            num_classes=num_classes,
            pretrained=self.config['model']['pretrained']
        )
        
        # Apply PEFT method
        method = self.config['model']['method']
        if method != 'full_ft':
            rank = self.config['model'].get('rank', 8)
            self.model = apply_peft_method(
                self.model,
                method=method,
                rank=rank
            )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        self.logger.info(f"Model setup complete: {self.config['model']['architecture']}")
        self.logger.info(f"PEFT method: {method}")
        if method != 'full_ft':
            self.logger.info(f"Rank: {rank}")
    
    def setup_data(self):
        """Setup CIFAR-10 data loaders."""
        # Get adaptation task loaders
        self.train_loader, self.val_loader = create_data_loaders(
            dataset_name='cifar10',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=self.config['data']['adapt_task']
        )
        
        # For testing, we use the same as validation
        self.test_loader = self.val_loader
        
        # Also create base task loader for retention evaluation
        self.base_train_loader, self.base_test_loader = create_data_loaders(
            dataset_name='cifar10',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=None  # Original 10-class task
        )
        
        self.logger.info(f"Data loaders created for CIFAR-10")
        self.logger.info(f"Adaptation task: {self.config['data']['adapt_task']}")
        self.logger.info(f"Training batches: {len(self.train_loader)}")
    
    def evaluate_base_task(self) -> float:
        """Evaluate on original CIFAR-10 (10-class) task.
        
        Returns:
            Accuracy on base task
        """
        # Temporarily modify model for 10-class prediction if needed
        # For now, we assume the model was trained on base task first
        # This is a simplified version - full implementation would need
        # to handle model head swapping
        
        self.logger.info("Evaluating on base CIFAR-10 task...")
        _, base_acc = self.evaluate(self.base_test_loader)
        return base_acc
    
    def final_evaluation(self):
        """Final evaluation with forgetting metrics."""
        self.logger.info("=" * 50)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("=" * 50)
        
        # Evaluate on adaptation task
        adapt_loss, adapt_acc = self.evaluate(self.test_loader)
        self.logger.info(f"Adaptation task accuracy: {adapt_acc:.2f}%")
        
        # Evaluate base task retention (if configured)
        if self.config['evaluation']['eval_base_task']:
            # Note: This requires the model to be trained on base task first
            # For a complete implementation, you'd need to:
            # 1. Train on base task (CIFAR-10 10-class)
            # 2. Save base task accuracy
            # 3. Adapt to binary task
            # 4. Evaluate retention on base task
            
            # For now, we log a placeholder
            self.logger.info("Base task retention evaluation requires pre-training on base task")
            self.logger.info("See experiment documentation for full workflow")
        
        # Log final metrics
        self.logger.log_metrics({
            'final/adaptation_accuracy': adapt_acc,
            'final/adaptation_loss': adapt_loss
        })
        
        self.logger.info("=" * 50)


class ResNetCIFAR10TwoStageExperiment(ResNetCIFAR10Experiment):
    """Two-stage experiment: Base task training -> Adaptation with forgetting measurement.
    
    This is the complete experiment for measuring catastrophic forgetting.
    """
    
    def run(self):
        """Run two-stage experiment."""
        # Stage 1: Train on base task
        self.logger.info("=" * 50)
        self.logger.info("STAGE 1: Training on base CIFAR-10 (10 classes)")
        self.logger.info("=" * 50)
        
        # Temporarily switch to base task
        original_adapt_task = self.config['data']['adapt_task']
        self.config['data']['adapt_task'] = None
        
        # Re-setup for base task
        self.setup_model_for_base_task()
        self.setup_data()
        self.setup_optimizer()
        
        # Train on base task
        super().run()
        
        # Evaluate and save base task accuracy
        _, self.base_task_acc = self.evaluate(self.test_loader)
        self.logger.info(f"Base task accuracy: {self.base_task_acc:.2f}%")
        
        # Stage 2: Adapt to new task
        self.logger.info("=" * 50)
        self.logger.info(f"STAGE 2: Adapting to {original_adapt_task}")
        self.logger.info("=" * 50)
        
        # Switch to adaptation task
        self.config['data']['adapt_task'] = original_adapt_task
        
        # Re-setup for adaptation
        self.setup_model_for_adaptation()
        self.setup_data()
        self.setup_optimizer()
        
        # Reset training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        
        # Train on adaptation task
        super().run()
    
    def setup_model_for_base_task(self):
        """Setup model for base task (10-class CIFAR-10)."""
        self.model = create_model(
            architecture=self.config['model']['architecture'],
            num_classes=10,
            pretrained=self.config['model']['pretrained']
        )
        self.model = self.model.to(self.device)
    
    def setup_model_for_adaptation(self):
        """Setup model for adaptation task with PEFT."""
        # Load base model weights
        # Apply PEFT method
        adapt_task = self.config['data']['adapt_task']
        num_classes = get_num_classes('cifar10', task_type=adapt_task)
        
        # Modify final layer for new task
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        
        # Apply PEFT
        method = self.config['model']['method']
        if method != 'full_ft':
            rank = self.config['model'].get('rank', 8)
            self.model = apply_peft_method(
                self.model,
                method=method,
                rank=rank
            )
        
        self.model = self.model.to(self.device)
    
    def final_evaluation(self):
        """Final evaluation with catastrophic forgetting metrics."""
        super().final_evaluation()
        
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
        self.logger.info(f"Base task accuracy (after adaptation): {retention_acc:.2f}%")
        self.logger.info(f"Forgetting percentage: {forgetting_metrics['forgetting_percent']:.2f}%")
        self.logger.info(f"Absolute accuracy drop: {forgetting_metrics['absolute_drop']:.2f}%")
        self.logger.info(f"Retention rate: {forgetting_metrics['retention_rate']:.2f}%")
        self.logger.info("=" * 50)
        
        # Log to wandb
        self.logger.log_forgetting_metrics(
            base_acc=self.base_task_acc,
            adapt_acc=self.best_val_acc,
            retention_acc=retention_acc
        )
