"""ResNet CIFAR-10 experiment implementation with proper two-stage training."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from torchvision import models
import shutil
from pathlib import Path

from experiments.base_experiment import BaseExperiment
from models import LoRALinear, GeometricLinear
from utils.data_utils import create_data_loaders, get_num_classes
from utils.metrics import compute_forgetting_metrics, compute_accuracy
from utils.peft_utils import (
    apply_peft_to_resnet_layer,
    get_class_aggregation_mapping,
    aggregate_logits
)


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
        self.aggregation_mapping = None  # For aggregating base class logits to adapted task
    
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
        """Adapt model to binary task using PEFT on backbone.
        
        NEW APPROACH (C2-Modified):
        1. Keep 10-class head TRAINABLE (not frozen)
        2. Apply PEFT (LoRA/GFT) to backbone layers (layer3, layer4)
        3. During training: aggregate 10-class logits → binary logits
        4. During base eval: use all 10 logits directly
        
        This tests whether backbone features are preserved despite full model fine-tuning.
        """
        self.logger.info("=" * 50)
        self.logger.info("STAGE 2: Adapting to binary task (Approach C2)")
        self.logger.info("=" * 50)
        
        # Save base model state for reference
        self.base_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Get adaptation task and method
        adapt_task = self.config['data']['adapt_task']
        method = self.config['model']['method']
        rank = self.config['model'].get('rank', 8)
        
        self.logger.info(f"Adaptation task: {adapt_task}")
        self.logger.info(f"PEFT method: {method}")
        self.logger.info(f"Rank: {rank}")
        
        # Get aggregation mapping for this task
        self.aggregation_mapping = get_class_aggregation_mapping(adapt_task, num_base_classes=10)
        self.logger.info(f"Class aggregation mapping: {self.aggregation_mapping}")
        
        # 1. Freeze entire model
        for param in self.model.parameters():
            param.requires_grad = False
        self.logger.info("  ✓ Entire model frozen")
            
        # 2. Apply PEFT to backbone layers (layer3 and layer4)
        if method in ['lora', 'gft']:
            self.logger.info(f"Applying {method.upper()} to backbone layers (layer3, layer4)...")
            
            # Apply to layer3 and layer4
            self.model.layer3 = apply_peft_to_resnet_layer(self.model.layer3, method, rank)
            self.model.layer4 = apply_peft_to_resnet_layer(self.model.layer4, method, rank)
            
            # 3. FC head remains FROZEN
            # This is critical: if the head is trainable, the model will just 
            # adjust the head weights/biases and never "pressure" the backbone 
            # to adapt, leading to artificially low forgetting.
            self.model.fc.weight.requires_grad = False
            self.model.fc.bias.requires_grad = False
            self.logger.info("  ✓ FC head (10 classes) is FROZEN (forcing backbone to adapt)")
            
            # Log number of trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"  ✓ Trainable parameters in Stage 2 (PEFT only): {trainable_params:,}")
            
        elif method == 'full_ft':
            # Full fine-tuning: everything is trainable
            for param in self.model.parameters():
                param.requires_grad = True
            self.logger.info("Using full fine-tuning (all parameters trainable)")
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup new data loaders for binary task
        # Note: We still use the original 10-class labels, but will aggregate during training
        self.train_loader, self.val_loader = create_data_loaders(
            dataset_name='cifar10',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=adapt_task  # This gives us binary labels
        )
        self.test_loader = self.val_loader
        
        # Reset optimizer for adaptation
        self.setup_optimizer()
        
        # Reset training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        
        self.logger.info("Adaptation setup complete!")
        self.logger.info(f"  - Backbone: {method.upper()} applied to layer3, layer4")
        self.logger.info("  - FC head: FROZEN (10 classes)")
        self.logger.info("  - Training: Binary task via Collision Mapping (slicing outputs to :2)")
        self.logger.info("  - Evaluation: Full 10-class for base task retention")
    
    def evaluate_base_task(self) -> float:
        """Evaluate on original CIFAR-10 (10-class) task.
        
        COLISSION MAPPING EVALUATION:
        We use the full original 10-class head. This tests whether 
        features for all classes (even those not used in Stage 2) 
        have been warped/smashed by the adaptation.
        
        Returns:
            Accuracy on base task
        """
        self.logger.info("Evaluating retention on base CIFAR-10 task...")
        
        # Create base task data loader (10-class labels)
        _, base_test_loader = create_data_loaders(
            dataset_name='cifar10',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=None  # Original 10-class
        )
        
        # Temporarily clear aggregation mapping to use standard evaluate (full 10-class)
        orig_mapping = self.aggregation_mapping
        self.aggregation_mapping = None
        
        _, base_acc = self.evaluate(base_test_loader)
        
        # Restore mapping
        self.aggregation_mapping = orig_mapping
        
        return base_acc

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with Collision Mapping."""
        # Stage 1 or Full FT doesn't use aggregation mapping
        if self.aggregation_mapping is None:
            return super().train_epoch(epoch)
            
        self.model.train()
        self.metrics_tracker.reset()
        self.progress.start_epoch(epoch)
        
        # Determine number of classes for Collision Mapping
        num_adapted_classes = len(self.aggregation_mapping)
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # Base task logits [B, 10]
            
            # COLLISION MAPPING: 
            # Slice the base output to force new features into old neurons.
            # This creates the "Adaptation Pressure" (The POC approach).
            collision_outputs = outputs[:, :num_adapted_classes]
            
            loss = self.criterion(collision_outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy on adaptation task
            acc = compute_accuracy(collision_outputs, targets)
            
            self.metrics_tracker.update_train(loss.item(), acc)
            
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.progress.update_batch(batch_idx, {'loss': loss.item(), 'acc': acc})
                self.logger.log_metrics(
                    {'loss': loss.item(), 'accuracy': acc, 'lr': self.optimizer.param_groups[0]['lr']},
                    prefix='batch/'
                )
        
        avg_loss, avg_acc = self.metrics_tracker.get_average_train()
        self.progress.finish_epoch({'loss': avg_loss, 'acc': avg_acc})
        return avg_loss, avg_acc

    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Evaluate model with optional Collision Mapping."""
        # If no mapping (Stage 1 or Base Review), use standard evaluation
        if self.aggregation_mapping is None:
            return super().evaluate(data_loader)

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Determine number of classes for Collision Mapping
        num_adapted_classes = len(self.aggregation_mapping)
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                # Check if we are evaluating the ADAPTATION task (targets are 0, 1 etc)
                # If targets max is low (within num_adapted_classes), we use the Collision Neurons.
                # If we are evaluating the BASE task, this loop is not called (self.aggregation_mapping is None)
                if targets.max() < num_adapted_classes:
                    outputs = outputs[:, :num_adapted_classes]
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                acc = compute_accuracy(outputs, targets)
                correct += acc * targets.size(0) / 100.0
                total += targets.size(0)
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        avg_acc = 100.0 * correct / total if total > 0 else 0
        return avg_loss, avg_acc
    
    def run(self):
        """Run two-stage experiment: base task → adaptation."""
        # Check if we should skip stage 1
        skip_stage1 = self.config['training'].get('skip_stage1', False)
        
        if skip_stage1:
            self.logger.info("=" * 50)
            self.logger.info("SKIPPING STAGE 1: Using pre-trained base model")
            self.logger.info("=" * 50)
            
            # Load pre-trained model if path provided
            checkpoint_path = self.config['training'].get('stage1_checkpoint')
            if checkpoint_path and Path(checkpoint_path).exists():
                self.logger.info(f"Loading base model from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                if checkpoint_path:
                    self.logger.error(f"Checkpoint path {checkpoint_path} not found!")
                self.logger.warning("Using current model state for adaptation.")
            
            # Setup data for evaluation of base task
            self.setup_data()
            
            # Evaluate base task to get accuracy
            _, self.base_task_acc = self.evaluate(self.test_loader)
            self.logger.info(f"Base task accuracy: {self.base_task_acc:.2f}%")
        else:
            # Stage 1: Train on base task
            self.logger.info("=" * 50)
            self.logger.info("STAGE 1: Training on base CIFAR-10 (10 classes)")
            self.logger.info("=" * 50)
            
            super().run(finish=False)
            
            # Save base task accuracy
            _, self.base_task_acc = self.evaluate(self.test_loader)
            self.logger.info(f"Base task accuracy: {self.base_task_acc:.2f}%")
            
            # Save Stage 1 model explicitly
            best_path = self.checkpoint_manager.get_best_checkpoint_path()
            if best_path:
                stage1_path = Path(best_path).parent / "stage1_best.pt"
                shutil.copy(best_path, stage1_path)
                self.logger.info(f"Saved Stage 1 best model to {stage1_path}")
        
        # Stage 2: Adapt to binary task
        self.adapt_to_binary_task()
        
        # Train on adaptation task
        super().run(finish=True)
        
        # Save Stage 2 model explicitly
        best_path = self.checkpoint_manager.get_best_checkpoint_path()
        if best_path:
            stage2_path = Path(best_path).parent / "stage2_best.pt"
            shutil.copy(best_path, stage2_path)
            self.logger.info(f"Saved Stage 2 best model to {stage2_path}")
    
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
        if self.base_task_acc is not None:
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
        else:
            self.logger.warning("Base task accuracy is None. Skipping forgetting analysis.")
        
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
        """Adapt to 20 coarse classes using backbone PEFT (Approach C2)."""
        self.logger.info("=" * 50)
        self.logger.info("STAGE 2: Adapting to 20 coarse classes (Approach C2)")
        self.logger.info("=" * 50)
        
        # Save base model state
        self.base_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Get adaptation task and method
        method = self.config['model']['method']
        rank = self.config['model'].get('rank', 16)
        
        self.logger.info(f"PEFT method: {method}")
        self.logger.info(f"Rank: {rank}")
        
        # Get aggregation mapping for CIFAR-100 coarse task
        self.aggregation_mapping = get_class_aggregation_mapping('coarse', num_base_classes=100)
        
        # 1. Freeze entire model
        for param in self.model.parameters():
            param.requires_grad = False
        self.logger.info("  ✓ Entire model frozen")
        
        # 2. Apply PEFT to backbone layers
        if method in ['lora', 'gft']:
            self.logger.info(f"Applying {method.upper()} to backbone layers (layer3, layer4)...")
            self.model.layer3 = apply_peft_to_resnet_layer(self.model.layer3, method, rank)
            self.model.layer4 = apply_peft_to_resnet_layer(self.model.layer4, method, rank)
            
            # 3. FC head remains FROZEN
            self.model.fc.weight.requires_grad = False
            self.model.fc.bias.requires_grad = False
            self.logger.info("  ✓ FC head (100 classes) is FROZEN")
            
            # Log number of trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"  ✓ Trainable parameters in Stage 2 (PEFT only): {trainable_params:,}")
            
        elif method == 'full_ft':
            for param in self.model.parameters():
                param.requires_grad = True
            self.logger.info("Using full fine-tuning")
        
        self.model = self.model.to(self.device)
        
        # Setup coarse label data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            dataset_name='cifar100',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type='coarse'
        )
        self.test_loader = self.val_loader
        
        # Reset optimizer
        self.setup_optimizer()
        self.current_epoch = 0
        self.best_val_acc = 0.0

    def run(self):
        """Run two-stage experiment: CIFAR-100 base task → coarse adaptation."""
        # Check if we should skip stage 1
        skip_stage1 = self.config['training'].get('skip_stage1', False)
        
        if skip_stage1:
            self.logger.info("=" * 50)
            self.logger.info("SKIPPING STAGE 1: Using pre-trained base model")
            self.logger.info("=" * 50)
            
            checkpoint_path = self.config['training'].get('stage1_checkpoint')
            if checkpoint_path and Path(checkpoint_path).exists():
                self.logger.info(f"Loading base model from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            
            # Setup data for evaluation
            self.setup_data()
            
            _, self.base_task_acc = self.evaluate(self.test_loader)
            self.logger.info(f"Base task accuracy: {self.base_task_acc:.2f}%")
        else:
            # Stage 1: Train on base task
            self.logger.info("=" * 50)
            self.logger.info("STAGE 1: Training on base CIFAR-100 (100 classes)")
            self.logger.info("=" * 50)
            
            # Call BaseExperiment.run()
            super(ResNetCIFAR10Experiment, self).run(finish=False)
            
            # Save base task accuracy
            _, self.base_task_acc = self.evaluate(self.test_loader)
            self.logger.info(f"Base task accuracy: {self.base_task_acc:.2f}%")
            
            # Save Stage 1 model explicitly
            best_path = self.checkpoint_manager.get_best_checkpoint_path()
            if best_path:
                stage1_path = Path(best_path).parent / "stage1_best.pt"
                shutil.copy(best_path, stage1_path)
                self.logger.info(f"Saved Stage 1 best model to {stage1_path}")
        
        # Stage 2: Adapt to coarse task
        self.adapt_to_binary_task()
        
        # Train on adaptation task
        # Call BaseExperiment.run() again
        super(ResNetCIFAR10Experiment, self).run(finish=True)
        
        # Save Stage 2 model explicitly
        best_path = self.checkpoint_manager.get_best_checkpoint_path()
        if best_path:
            stage2_path = Path(best_path).parent / "stage2_best.pt"
            shutil.copy(best_path, stage2_path)
            self.logger.info(f"Saved Stage 2 best model to {stage2_path}")

    def evaluate_base_task(self) -> float:
        """Evaluate on original CIFAR-100 (100-class) task."""
        self.logger.info("Evaluating retention on base CIFAR-100 task...")
        
        # Create base task data loader
        _, base_test_loader = create_data_loaders(
            dataset_name='cifar100',
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=None  # Original 100-class
        )
        
        # Evaluate directly
        _, base_acc = self.evaluate(base_test_loader)
        
        return base_acc
