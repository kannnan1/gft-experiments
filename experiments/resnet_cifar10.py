"""ResNet CIFAR experiment implementation with Geometric Analysis and Stage A/B/C Protocol."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from torchvision import models
import shutil
from pathlib import Path

from experiments.base_experiment import BaseExperiment
from utils.data_utils import create_data_loaders, get_num_classes
from utils.metrics import (
    compute_forgetting_metrics, 
    compute_accuracy,
    compute_subspace_angle_drift,
    compute_pairwise_distance_distortion,
    compute_class_centroid_drift,
    compute_cka_similarity
)
from utils.peft_utils import (
    apply_peft_to_resnet_layer,
    FeatureExtractor
)
from utils.geo_plots import (
    plot_pca_manifold_comparison,
    plot_geometric_metrics_summary
)


class ResNetCIFARExperiment(BaseExperiment):
    """ResNet experiment with Stage A/B/C Protocol and Geometric Manifold Analysis.
    
    Stage A: Base Training (Full Model on Dataset A)
    Stage B: Adaptation (Backbone PEFT + FC Swap on Dataset B)
    Stage C: Forgetting Evaluation (FC Restore + Geometric metrics on Dataset A)
    """
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        super().__init__(config, seed)
        self.base_task_acc = None
        self.base_fc_state = None      # Stores the original FC weights/bias
        self.base_features = {}        # Stores layer3/layer4 features from Stage A
        self.base_targets = None       # Stores targets for manifold plotting
        self.dataset_name = config['data']['base_task']
        self.num_base_classes = 10 if self.dataset_name == 'cifar10' else 100

    def setup_model(self):
        """Stage A: Setup base model."""
        arch = self.config['model']['architecture']
        if arch == 'resnet18':
            self.model = models.resnet18(pretrained=self.config['model']['pretrained'])
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_base_classes)
        elif arch == 'resnet50':
            self.model = models.resnet50(pretrained=self.config['model']['pretrained'])
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_base_classes)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
            
        self.model = self.model.to(self.device)
        self.logger.info(f"Model {arch} initialized for {self.num_base_classes} base classes.")

    def setup_data(self, task_type: Optional[str] = None):
        """Setup data loaders for either Base (None) or Adapted (task_type) tasks."""
        self.train_loader, self.val_loader = create_data_loaders(
            dataset_name=self.dataset_name,
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=task_type
        )
        self.test_loader = self.val_loader
        self.logger.info(f"Data setup: {self.dataset_name} Task={task_type}")

    def capture_features(self, n_batches: int = 5) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Extract layer3 and layer4 features for geometric analysis."""
        self.model.eval()
        extractor = FeatureExtractor(self.model, layers=['layer3', 'layer4'], pool=True)
        
        all_features = {'layer3': [], 'layer4': []}
        all_targets = []
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                if i >= n_batches: break
                inputs = inputs.to(self.device)
                features = extractor(inputs)
                
                for k in all_features:
                    all_features[k].append(features[k].cpu())
                all_targets.append(targets)
        
        extractor.remove_hooks()
        
        # Concatenate
        for k in all_features:
            all_features[k] = torch.cat(all_features[k], dim=0)
        
        return all_features, torch.cat(all_targets, dim=0)

    def adapt_to_task(self):
        """Stage B: Adaptation via Backbone PEFT and FC Swap."""
        self.logger.info("="*50)
        self.logger.info("STAGE B: Adaptation")
        self.logger.info("="*50)
        
        # 1. Capture base features before any backbone warping
        self.logger.info("Capturing base semantic manifold features...")
        self.base_features, self.base_targets = self.capture_features()
        
        # 2. Save base FC head state for Stage C
        self.base_fc_state = {k: v.clone() for k, v in self.model.fc.state_dict().items()}
        
        # 3. Swap FC head for target task
        adapt_task = self.config['data']['adapt_task']
        num_new_classes = 2 # Default for binary
        if self.dataset_name == 'cifar100' and adapt_task == 'coarse':
            num_new_classes = 20
            
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_new_classes).to(self.device)
        self.logger.info(f"Swapped FC head: {self.num_base_classes} -> {num_new_classes} classes.")
        
        # 4. Apply PEFT to layer3 and layer4
        method = self.config['model']['method']
        rank = self.config['model'].get('rank', 8)
        
        # Freeze entire model first
        for param in self.model.parameters():
            param.requires_grad = False
            
        if method in ['lora', 'gft']:
            self.logger.info(f"Applying {method.upper()} to layer3 and layer4 (rank={rank})")
            self.model.layer3 = apply_peft_to_resnet_layer(self.model.layer3, method, rank)
            self.model.layer4 = apply_peft_to_resnet_layer(self.model.layer4, method, rank)
            
            # Ensure the newly added PEFT parameters have requires_grad=True
            # (They should by default, but this is for safety)
            for name, param in self.model.named_parameters():
                if any(x in name for x in ['.A', '.B', '.U', '.V']):
                    param.requires_grad = True
                    
        elif method == 'full_ft':
            self.logger.info("Allowing full backbone fine-tuning.")
            for param in self.model.parameters():
                param.requires_grad = True
        
        # New FC must be trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True
            
        # Log which parameters are actually being trained
        self.logger.info("--- Trainable Parameters for Stage B ---")
        trainable_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.logger.info(f"  [TRAINABLE] {name}: {param.numel():,}")
                trainable_count += param.numel()
        self.logger.info(f"Total trainable parameters: {trainable_count:,}")
        self.logger.info("-" * 40)
            
        self.model = self.model.to(self.device)
        self.setup_optimizer()
        
        # 5. Load Adaptation Data
        self.setup_data(task_type=adapt_task)
        self.current_epoch = 0
        self.best_val_acc = 0.0

    def run(self):
        """Execute Stage A/B/C Protocol."""
        # --- Stage A: Base Training ---
        skip_stage1 = self.config['training'].get('skip_stage1', False)
        self.setup_data(task_type=None) # Base task
        
        if skip_stage1:
            checkpoint_path = self.config['training'].get('stage1_checkpoint')
            self.logger.info(f"Loading Stage A checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        else:
            self.logger.info("STAGE A: Training on base task")
            super().run(finish=False)
            # Save Stage A result
            stage1_path = Path(self.checkpoint_manager.get_best_checkpoint_path()).parent / "stage1_best.pt"
            shutil.copy(self.checkpoint_manager.get_best_checkpoint_path(), stage1_path)
            
        _, self.base_task_acc = self.evaluate(self.test_loader)
        self.logger.info(f"Base Task A Accuracy: {self.base_task_acc:.2f}%")
        
        # --- Stage B: Adaptation ---
        self.adapt_to_task()
        super().run(finish=True)
        
        # Save Stage B result
        best_path = self.checkpoint_manager.get_best_checkpoint_path()
        if best_path:
            stage2_path = Path(best_path).parent / "stage2_best.pt"
            shutil.copy(best_path, stage2_path)
            
        # --- Stage C: Forgetting Evaluation ---
        self.final_evaluation()

    def final_evaluation(self):
        """Stage C: Restore FC10 and perform Geometric Analysis."""
        self.logger.info("="*50)
        self.logger.info("STAGE C: Forgetting Evaluation & Geometric Analysis")
        self.logger.info("="*50)
        
        # 1. Evaluate on Task B (Adaptation)
        self.setup_data(task_type=self.config['data']['adapt_task'])
        _, adapt_acc = self.evaluate(self.test_loader)
        
        # 2. Extract Adapted Features
        adapted_features, _ = self.capture_features()
        
        # 3. Restore FC10 and evaluate Task A (Base)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_base_classes).to(self.device)
        self.model.fc.load_state_dict(self.base_fc_state)
        
        self.setup_data(task_type=None)
        _, retention_acc = self.evaluate(self.test_loader)
        
        # 4. Compute Accuracy Metrics
        forgetting = compute_forgetting_metrics(self.base_task_acc, retention_acc)
        self.logger.info(f"Base: {self.base_task_acc:.2f}%, Retention: {retention_acc:.2f}%")
        self.logger.info(f"Forgetting %: {forgetting['forgetting_percent']:.2f}%")
        
        # 5. PERFORM GEOMETRIC ANALYSIS
        self.logger.info("Computing geometric distortion metrics...")
        plots_dir = Path(self.config['paths']['log_dir']) / self.config['experiment']['exp_id'] / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        geo_results = {}
        for layer in ['layer3', 'layer4']:
            F_b = self.base_features[layer]
            F_a = adapted_features[layer]
            
            # Distance Distortion (Frobenius)
            dist_warp = compute_pairwise_distance_distortion(F_b, F_a)
            # Subspace Drift (PCA Principal Angles)
            drift_deg = compute_subspace_angle_drift(F_b, F_a, k=50)
            # Class Centroid Drift
            center_drift = compute_class_centroid_drift(F_b, F_a, self.base_targets)
            # CKA Similarity
            cka = compute_cka_similarity(F_b, F_a)
            
            geo_results[f"{layer}/Distance distortion"] = dist_warp
            geo_results[f"{layer}/Average subspace drift (deg)"] = drift_deg
            geo_results[f"{layer}/Centroid drift"] = center_drift
            geo_results[f"{layer}/CKA Similarity"] = cka
            
            # Plot PCA Comparison
            plot_pca_manifold_comparison(
                F_b, F_a, self.base_targets, layer, plots_dir, 
                self.config['experiment']['exp_id']
            )
            
        # Log to wandb and console
        self.logger.log_metrics(geo_results, prefix="geometric/")
        plot_geometric_metrics_summary(geo_results, plots_dir, self.config['experiment']['exp_id'])
        
        self.logger.log_forgetting_metrics(self.base_task_acc, adapt_acc, retention_acc)
        self.logger.info("Geometric analysis complete. Plots saved to subdirectory.")


class ResNetCIFAR10Experiment(ResNetCIFARExperiment):
    """Legacy wrapper for CIFAR-10."""
    pass

class ResNetCIFAR100Experiment(ResNetCIFARExperiment):
    """Legacy wrapper for CIFAR-100."""
    pass
