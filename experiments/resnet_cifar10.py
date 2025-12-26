"""ResNet CIFAR experiment implementation with Geometric Analysis and Stage A/B/C Protocol."""

import torch
import torch.nn as nn
import copy
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
        
        # Storage for separate Stage components
        self.model_base = None      # Clean Stage A model
        self.loader_a = {}          # Stage A loaders
        self.loader_b = {}          # Stage B loaders

    def setup_model(self):
        """Stage A: Setup base model."""
        arch = self.config['model']['architecture']
        if arch == 'resnet18':
            self.model = models.resnet18(weights=self.config['model']['pretrained'])
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_base_classes)
        elif arch == 'resnet50':
            self.model = models.resnet50(pretrained=self.config['model']['pretrained'])
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_base_classes)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
            
        self.model = self.model.to(self.device)
        self.logger.info(f"Model {arch} initialized for {self.num_base_classes} base classes.")

    def setup_data(self, task_type: Optional[str] = None):
        """Setup data loaders. We now pre-create both A and B sets if task_type is None."""
        # Setup Stage A (Base Task)
        tr_a, te_a = create_data_loaders(
            dataset_name=self.dataset_name,
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=None
        )
        self.loader_a = {'train': tr_a, 'test': te_a}
        
        # Setup Stage B (Adapt Task)
        tr_b, te_b = create_data_loaders(
            dataset_name=self.dataset_name,
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task_type=self.config['data']['adapt_task']
        )
        self.loader_b = {'train': tr_b, 'test': te_b}
        
        # Set active loaders (default to A)
        if task_type == self.config['data']['adapt_task']:
            self.train_loader, self.val_loader = tr_b, te_b
        else:
            self.train_loader, self.val_loader = tr_a, te_a
            
        self.test_loader = self.val_loader
        self.logger.info(f"Data setup: Stage A ({len(tr_a.dataset)} samples), Stage B ({len(tr_b.dataset)} samples)")

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
        #self.base_features, self.base_targets = self.capture_features()
        
        # 2. Save base FC head state and keep a frozen copy of the base model
        self.base_fc_state = {k: v.clone() for k, v in self.model.fc.state_dict().items()}
        self.model_base = copy.deepcopy(self.model)
        self.model_base.eval()
        for p in self.model_base.parameters():
            p.requires_grad = False
        
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
        
        # Log initial LoRA state
        self._log_lora_stats("Initial Stage B")

    def run(self):
        """Execute Stage A/B/C Protocol."""
        # --- Stage A: Base Training ---
        skip_stage1 = self.config['training'].get('skip_stage1', False)
        skip_stage2 = self.config['training'].get('skip_stage2', False)

        self.setup_data(task_type=None) # Base task
        
        if skip_stage1:
            checkpoint_path = self.config['training'].get('stage1_checkpoint')
            self.logger.info(f"Loading Stage A checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Support both full checkpoints and state-only dicts
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
        else:
            if continue_from_stage1:
                checkpoint_path = self.config['training'].get('stage1_checkpoint')
                self.logger.info(f"Loading Stage A checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                # Support both full checkpoints and state-only dicts
                state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                self.model.load_state_dict(state_dict)
            self.logger.info("STAGE A: Training on base task")
            super().run(finish=False)
            
            # 1. Capture and Save explicitly
            stage1_path = Path(self.checkpoint_manager.checkpoint_dir) / "stage1_final_cont.pt" if continue_from_stage1 else Path(self.checkpoint_manager.checkpoint_dir) / "stage1_final.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'accuracy': self.best_val_acc
            }, stage1_path)
            self.logger.info(f"Stage 1 model explicitly saved to {stage1_path}")

            # 2. Also try to copy 'best' if it exists (legacy support)
            best_path = self.checkpoint_manager.get_best_checkpoint_path()
            if best_path:
                shutil.copy(best_path, Path(best_path).parent / "stage1_best.pt")
            
        _, self.base_task_acc = self.evaluate(self.test_loader)
        self.logger.info(f"Base Task A Accuracy: {self.base_task_acc:.2f}%")
        
        if skip_stage2==False:

            # --- Stage B: Adaptation ---
            self.adapt_to_task()
            
            super().run(finish=True)
            
            # Log final LoRA state
            self._log_lora_stats("Final Stage B")
            
            # Save Stage B result explicitly
            stage2_path = Path(self.checkpoint_manager.checkpoint_dir) / "stage2_final.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'accuracy': self.best_val_acc
            }, stage2_path)
            self.logger.info(f"Stage 2 model explicitly saved to {stage2_path}")

            # Legacy copy of best if exists
            best_path = self.checkpoint_manager.get_best_checkpoint_path()
            if best_path:
                shutil.copy(best_path, Path(best_path).parent / "stage2_best.pt")
                
            # --- Stage C: Forgetting Evaluation ---
            self.final_evaluation()

    def final_evaluation(self):
        """Stage C: Restore FC10 and perform Geometric Analysis."""
        self.logger.info("="*50)
        self.logger.info("STAGE C: Forgetting Evaluation & Geometric Analysis")
        self.logger.info("="*50)
        
        # 1. Evaluate Mode B model on Task B (Adaptation)
        self.logger.info("Evaluating ADAPTED model on Task B...")
        _, adapt_acc = self.evaluate(self.loader_b['test'])
        self.logger.info(f"Adapted Model on Task B Acc: {adapt_acc:.2f}%")
        
        # 2. Extract Adapted Features (for Stage B model)
        # Using loader_a['test'] to see how base task features changed
        self.test_loader = self.loader_a['test']
        adapted_features, _ = self.capture_features()
        
        # 3. Restore FC10 and evaluate Task A (Base) on Adapted Backbone
        self.logger.info("Restoring FC10 on Adapted Model for Forgetting Analysis...")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_base_classes).to(self.device).to(self.device)
        self.model.fc.load_state_dict(self.base_fc_state)
        
        # Evaluate Adapted Model on Task A
        _, retention_acc = self.evaluate(self.loader_a['test'])
        
        # 4. Reference: Evaluate BASE Model on Task A (Sanity Check)
        self.logger.info("Evaluating original BASE Model on Task A (Reference)...")
        _, base_ref_acc = self.evaluate_model(self.model_base, self.loader_a['test'])
        self.logger.info(f"Base Model Reference Acc: {base_ref_acc:.2f}%")
        
        # 4. Compute Accuracy Metrics
        forgetting = compute_forgetting_metrics(self.base_task_acc, retention_acc)
        self.logger.info(f"Base: {self.base_task_acc:.2f}%, Retention: {retention_acc:.2f}%")
        self.logger.info(f"Forgetting %: {forgetting['forgetting_percent']:.2f}%")
        
        # 5. PERFORM GEOMETRIC ANALYSIS
        # self.logger.info("Computing geometric distortion metrics...")
        # plots_dir = Path(self.config['paths']['log_dir']) / self.config['experiment']['exp_id'] / "plots"
        # plots_dir.mkdir(parents=True, exist_ok=True)
        
        # geo_results = {}
        # for layer in ['layer3', 'layer4']:
        #     F_b = self.base_features[layer]
        #     F_a = adapted_features[layer]
            
        #     # Distance Distortion (Frobenius)
        #     dist_warp = compute_pairwise_distance_distortion(F_b, F_a)
        #     # Subspace Drift (PCA Principal Angles)
        #     drift_deg = compute_subspace_angle_drift(F_b, F_a, k=50)
        #     # Class Centroid Drift
        #     center_drift = compute_class_centroid_drift(F_b, F_a, self.base_targets)
        #     # CKA Similarity
        #     cka = compute_cka_similarity(F_b, F_a)
            
        #     geo_results[f"{layer}/Distance distortion"] = dist_warp
        #     geo_results[f"{layer}/Average subspace drift (deg)"] = drift_deg
        #     geo_results[f"{layer}/Centroid drift"] = center_drift
        #     geo_results[f"{layer}/CKA Similarity"] = cka
            
        #     # Plot PCA Comparison
        #     plot_pca_manifold_comparison(
        #         F_b, F_a, self.base_targets, layer, plots_dir, 
        #         self.config['experiment']['exp_id']
        #     )
            
        # # Log to wandb and console
        # self.logger.log_metrics(geo_results, prefix="geometric/")
        # plot_geometric_metrics_summary(geo_results, plots_dir, self.config['experiment']['exp_id'])
        
        # self.logger.log_forgetting_metrics(self.base_task_acc, adapt_acc, retention_acc)
        # self.logger.info("Geometric analysis complete. Plots saved to subdirectory.")

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Utility to evaluate any model on any loader without modifying self.model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                acc = compute_accuracy(outputs, targets)
                correct += acc * targets.size(0) / 100.0
                total += targets.size(0)
        
        return total_loss / len(data_loader), 100.0 * correct / total

    def _log_lora_stats(self, stage_name: str):
        """Debug helper to log the magnitude of LoRA weights."""
        lora_params = []
        for name, param in self.model.named_parameters():
            if any(x in name for x in ['.A', '.B']):
                lora_params.append(param.data.abs().mean().item())
        
        if lora_params:
            avg_magnitude = sum(lora_params) / len(lora_params)
            self.logger.info(f"[{stage_name}] Average LoRA parameter magnitude: {avg_magnitude:.6f}")
        else:
            self.logger.info(f"[{stage_name}] No LoRA parameters found.")


class ResNetCIFAR10Experiment(ResNetCIFARExperiment):
    """Legacy wrapper for CIFAR-10."""
    pass

class ResNetCIFAR100Experiment(ResNetCIFARExperiment):
    """Legacy wrapper for CIFAR-100."""
    pass
