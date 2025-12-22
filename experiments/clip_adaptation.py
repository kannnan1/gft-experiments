"""CLIP adaptation experiment implementation for domain shift tasks."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader

from experiments.base_experiment import BaseExperiment
from models import apply_peft_method
from utils.metrics import compute_forgetting_metrics, compute_accuracy


class CLIPAdaptationExperiment(BaseExperiment):
    """CLIP experiment for domain adaptation tasks.
    
    Supports:
    - Base task: ImageNet zero-shot classification
    - Adaptation tasks: ImageNet-Sketch, Medical images, Satellite images
    """
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """Initialize CLIP adaptation experiment.
        
        Args:
            config: Experiment configuration
            seed: Random seed
        """
        super().__init__(config, seed)
        
        # CLIP-specific components
        self.processor = None
        self.text_embeddings = None
        self.base_zero_shot_acc = None
    
    def setup_model(self):
        """Setup CLIP model with PEFT method."""
        # Load CLIP model
        model_name = self.config['model'].get('clip_model', 'openai/clip-vit-base-patch32')
        self.logger.info(f"Loading CLIP model: {model_name}")
        
        clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Wrap CLIP for classification
        num_classes = self.config['data'].get('num_classes', 1000)
        self.model = CLIPClassifier(clip_model, num_classes)
        
        # Apply PEFT method to vision encoder
        method = self.config['model']['method']
        if method != 'full_ft':
            rank = self.config['model'].get('rank', 16)
            self.model.vision_model = apply_peft_method(
                self.model.vision_model,
                method=method,
                rank=rank
            )
            self.logger.info(f"Applied {method} (rank={rank}) to vision encoder")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        self.logger.info("CLIP model setup complete")
    
    def setup_data(self):
        """Setup data loaders for CLIP experiments."""
        # Import dataset utilities
        from utils.clip_data_utils import create_clip_data_loaders
        
        # Create data loaders based on adaptation task
        adapt_task = self.config['data']['adapt_task']
        
        self.train_loader, self.val_loader = create_clip_data_loaders(
            task=adapt_task,
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            processor=self.processor
        )
        
        self.test_loader = self.val_loader
        
        # Create ImageNet loader for base task evaluation
        if self.config['evaluation']['eval_base_task']:
            self.base_train_loader, self.base_test_loader = create_clip_data_loaders(
                task='imagenet',
                data_dir=self.config['paths']['data_dir'],
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                processor=self.processor
            )
        
        self.logger.info(f"Data loaders created for CLIP {adapt_task}")
    
    def evaluate_zero_shot(self, data_loader: DataLoader, class_names: list) -> float:
        """Evaluate CLIP zero-shot performance.
        
        Args:
            data_loader: Data loader
            class_names: List of class names for text prompts
            
        Returns:
            Zero-shot accuracy
        """
        self.model.eval()
        
        # Generate text embeddings for class names
        text_inputs = self.processor(
            text=[f"a photo of a {name}" for name in class_names],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get image features
                image_features = self.model.clip_model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                _, predictions = similarity.max(dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return 100.0 * correct / total
    
    def final_evaluation(self):
        """Final evaluation with zero-shot retention metrics."""
        self.logger.info("=" * 50)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("=" * 50)
        
        # Evaluate on adaptation task
        adapt_loss, adapt_acc = self.evaluate(self.test_loader)
        self.logger.info(f"Adaptation task accuracy: {adapt_acc:.2f}%")
        
        # Evaluate zero-shot retention on ImageNet (if configured)
        if self.config['evaluation']['eval_base_task'] and hasattr(self, 'base_test_loader'):
            # Get ImageNet class names
            from utils.clip_data_utils import get_imagenet_classes
            imagenet_classes = get_imagenet_classes()
            
            retention_acc = self.evaluate_zero_shot(self.base_test_loader, imagenet_classes)
            self.logger.info(f"Zero-shot ImageNet retention: {retention_acc:.2f}%")
            
            # Compute forgetting if we have base accuracy
            if self.base_zero_shot_acc is not None:
                forgetting_metrics = compute_forgetting_metrics(
                    base_acc=self.base_zero_shot_acc,
                    retention_acc=retention_acc
                )
                
                self.logger.info("=" * 50)
                self.logger.info("ZERO-SHOT RETENTION ANALYSIS")
                self.logger.info("=" * 50)
                self.logger.info(f"Base zero-shot accuracy: {self.base_zero_shot_acc:.2f}%")
                self.logger.info(f"Retention zero-shot accuracy: {retention_acc:.2f}%")
                self.logger.info(f"Forgetting percentage: {forgetting_metrics['forgetting_percent']:.2f}%")
                self.logger.info("=" * 50)
                
                # Log to wandb
                self.logger.log_forgetting_metrics(
                    base_acc=self.base_zero_shot_acc,
                    adapt_acc=adapt_acc,
                    retention_acc=retention_acc
                )
        
        # Log final metrics
        self.logger.log_metrics({
            'final/adaptation_accuracy': adapt_acc,
            'final/adaptation_loss': adapt_loss
        })


class CLIPClassifier(nn.Module):
    """Wrapper to use CLIP as a classifier."""
    
    def __init__(self, clip_model: CLIPModel, num_classes: int):
        """Initialize CLIP classifier.
        
        Args:
            clip_model: Pre-trained CLIP model
            num_classes: Number of output classes
        """
        super().__init__()
        self.clip_model = clip_model
        self.vision_model = clip_model.vision_model
        
        # Classification head
        vision_embed_dim = clip_model.config.vision_config.hidden_size
        self.classifier = nn.Linear(vision_embed_dim, num_classes)
    
    def forward(self, pixel_values):
        """Forward pass.
        
        Args:
            pixel_values: Input images
            
        Returns:
            Classification logits
        """
        # Get vision features
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        
        # Classify
        logits = self.classifier(image_embeds)
        return logits
