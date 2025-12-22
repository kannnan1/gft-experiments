"""BLIP captioning experiment implementation."""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from transformers import BlipForConditionalGeneration, BlipProcessor
from torch.utils.data import DataLoader

from experiments.base_experiment import BaseExperiment
from models import apply_peft_method
from utils.metrics import compute_forgetting_metrics


class BLIPCaptioningExperiment(BaseExperiment):
    """BLIP experiment for image captioning tasks.
    
    Supports:
    - Base task: COCO Captions
    - Adaptation tasks: Medical image captions, Scientific figure captions
    """
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """Initialize BLIP captioning experiment.
        
        Args:
            config: Experiment configuration
            seed: Random seed
        """
        super().__init__(config, seed)
        
        # BLIP-specific components
        self.processor = None
        self.base_cider_score = None
    
    def setup_model(self):
        """Setup BLIP model with PEFT method."""
        # Load BLIP model
        model_name = self.config['model'].get('blip_model', 'Salesforce/blip-image-captioning-base')
        self.logger.info(f"Loading BLIP model: {model_name}")
        
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        
        # Apply PEFT method to vision encoder
        method = self.config['model']['method']
        if method != 'full_ft':
            rank = self.config['model'].get('rank', 16)
            
            # Apply to vision encoder
            self.model.vision_model = apply_peft_method(
                self.model.vision_model,
                method=method,
                rank=rank
            )
            
            # Optionally apply to text decoder
            if self.config['model'].get('adapt_text_decoder', False):
                # This would require adapting transformer layers
                self.logger.info("Text decoder adaptation not yet implemented")
            
            self.logger.info(f"Applied {method} (rank={rank}) to vision encoder")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        self.logger.info("BLIP model setup complete")
    
    def setup_data(self):
        """Setup data loaders for BLIP experiments."""
        # Import dataset utilities
        from utils.blip_data_utils import create_blip_data_loaders
        
        # Create data loaders based on adaptation task
        adapt_task = self.config['data']['adapt_task']
        
        self.train_loader, self.val_loader = create_blip_data_loaders(
            task=adapt_task,
            data_dir=self.config['paths']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            processor=self.processor
        )
        
        self.test_loader = self.val_loader
        
        # Create COCO loader for base task evaluation
        if self.config['evaluation']['eval_base_task']:
            self.base_train_loader, self.base_test_loader = create_blip_data_loaders(
                task='coco',
                data_dir=self.config['paths']['data_dir'],
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                processor=self.processor
            )
        
        self.logger.info(f"Data loaders created for BLIP {adapt_task}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch (override for captioning)."""
        self.model.train()
        self.metrics_tracker.reset()
        
        self.progress.start_epoch(epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            self.metrics_tracker.update_train(loss.item(), 0.0)  # No accuracy for captioning
            
            # Update progress bar
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.progress.update_batch(batch_idx, {'loss': loss.item()})
        
        avg_loss, _ = self.metrics_tracker.get_average_train()
        self.progress.finish_epoch({'loss': avg_loss})
        
        return avg_loss, 0.0
    
    def evaluate(self, data_loader: DataLoader):
        """Evaluate model on captioning task."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss, 0.0  # No accuracy for captioning
    
    def generate_captions(self, data_loader: DataLoader, max_samples: int = 100) -> List[Dict]:
        """Generate captions for evaluation.
        
        Args:
            data_loader: Data loader
            max_samples: Maximum number of samples to generate
            
        Returns:
            List of dictionaries with generated and reference captions
        """
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_samples:
                    break
                
                pixel_values = batch['pixel_values'].to(self.device)
                
                # Generate captions
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=50,
                    num_beams=4
                )
                
                generated_captions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                # Get reference captions
                reference_captions = self.processor.batch_decode(
                    batch['input_ids'],
                    skip_special_tokens=True
                )
                
                for gen, ref in zip(generated_captions, reference_captions):
                    results.append({
                        'generated': gen,
                        'reference': ref
                    })
        
        return results
    
    def compute_caption_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute captioning metrics (BLEU, CIDEr, etc.).
        
        Args:
            results: List of generated and reference captions
            
        Returns:
            Dictionary of metrics
        """
        # This would require pycocoevalcap or similar
        # For now, return placeholder
        self.logger.warning("Caption metrics (CIDEr, BLEU) require pycocoevalcap package")
        
        return {
            'bleu4': 0.0,
            'cider': 0.0,
            'meteor': 0.0
        }
    
    def final_evaluation(self):
        """Final evaluation with caption quality metrics."""
        self.logger.info("=" * 50)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("=" * 50)
        
        # Generate captions on adaptation task
        self.logger.info("Generating captions on adaptation task...")
        adapt_results = self.generate_captions(self.test_loader, max_samples=100)
        
        # Compute metrics
        adapt_metrics = self.compute_caption_metrics(adapt_results)
        self.logger.info(f"Adaptation CIDEr: {adapt_metrics.get('cider', 0):.2f}")
        
        # Show some examples
        self.logger.info("\nSample captions:")
        for i, result in enumerate(adapt_results[:5]):
            self.logger.info(f"  Example {i+1}:")
            self.logger.info(f"    Generated: {result['generated']}")
            self.logger.info(f"    Reference: {result['reference']}")
        
        # Evaluate on base task if configured
        if self.config['evaluation']['eval_base_task'] and hasattr(self, 'base_test_loader'):
            self.logger.info("\nGenerating captions on base COCO task...")
            base_results = self.generate_captions(self.base_test_loader, max_samples=100)
            base_metrics = self.compute_caption_metrics(base_results)
            
            self.logger.info(f"Base task CIDEr: {base_metrics.get('cider', 0):.2f}")
            
            # Compute forgetting
            if self.base_cider_score is not None:
                retention_cider = base_metrics.get('cider', 0)
                forgetting_pct = ((self.base_cider_score - retention_cider) / self.base_cider_score) * 100
                
                self.logger.info("=" * 50)
                self.logger.info("CAPTION QUALITY RETENTION")
                self.logger.info("=" * 50)
                self.logger.info(f"Base CIDEr: {self.base_cider_score:.2f}")
                self.logger.info(f"Retention CIDEr: {retention_cider:.2f}")
                self.logger.info(f"Degradation: {forgetting_pct:.2f}%")
                self.logger.info("=" * 50)
        
        # Log final metrics
        self.logger.log_metrics({
            'final/adaptation_cider': adapt_metrics.get('cider', 0),
            'final/adaptation_bleu4': adapt_metrics.get('bleu4', 0)
        })
