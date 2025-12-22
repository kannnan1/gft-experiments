"""Data utilities for BLIP experiments."""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict
import json


class BLIPCaptionDataset(Dataset):
    """Dataset for BLIP image captioning."""
    
    def __init__(self, image_paths: List[str], captions: List[str], processor):
        """Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            captions: List of captions
            processor: BLIP processor
        """
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        caption = self.captions[idx]
        
        # Process image and text
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=50,
            truncation=True
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return encoding


def create_blip_data_loaders(
    task: str,
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    processor = None
) -> Tuple[DataLoader, DataLoader]:
    """Create BLIP data loaders for different tasks.
    
    Args:
        task: Task name ('coco', 'medical', 'scientific', etc.)
        data_dir: Data directory
        batch_size: Batch size
        num_workers: Number of workers
        processor: BLIP processor
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_path = Path(data_dir)
    
    if task == 'coco':
        # COCO Captions
        train_dataset = _create_coco_dataset(data_path / 'coco', processor, split='train')
        val_dataset = _create_coco_dataset(data_path / 'coco', processor, split='val')
    
    elif task == 'medical':
        # Medical image captions (placeholder)
        raise NotImplementedError("Medical caption dataset not yet implemented")
    
    elif task == 'scientific':
        # Scientific figure captions (placeholder)
        raise NotImplementedError("Scientific caption dataset not yet implemented")
    
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Custom collate function for BLIP
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch])
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def _create_coco_dataset(data_path: Path, processor, split: str = 'val'):
    """Create COCO Captions dataset.
    
    Note: This is a placeholder. Full implementation would require
    downloading COCO dataset and annotations.
    """
    # Placeholder implementation
    # In practice, you'd load COCO annotations and create dataset
    raise NotImplementedError(
        "COCO dataset requires manual download. "
        "Please download COCO 2017 from: https://cocodataset.org/"
    )


def load_coco_annotations(annotation_file: str) -> List[Dict]:
    """Load COCO annotation file.
    
    Args:
        annotation_file: Path to COCO annotation JSON
        
    Returns:
        List of annotation dictionaries
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Process annotations
    annotations = []
    for ann in data['annotations']:
        annotations.append({
            'image_id': ann['image_id'],
            'caption': ann['caption']
        })
    
    return annotations
