"""Data utilities for CLIP experiments."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List
import json


class CLIPImageDataset(Dataset):
    """Dataset for CLIP image classification."""
    
    def __init__(self, image_paths: List[str], labels: List[int], processor):
        """Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of labels
            processor: CLIP processor
        """
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values, self.labels[idx]


def create_clip_data_loaders(
    task: str,
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    processor = None
) -> Tuple[DataLoader, DataLoader]:
    """Create CLIP data loaders for different tasks.
    
    Args:
        task: Task name ('imagenet', 'imagenet_sketch', 'medical', etc.)
        data_dir: Data directory
        batch_size: Batch size
        num_workers: Number of workers
        processor: CLIP processor
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_path = Path(data_dir)
    
    if task == 'imagenet':
        # ImageNet validation set
        train_dataset = _create_imagenet_dataset(data_path / 'imagenet', processor, split='train')
        val_dataset = _create_imagenet_dataset(data_path / 'imagenet', processor, split='val')
    
    elif task == 'imagenet_sketch':
        # ImageNet-Sketch
        train_dataset = _create_imagenet_sketch_dataset(data_path / 'imagenet-sketch', processor, split='train')
        val_dataset = _create_imagenet_sketch_dataset(data_path / 'imagenet-sketch', processor, split='val')
    
    elif task == 'medical':
        # Medical images (placeholder - would need specific dataset)
        raise NotImplementedError("Medical dataset loader not yet implemented")
    
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def _create_imagenet_dataset(data_path: Path, processor, split: str = 'val'):
    """Create ImageNet dataset.
    
    Note: This is a placeholder. Full implementation would require
    downloading and organizing ImageNet data.
    """
    # Placeholder implementation
    # In practice, you'd use torchvision.datasets.ImageNet or similar
    raise NotImplementedError(
        "ImageNet dataset requires manual download. "
        "Please download ImageNet and organize in standard format."
    )


def _create_imagenet_sketch_dataset(data_path: Path, processor, split: str = 'val'):
    """Create ImageNet-Sketch dataset.
    
    Note: This is a placeholder. Full implementation would require
    downloading ImageNet-Sketch.
    """
    raise NotImplementedError(
        "ImageNet-Sketch dataset requires manual download. "
        "Download from: https://github.com/HaohanWang/ImageNet-Sketch"
    )


def get_imagenet_classes() -> List[str]:
    """Get ImageNet class names.
    
    Returns:
        List of 1000 ImageNet class names
    """
    # Simplified version - full implementation would load from file
    # For now, return placeholder
    return [f"class_{i}" for i in range(1000)]


def get_imagenet_sketch_classes() -> List[str]:
    """Get ImageNet-Sketch class names (same as ImageNet).
    
    Returns:
        List of 1000 class names
    """
    return get_imagenet_classes()
