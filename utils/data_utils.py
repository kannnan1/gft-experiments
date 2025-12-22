"""Dataset utilities for loading and preprocessing data."""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional, List
import numpy as np


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    """Get CIFAR-10 transforms.
    
    Args:
        train: Whether for training (with augmentation) or testing
        
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


def get_cifar100_transforms(train: bool = True) -> transforms.Compose:
    """Get CIFAR-100 transforms.
    
    Args:
        train: Whether for training (with augmentation) or testing
        
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])


class BinaryTaskDataset(Dataset):
    """Wrapper dataset for binary classification tasks."""
    
    def __init__(self, base_dataset: Dataset, class_mapping: dict):
        """Initialize binary task dataset.
        
        Args:
            base_dataset: Base dataset (e.g., CIFAR-10)
            class_mapping: Dictionary mapping original classes to binary labels
                          e.g., {0: 0, 2: 0, 4: 0, 6: 0, 8: 0, 1: 1, 3: 1, 5: 1, 7: 1, 9: 1}
        """
        self.base_dataset = base_dataset
        self.class_mapping = class_mapping
        
        # Filter dataset to only include mapped classes
        self.valid_indices = [
            i for i, (_, label) in enumerate(base_dataset)
            if label in class_mapping
        ]
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        image, label = self.base_dataset[real_idx]
        binary_label = self.class_mapping[label]
        return image, binary_label


def get_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    train: bool = True,
    task_type: Optional[str] = None
) -> Dataset:
    """Get dataset by name.
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100')
        data_dir: Directory to store/load data
        train: Whether to get training or test set
        task_type: Optional task type for binary classification
                  ('even_odd', 'animal_vehicle', 'living_nonliving')
        
    Returns:
        PyTorch Dataset
    """
    if dataset_name == 'cifar10':
        transform = get_cifar10_transforms(train=train)
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
        
        # Apply binary task mapping if specified
        if task_type == 'even_odd':
            # Even: 0,2,4,6,8 -> 0, Odd: 1,3,5,7,9 -> 1
            class_mapping = {i: i % 2 for i in range(10)}
            dataset = BinaryTaskDataset(dataset, class_mapping)
        
        elif task_type == 'animal_vehicle':
            # Animals: bird(2), cat(3), deer(4), dog(5), frog(6), horse(7) -> 0
            # Vehicles: airplane(0), automobile(1), ship(8), truck(9) -> 1
            class_mapping = {
                0: 1, 1: 1, 2: 0, 3: 0, 4: 0,
                5: 0, 6: 0, 7: 0, 8: 1, 9: 1
            }
            dataset = BinaryTaskDataset(dataset, class_mapping)
        
        elif task_type == 'living_nonliving':
            # Living: bird(2), cat(3), deer(4), dog(5), frog(6), horse(7) -> 0
            # Non-living: airplane(0), automobile(1), ship(8), truck(9) -> 1
            class_mapping = {
                0: 1, 1: 1, 2: 0, 3: 0, 4: 0,
                5: 0, 6: 0, 7: 0, 8: 1, 9: 1
            }
            dataset = BinaryTaskDataset(dataset, class_mapping)
    
    elif dataset_name == 'cifar100':
        transform = get_cifar100_transforms(train=train)
        dataset = datasets.CIFAR100(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
        
        # For coarse classification, we'd need to map fine to coarse labels
        # This is handled separately if needed
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset


def create_data_loaders(
    dataset_name: str,
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    task_type: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders.
    
    Args:
        dataset_name: Name of dataset
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        task_type: Optional task type for binary classification
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = get_dataset(dataset_name, data_dir, train=True, task_type=task_type)
    test_dataset = get_dataset(dataset_name, data_dir, train=False, task_type=task_type)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_num_classes(dataset_name: str, task_type: Optional[str] = None) -> int:
    """Get number of classes for a dataset/task.
    
    Args:
        dataset_name: Name of dataset
        task_type: Optional task type
        
    Returns:
        Number of classes
    """
    if task_type in ['even_odd', 'animal_vehicle', 'living_nonliving']:
        return 2
    elif dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'cifar100':
        return 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
