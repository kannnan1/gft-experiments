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
        
        elif task_type == 'gt5_lt5':
            # Classes >= 5 -> 1, Classes < 5 -> 0
            class_mapping = {i: (1 if i >= 5 else 0) for i in range(10)}
            dataset = BinaryTaskDataset(dataset, class_mapping)
        elif task_type == 'cifar10_p':
            # Classes >= 5 -> 1, Classes < 5 -> 0
            class_mapping = {
                0: 6, 1: 2, 2: 9, 3: 0, 4: 8,
                5: 1, 6: 3, 7: 4, 8: 7, 9: 5
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
        
        # Mapping for 20 coarse superclasses (CIFAR-100 standard)
        if task_type == 'coarse':
            coarse_mapping = {
                # aquatic mammals
                4: 0, 30: 0, 55: 0, 72: 0, 95: 0,
                # fish
                1: 1, 32: 1, 67: 1, 73: 1, 91: 1,
                # flowers
                54: 2, 62: 2, 70: 2, 82: 2, 92: 2,
                # food containers
                9: 3, 10: 3, 16: 3, 28: 3, 61: 3,
                # fruit and vegetables
                0: 4, 51: 4, 53: 4, 57: 4, 83: 4,
                # household electrical devices
                22: 5, 39: 5, 40: 5, 86: 5, 87: 5,
                # household furniture
                5: 6, 20: 6, 25: 6, 84: 6, 94: 6,
                # insects
                6: 7, 7: 7, 14: 7, 18: 7, 24: 7,
                # large carnivores
                3: 8, 42: 8, 43: 8, 88: 8, 97: 8,
                # large outdoor man-made objects
                12: 9, 17: 9, 37: 9, 68: 9, 76: 9,
                # large natural outdoor scenes
                23: 10, 33: 10, 49: 10, 60: 10, 71: 10,
                # large omnivores and herbivores
                15: 11, 19: 11, 21: 11, 31: 11, 38: 11,
                # medium-sized mammals
                34: 12, 63: 12, 64: 12, 66: 12, 75: 12,
                # non-insect invertebrates
                26: 13, 45: 13, 77: 13, 79: 13, 99: 13,
                # people
                2: 14, 11: 14, 35: 14, 46: 14, 98: 14,
                # reptiles
                27: 15, 29: 15, 44: 15, 78: 15, 93: 15,
                # small mammals
                36: 16, 50: 16, 65: 16, 74: 16, 80: 16,
                # trees
                47: 17, 52: 17, 56: 17, 59: 17, 96: 17,
                # vehicles 1
                8: 18, 13: 18, 48: 18, 58: 18, 90: 18,
                # vehicles 2
                41: 19, 69: 19, 81: 19, 85: 19, 89: 19
            }
            dataset = BinaryTaskDataset(dataset, coarse_mapping)
    
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
    if task_type in ['even_odd', 'animal_vehicle', 'living_nonliving', 'gt5_lt5']:
        return 2
    elif task_type == 'coarse':
        return 20
    elif dataset_name == 'cifar10':
        return 10
    elif task_type == 'cifar10_p':
        return 10
    elif dataset_name == 'cifar100':
        return 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
