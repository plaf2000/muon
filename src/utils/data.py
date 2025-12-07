"""
Data loading utilities for MNIST, CIFAR, or toy datasets.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple


def load_mnist(
    root: str = "./data",
    batch_size: Optional[int] = None,
    train: bool = True,
    download: bool = True,
    full_batch: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load MNIST dataset.
    
    Args:
        root: Root directory for data storage
        batch_size: Batch size for DataLoader (ignored if full_batch=True)
        train: Whether to load training or test set
        download: Whether to download dataset if not present
        full_batch: If True, return full dataset as tensors (for full-batch training)
        
    Returns:
        If full_batch: Tuple of (data, targets) tensors
        Otherwise: DataLoader for MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    if full_batch:
        # Return full dataset as tensors
        data_list = []
        targets_list = []
        for img, target in dataset:
            data_list.append(img)
            targets_list.append(target)
        
        data = torch.stack(data_list)  # (N, 1, 28, 28)
        targets = torch.tensor(targets_list, dtype=torch.long)  # (N,)
        
        return data, targets
    else:
        # Return DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size if batch_size else 64,
            shuffle=train,
            num_workers=2,
            pin_memory=True
        )


def load_cifar10(
    root: str = "./data",
    batch_size: Optional[int] = None,
    train: bool = True,
    download: bool = True,
    full_batch: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load CIFAR-10 dataset.
    
    Args:
        root: Root directory for data storage
        batch_size: Batch size for DataLoader (ignored if full_batch=True)
        train: Whether to load training or test set
        download: Whether to download dataset if not present
        full_batch: If True, return full dataset as tensors (for full-batch training)
        
    Returns:
        If full_batch: Tuple of (data, targets) tensors
        Otherwise: DataLoader for CIFAR-10 dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )  # CIFAR-10 mean and std
    ])
    
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    if full_batch:
        # Return full dataset as tensors
        data_list = []
        targets_list = []
        for img, target in dataset:
            data_list.append(img)
            targets_list.append(target)
        
        data = torch.stack(data_list)  # (N, 3, 32, 32)
        targets = torch.tensor(targets_list, dtype=torch.long)  # (N,)
        
        return data, targets
    else:
        # Return DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size if batch_size else 64,
            shuffle=train,
            num_workers=2,
            pin_memory=True
        )


def create_toy_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a synthetic toy dataset for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (data, targets) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random data
    data = torch.randn(n_samples, n_features)
    
    # Generate labels based on a simple rule
    # Use first feature to determine class
    thresholds = torch.linspace(-2, 2, n_classes - 1)
    targets = torch.zeros(n_samples, dtype=torch.long)
    for i, threshold in enumerate(thresholds):
        targets[data[:, 0] > threshold] = i + 1
    
    return data, targets
