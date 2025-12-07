"""
Generic full-batch training loop utilities.
"""

import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing import Callable, Optional, Dict, List
from tqdm import tqdm


def train_full_batch(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    data: torch.Tensor,
    targets: torch.Tensor,
    num_epochs: int = 100,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    track_curvature: bool = False,
    curvature_fn: Optional[Callable] = None,
    track_frequency: int = 1
) -> Dict[str, List[float]]:
    """
    Generic full-batch training loop.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        loss_fn: Loss function
        data: Full batch of training data
        targets: Full batch of target labels
        num_epochs: Number of training epochs
        device: Device to train on
        verbose: Whether to print training progress
        track_curvature: Whether to track curvature metrics
        curvature_fn: Function to compute curvature (e.g., lambda_max)
        track_frequency: Track curvature every N epochs
        
    Returns:
        Dictionary with training history (loss, accuracy, curvature, etc.)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model = model.to(device)
    data = data.to(device)
    targets = targets.to(device)
    
    history = {
        "loss": [],
        "accuracy": [],
        "lambda_max": [] if track_curvature else None
    }
    
    model.train()
    
    for epoch in range(num_epochs):
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            accuracy = (preds == targets).float().mean().item()
            loss_value = loss.item()
        
        history["loss"].append(loss_value)
        history["accuracy"].append(accuracy)
        
        # Track curvature if requested
        if track_curvature and curvature_fn is not None and (epoch % track_frequency == 0 or epoch == num_epochs - 1):
            try:
                lambda_max = curvature_fn(model, loss_fn, data, targets)
                if history["lambda_max"] is not None:
                    history["lambda_max"].append(lambda_max)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not compute curvature at epoch {epoch}: {e}")
                if history["lambda_max"] is not None:
                    history["lambda_max"].append(None)
        
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch:4d}/{num_epochs}: Loss={loss_value:.4f}, Acc={accuracy:.4f}")
    
    return history


def evaluate_model(
    model: Module,
    loss_fn: Callable,
    data: torch.Tensor,
    targets: torch.Tensor,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate model on given data.
    
    Args:
        model: PyTorch model
        loss_fn: Loss function
        data: Input data
        targets: Target labels
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics (loss, accuracy, etc.)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model = model.to(device)
    data = data.to(device)
    targets = targets.to(device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == targets).float().mean().item()
    
    return {
        "loss": loss.item(),
        "accuracy": accuracy
    }
