"""
Visualization utilities for model predictions and training analysis.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path


def visualize_predictions(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    num_samples: int = 16,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    device: Optional[torch.device] = None
):
    """
    Visualize model predictions on a grid of samples.
    
    Args:
        model: Trained model
        data: Input data tensor (N, C, H, W) or (N, features)
        targets: True labels
        num_samples: Number of samples to visualize
        class_names: List of class names (e.g., ['0', '1', ..., '9'] for MNIST)
        save_path: Path to save the figure
        device: Device to run inference on
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Select random samples
    indices = torch.randperm(len(data))[:num_samples]
    sample_data = data[indices].to(device)
    sample_targets = targets[indices]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(sample_data)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1).cpu()
        confidences = probs.max(dim=1)[0].cpu()
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    # Handle different input shapes
    is_image = len(sample_data.shape) == 4  # (N, C, H, W)
    
    for i, ax in enumerate(axes):
        if i >= num_samples:
            ax.axis('off')
            continue
        
        # Get image
        if is_image:
            img = sample_data[i].cpu()
            # Handle different channel formats
            if img.shape[0] == 1:  # Grayscale
                img = img.squeeze(0)
            elif img.shape[0] == 3:  # RGB
                img = img.permute(1, 2, 0)
            # Denormalize if needed (assuming ImageNet normalization)
            if img.min() < 0:
                img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        else:
            # For flattened data, show as bar chart
            ax.bar(range(len(sample_data[i])), sample_data[i].cpu().numpy())
            ax.set_title(f'Sample {i}')
        
        # Get labels
        true_label = sample_targets[i].item()
        pred_label = preds[i].item()
        confidence = confidences[i].item()
        
        # Format label text
        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
        else:
            true_name = str(true_label)
            pred_name = str(pred_label)
        
        # Color: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_name} | Pred: {pred_name}\nConf: {confidence:.2f}'
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def plot_training_history(
    history: dict,
    save_path: Optional[Path] = None,
    show_lambda_max: bool = True
):
    """
    Plot training history (loss, accuracy, lambda_max).
    
    Args:
        history: Dictionary with keys: train_loss, train_acc, test_loss, test_acc, lambda_max, lambda_max_epochs
        save_path: Path to save the figure
        show_lambda_max: Whether to plot lambda_max on secondary axis
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    epochs = range(len(history["train_loss"]))
    
    # Plot 1: Loss
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, history["test_loss"], label='Test Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, history["train_acc"], label='Train Accuracy', color='blue', linewidth=2)
    ax2.plot(epochs, history["test_acc"], label='Test Accuracy', color='red', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add lambda_max if available (use correct epochs)
    if show_lambda_max and "lambda_max" in history and history["lambda_max"]:
        # Use lambda_max_epochs if available, otherwise infer from indices
        if "lambda_max_epochs" in history and history["lambda_max_epochs"]:
            lambda_epochs = history["lambda_max_epochs"]
            lambda_values = [val for val, epoch in zip(history["lambda_max"], lambda_epochs) if val is not None]
            lambda_epochs = [epoch for val, epoch in zip(history["lambda_max"], lambda_epochs) if val is not None]
        else:
            # Fallback: filter out None values and use indices
            lambda_epochs = [i for i, val in enumerate(history["lambda_max"]) if val is not None]
            lambda_values = [val for val in history["lambda_max"] if val is not None]
        
        if lambda_values:
            ax3 = ax2.twinx()
            ax3.plot(lambda_epochs, lambda_values, label='λ_max', color='green', 
                    marker='o', markersize=4, linewidth=1.5, linestyle='--')
            ax3.set_ylabel('λ_max (Sharpness)', color='green')
            ax3.tick_params(axis='y', labelcolor='green')
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


def plot_lambda_max(
    history: dict,
    save_path: Optional[Path] = None
):
    """
    Plot lambda_max vs epoch (standalone plot).
    
    Args:
        history: Dictionary with keys: lambda_max, lambda_max_epochs
        save_path: Path to save the figure
    """
    if "lambda_max" not in history or not history["lambda_max"]:
        print("No lambda_max data to plot")
        return None
    
    # Get epochs and values
    if "lambda_max_epochs" in history and history["lambda_max_epochs"]:
        lambda_epochs = [epoch for val, epoch in zip(history["lambda_max"], history["lambda_max_epochs"]) if val is not None]
        lambda_values = [val for val, epoch in zip(history["lambda_max"], history["lambda_max_epochs"]) if val is not None]
    else:
        # Fallback
        lambda_epochs = [i for i, val in enumerate(history["lambda_max"]) if val is not None]
        lambda_values = [val for val in history["lambda_max"] if val is not None]
    
    if not lambda_values:
        print("No valid lambda_max values to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lambda_epochs, lambda_values, marker='o', markersize=6, linewidth=2, 
            color='green', label='λ_max')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('λ_max (Largest Hessian Eigenvalue)', fontsize=12)
    ax.set_title('Sharpness Evolution (λ_max)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Lambda_max plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    device: Optional[torch.device] = None
):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        model: Trained model
        data: Input data
        targets: True labels
        class_names: List of class names
        save_path: Path to save the figure
        device: Device to run inference on
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    data = data.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(data)
        preds = outputs.argmax(dim=1).cpu().numpy()
    
    targets_np = targets.cpu().numpy()
    num_classes = len(np.unique(targets_np))
    
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(targets_np, preds):
        cm[true, pred] += 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig
