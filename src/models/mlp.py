"""
Simple MLP model for MNIST classification.
"""

import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for MNIST image classification.
    
    Designed for full-batch training with Muon optimizer.
    All weight matrices are 2D, making them suitable for Muon.
    
    Args:
        input_size: Size of input features (28*28 = 784 for MNIST)
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes (10 for MNIST)
        activation: Activation function to use (default: GELU)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] = [128, 64],
        num_classes: int = 10,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if activation is None:
            activation = nn.GELU()
        
        # Build layers
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            # Linear layer (2D weight matrix - suitable for Muon)
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer (typically optimized with AdamW, not Muon)
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten if input is 2D image
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        # Pass through hidden layers
        x = self.hidden_layers(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
