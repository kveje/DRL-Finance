import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class BaseNetwork(nn.Module):
    """
    Base class for all neural networks used in DRL agents.
    Provides common functionality and interface for all network implementations.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the base network.
        
        Args:
            input_dim: Dimension of the input state
            output_dim: Dimension of the output (actions or value)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use ("relu", "tanh", "sigmoid")
            dropout: Dropout rate
            device: Device to run the network on
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.device = device
        
        # Set activation function
        if activation.lower() == "relu":
            self.activation = F.relu
        elif activation.lower() == "tanh":
            self.activation = F.tanh
        elif activation.lower() == "sigmoid":
            self.activation = F.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build the network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Move network to specified device
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            else:
                x = layer(x)
        return x
    
    def save(self, path: str) -> None:
        """
        Save the network's state dictionary.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        """
        Load the network's state dictionary.
        
        Args:
            path: Path to load the model from
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
    
    def get_parameters(self) -> List[torch.Tensor]:
        """
        Get all network parameters.
        
        Returns:
            List of parameter tensors
        """
        return list(self.parameters()) 