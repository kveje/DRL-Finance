"""MLP backbone implementation."""
from typing import Dict, Any, List
import torch
import torch.nn as nn
from .base_backbone import BaseBackbone

class MLPBackbone(BaseBackbone):
    """Multi-Layer Perceptron backbone implementation."""
    
    def __init__(
        self,
        input_dim: int,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Initialize the MLP backbone.
        
        Args:
            input_dim: Input dimension of the features
            config: Configuration dictionary containing:
                - hidden_dims: List of hidden dimensions
                - dropout: Dropout rate (default: 0.1)
                - use_layer_norm: Whether to use layer normalization (default: True)
            device: Device to run the backbone on
        """
        super().__init__(input_dim, config, device)
        
        # Extract configuration
        self.hidden_dims = config.get("hidden_dims", [256, 128])
        self.dropout = config.get("dropout", 0.1)
        self.use_layer_norm = config.get("use_layer_norm", True)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Processed tensor of shape (batch_size, hidden_dims[-1])
        """
        return self.mlp(x)
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the MLP.
        
        Returns:
            Output dimension (last hidden dimension)
        """
        return self.hidden_dims[-1] 