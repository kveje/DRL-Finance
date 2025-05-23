"""Base backbone for neural networks."""
from typing import Dict, Any
import torch
import torch.nn as nn

class BaseBackbone(nn.Module):
    """Base class for all network backbones."""
    
    def __init__(
        self,
        input_dim: int,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Initialize the base backbone.
        
        Args:
            input_dim: Input dimension of the features
            config: Configuration dictionary for the backbone
            device: Device to run the backbone on
        """
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the backbone.
        
        Returns:
            Output dimension
        """
        raise NotImplementedError("Subclasses must implement get_output_dim method") 