from typing import Dict, Any
import torch
import torch.nn as nn

class BaseProcessor(nn.Module):
    """Base class for all observation processors."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        device: str = "cuda"
    ):
        """
        Initialize the base processor.
        
        Args:
            input_dim: Input dimension of the observation
            hidden_dim: Hidden dimension for processing
            device: Device to run the processor on
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the processor.
        
        Returns:
            Output dimension
        """
        raise NotImplementedError("Subclasses must implement get_output_dim method") 