"""Base head for trading agents."""
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseHead(nn.Module):
    """Base class for all decision heads."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        device: str = "cuda"
    ):
        """
        Initialize the base head.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for processing
            device: Device to run the head on
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Common processing layers
        self.processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input features through the head.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing the head's outputs
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the head."""
        raise NotImplementedError("Subclasses must implement get_output_dim method")

