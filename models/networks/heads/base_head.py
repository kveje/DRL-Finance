"""Base head for trading agents."""
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseHead(nn.Module):
    """Base head for trading agents."""
    def __init__(self, input_dim: int, output_dim: int):
        """Initialize the base head.
        
        Args:
            input_dim: Dimension of the input state
            output_dim: Dimension of the output (number of assets)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

