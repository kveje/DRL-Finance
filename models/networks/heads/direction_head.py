"""Direction head for trading agents."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.heads.base_head import BaseHead

class DirectionHead(BaseHead):
    """Direction head for trading agents."""
    def __init__(self, input_dim: int, output_dim: int):
        """Initialize the direction head.
        
        Args:
            input_dim: Dimension of the input state
            output_dim: Dimension of the output (number of assets)
        """
        super().__init__(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim * 3) # 3 heads: buy, sell, hold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return logits.view(-1, self.output_dim, 3)
    
    
