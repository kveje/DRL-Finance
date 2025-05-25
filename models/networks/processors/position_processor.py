from typing import Dict, Any
import torch
import torch.nn as nn
from .base_processor import BaseProcessor

class PositionProcessor(BaseProcessor):
    """Processor for position data (asset holdings)."""
    
    def __init__(
        self,
        n_assets: int,
        hidden_dim: int = 64,
        device: str = "cuda"
    ):
        """
        Initialize the position processor.
        
        Args:
            n_assets: Number of assets in the portfolio
            hidden_dim: Hidden dimension for processing
            device: Device to run the processor on
        """
        super().__init__(input_dim=n_assets, hidden_dim=hidden_dim, device=device)
        self.n_assets = n_assets
        
        # Use a one-layer network to capture relationships between assets
        self.processor = nn.Sequential(
            nn.Linear(n_assets, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the position data.
        
        Args:
            x: Position tensor of shape (batch_size, n_assets)
            
        Returns:
            Processed tensor
        """
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.processor(x)
    
    def get_output_dim(self) -> int:
        return self.hidden_dim 