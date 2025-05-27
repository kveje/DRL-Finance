from typing import Dict, Any
import torch
import torch.nn as nn
from .base_processor import BaseProcessor

class AffordabilityProcessor(BaseProcessor):
    """Processor for affordability data."""
    
    def __init__(
        self,
        n_assets: int,
        hidden_dim: int = 64,
        device: str = "cuda"
    ):
        """
        Initialize the affordability processor.
        
        Args:
            n_assets: Number of assets
            hidden_dim: Hidden dimension for processing
            device: Device to run the processor on
        """
        super().__init__(input_dim=n_assets, hidden_dim=hidden_dim, device=device)
        
        self.processor = nn.Sequential(
            nn.Linear(n_assets, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the affordability data.
        
        Args:
            x: Affordability tensor of shape (batch_size, n_assets) 
            or (n_assets) for single asset
            
        Returns:
            Processed tensor
        """
        # Case 1: Single asset
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Case 2: Batch of multiple assets
        # pass

        return self.processor(x)
    
    def get_output_dim(self) -> int:
        return self.hidden_dim 