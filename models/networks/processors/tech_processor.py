from typing import Dict, Any
import torch
import torch.nn as nn
from .base_processor import BaseProcessor

class TechProcessor(BaseProcessor):
    """Processor for technical indicators."""
    
    def __init__(
        self,
        n_assets: int,
        tech_dim: int,  # Number of technical indicators per asset
        hidden_dim: int = 64,
        device: str = "cuda"
    ):
        """
        Initialize the technical indicator processor.
        
        Args:
            n_assets: Number of assets
            tech_dim: Number of technical indicators per asset
            hidden_dim: Hidden dimension for processing
            device: Device to run the processor on
        """
        super().__init__(input_dim=n_assets * tech_dim, hidden_dim=hidden_dim, device=device)
        self.n_assets = n_assets
        self.tech_dim = tech_dim
        
        # First process each asset's technical indicators
        self.processor = nn.Sequential(
            nn.Linear(n_assets * tech_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the technical indicators.
        
        Args:
            x: Technical indicator tensor of shape (batch_size, n_assets, tech_dim)
               or (n_assets, tech_dim) for single asset
            
        Returns:
            Processed tensor of shape (batch_size, hidden_dim)
            or (1, hidden_dim) for single asset
        """
        # Case 1: Multiple assets (no batch dimension)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Case 2: Batch of multiple assets (batch dimension)
        # pass

        batch_size = x.shape[0]

        # Reshape input to (batch_size, n_assets * tech_dim)
        x = x.view(batch_size, -1)

        # Process input
        output = self.processor(x)  # (batch_size, hidden_dim)
        
        return output
    
    def get_output_dim(self) -> int:
        return self.hidden_dim 