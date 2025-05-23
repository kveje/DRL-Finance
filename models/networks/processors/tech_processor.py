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
        hidden_dim: int,
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
        self.asset_processor = nn.Sequential(
            nn.Linear(tech_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        ).to(device)
        
        # Then combine all assets' processed features
        self.combiner = nn.Sequential(
            nn.Linear(n_assets * (hidden_dim // 2), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the technical indicators.
        
        Args:
            x: Technical indicator tensor of shape (batch_size, n_assets, tech_dim)
               or (batch_size, tech_dim) for single asset
            
        Returns:
            Processed tensor
        """
        # Add batch dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Add asset dimension if needed (single asset case)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        batch_size = x.shape[0]
        n_assets = x.shape[1]
        
        # Process each asset's technical indicators
        processed_assets = []
        for i in range(n_assets):
            asset_tech = x[:, i, :]  # (batch_size, tech_dim)
            processed_asset = self.asset_processor(asset_tech)  # (batch_size, hidden_dim//2)
            processed_assets.append(processed_asset)
        
        # Combine all processed assets
        combined = torch.cat(processed_assets, dim=-1)  # (batch_size, n_assets * hidden_dim//2)
        
        # For single asset case, pad to match expected dimensions
        if n_assets == 1:
            padding = torch.zeros(batch_size, (self.n_assets - 1) * (self.hidden_dim // 2), device=self.device)
            combined = torch.cat([combined, padding], dim=-1)
        
        # Process through combiner
        output = self.combiner(combined)  # (batch_size, hidden_dim)
        
        # Add final normalization and bounding
        output = nn.LayerNorm(self.hidden_dim).to(self.device)(output)
        output = torch.tanh(output)  # Bound output between -1 and 1
        
        return output
    
    def get_output_dim(self) -> int:
        return self.hidden_dim 