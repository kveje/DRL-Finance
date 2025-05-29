from typing import Dict, Any
import torch
import torch.nn as nn
from .base_processor import BaseProcessor

class PriceProcessor(BaseProcessor):
    """Processor for price data using CNN architecture."""
    
    def __init__(
        self,
        window_size: int,
        n_assets: int,
        asset_embedding_dim: int = 32,
        hidden_dim: int = 128,
        device: str = "cuda"
    ):
        """
        Initialize the price processor.
        
        Args:
            window_size: Size of the price window
            hidden_dim: Hidden dimension for processing
            device: Device to run the processor on
        """
        super().__init__(input_dim=window_size, hidden_dim=hidden_dim, device=device)
        self.window_size = window_size
        self.n_assets = n_assets
        self.asset_embedding_dim = asset_embedding_dim
        self.hidden_dim = hidden_dim
        
        # CNN for processing price series (for each asset)
        self.asset_encoder = nn.Sequential(
            # First conv block - extract basic patterns (2-3 timesteps)
            nn.Conv1d(in_channels=1, out_channels=asset_embedding_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(asset_embedding_dim//4),
            nn.ReLU(),
            
            # Second conv block - capture medium-term patterns (4-5 timesteps)
            nn.Conv1d(in_channels=asset_embedding_dim//4, out_channels=asset_embedding_dim//2, kernel_size=5, padding=2),
            nn.BatchNorm1d(asset_embedding_dim//2),
            nn.ReLU(),

            # Third conv block - capture longer-term patterns (6-7 timesteps)
            nn.Conv1d(in_channels=asset_embedding_dim//2, out_channels=asset_embedding_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(asset_embedding_dim),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),

            # Flatten
            nn.Flatten(),
        ).to(device)

        # Linear layer to combine asset embeddings
        self.asset_combiner = nn.Sequential(
            nn.Linear(n_assets * asset_embedding_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),

            # Reduce to hidden dimension
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the price data.
        
        Args:
            x: Price tensor of shape (batch_size, n_assets, window_size)
               or (n_assets, window_size) for multiple assets without batch
            
        Returns:
            Processed tensor of shape (batch_size, hidden_dim)
        """
        # Case 1: Single price series
        if len(x.shape) == 2 and x.shape[1] == self.window_size: # multiple price series
            x = x.unsqueeze(0)  # [n_assets, window_size] -> [1, n_assets, window_size]
        
        # Case 2: Batch of multiple assets
        # pass
        
        batch_size, n_assets, window_size = x.shape

        # Step 1: Reshape for per-asset processing
        x = x.reshape(batch_size * n_assets, 1, window_size) # shape: (batch_size * n_assets, 1, window_size)

        # Step 2: Get asset embeddings
        asset_embeddings = self.asset_encoder(x) # shape: (batch_size * n_assets, asset_embedding_dim)

        # Step 3: Reshape back to batch format
        asset_embeddings = asset_embeddings.view(batch_size, n_assets, -1) # shape: (batch_size, n_assets, asset_embedding_dim)

        # Step 4: Reshape to fit combiner input (batch_size, n_assets*asset_embedding_dim)
        asset_embeddings = asset_embeddings.view(batch_size, n_assets * self.asset_embedding_dim) # shape: (batch_size, n_assets*asset_embedding_dim)

        # Step 5: Process through combiner
        output = self.asset_combiner(asset_embeddings) # shape: (batch_size, hidden_dim)
        
        return output 
    
    def get_output_dim(self) -> int:
        return self.hidden_dim
    
    def get_asset_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Utility method to get just the asset embeddings without final combination.
        Useful for analysis/debugging.

        Args:
            x: Price tensor of shape (batch_size, n_assets, window_size) 
               or (n_assets, window_size) for multiple assets without batch
               or (window_size) for single price series

        Returns:
            Asset embeddings of shape (batch_size, n_assets, asset_embedding_dim)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # [window_size] -> [1, window_size]
            x = x.unsqueeze(1)  # [1, window_size] -> [1, 1, window_size]
        elif len(x.shape) == 2 and x.shape[1] == self.window_size:
            x = x.unsqueeze(0)  # [n_assets, window_size] -> [1, n_assets, window_size]
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        batch_size, n_assets, window_size = x.shape

        x_reshaped = x.reshape(batch_size * n_assets, 1, window_size)
        asset_embeddings = self.asset_encoder(x_reshaped) # shape: (batch_size * n_assets, asset_embedding_dim)
        asset_embeddings = asset_embeddings.view(batch_size, n_assets, -1) # shape: (batch_size, n_assets, asset_embedding_dim)
        return asset_embeddings
    
    
