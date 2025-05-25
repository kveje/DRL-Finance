from typing import Dict, Any
import torch
import torch.nn as nn
from .base_processor import BaseProcessor

class OHLCVProcessor(BaseProcessor):
    """Simplified version with clearer architecture."""
    
    def __init__(
        self,
        window_size: int = 20,
        asset_embedding_dim: int = 32,
        hidden_dim: int = 256,
        device: str = "cuda",
        n_assets: int = 3,
    ):
        super().__init__(input_dim=window_size * 5, hidden_dim=hidden_dim, device=device)
        self.asset_embedding_dim = asset_embedding_dim
        self.window_size = window_size
        self.n_assets = n_assets
        
        # Simple 2D CNN for each asset's (time, OHLCV) data
        self.feature_extractor = nn.Sequential(
            # Stage 1: OHLCV relationship
            nn.Conv2d(1, asset_embedding_dim//4, kernel_size=(1, 5), padding=(0, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(asset_embedding_dim//4),
            
            # Stage 2: Temporal relationship
            nn.Conv2d(asset_embedding_dim//4, asset_embedding_dim//2, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(asset_embedding_dim//2),

            # Stage 3: Integration of OHLCV and temporal relationship
            nn.Conv2d(asset_embedding_dim//2, asset_embedding_dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(asset_embedding_dim),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ).to(device)
        
        # Create market embedding that preserves asset information
        self.output_layer = nn.Sequential(
            nn.Linear(asset_embedding_dim * n_assets, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: OHLCV tensor with shape:
               - (batch_size, n_assets, window_size, 5) for batched input
               - (n_assets, window_size, 5) for single sample
        Returns:
            (batch_size, hidden_dim) for batched input
            (1, hidden_dim) for single sample
        """
        
        # Case 1: Single sample
        if len(x.shape) == 3:
            # Single sample: (n_assets, window_size, 5) -> (1, n_assets, window_size, 5)
            x = x.unsqueeze(0)
        
        # Case 2: Batched input
        # pass
        
        batch_size, _, _, _ = x.shape # (batch_size, n_assets, window_size, 5)

        # Step 1: Reshape for per-asset processing
        x = x.reshape(batch_size * self.n_assets, 1, self.window_size, 5)

        # Step 2: Get asset embeddings
        asset_embeddings = self.feature_extractor(x)  # (batch * n_assets, asset_embedding_dim)

        # Step 3: Reshape back to batch format
        asset_embeddings = asset_embeddings.view(batch_size, self.n_assets, -1)

        # Step 4: Reshape to fit combiner input (batch_size, n_assets*asset_embedding_dim)
        asset_embeddings = asset_embeddings.view(batch_size, self.n_assets * self.asset_embedding_dim)

        # Step 5: Create market embedding that preserves asset information
        market_embedding = self.output_layer(asset_embeddings)  # (batch_size, hidden_dim)

        # Always return with batch dimension for consistency
        return market_embedding
    
    def get_output_dim(self) -> int:
        return self.hidden_dim
    
    def get_asset_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the asset embeddings for a given input tensor.

        Args:
            x: OHLCV tensor with shape:
               - (batch_size, n_assets, window_size, 5) for batched input
               - (n_assets, window_size, 5) for single sample
               - (window_size, 5) for single asset
        Returns:
            Asset embeddings of shape (batch_size, n_assets, hidden_dim)
        """
        
        # Case 1: Single asset
        if len(x.shape) == 2:
            x = x.unsqueeze(0) # (window_size, 5) -> (1, window_size, 5)
            x = x.unsqueeze(1) # (1, window_size, 5) -> (1, 1, window_size, 5)
        
        # Case 2: Single Sample (non-batched)
        elif len(x.shape) == 3:
            x = x.unsqueeze(0) # (n_assets, window_size, 5) -> (1, n_assets, window_size, 5)
        
        # Case 3: Batched input
        # pass

        batch_size, n_assets, window_size, _ = x.shape # (batch_size, n_assets, window_size, 5)

        # Step 1: Reshape for per-asset processing
        x = x.reshape(batch_size * n_assets, 1, self.window_size, 5)

        # Step 2: Get asset embeddings
        asset_embeddings = self.feature_extractor(x)  # (batch * n_assets, hidden_dim)

        # Step 3: Reshape to (batch_size, n_assets * hidden_dim) - keep all asset info
        asset_embeddings = asset_embeddings.view(batch_size, n_assets * self.hidden_dim)

        # Step 4: Reshape to (batch_size, n_assets, hidden_dim)
        asset_embeddings = asset_embeddings.view(batch_size, n_assets, self.hidden_dim)

        return asset_embeddings
