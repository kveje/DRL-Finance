from typing import Dict, Any
import torch
import torch.nn as nn
from .base_processor import BaseProcessor

class OHLCVProcessor(BaseProcessor):
    """Processor for OHLCV data using CNN and attention mechanisms."""
    
    def __init__(
        self,
        window_size: int,
        hidden_dim: int,
        n_heads: int = 4,
        device: str = "cuda"
    ):
        """
        Initialize the OHLCV processor.
        
        Args:
            window_size: Size of the price window
            hidden_dim: Hidden dimension for processing
            n_heads: Number of attention heads
            device: Device to run the processor on
        """
        super().__init__(input_dim=window_size * 5, hidden_dim=hidden_dim, device=device)  # 5 for OHLCV
        self.window_size = window_size
        self.n_heads = n_heads
        
        # CNN for temporal feature extraction
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(5, hidden_dim, kernel_size=3, padding=1),  # Input: (batch, 5, window_size)
            nn.LayerNorm([hidden_dim, window_size]),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([hidden_dim, window_size]),
            nn.ReLU()
        ).to(device)
        
        # Multi-head self-attention for capturing relationships between price components
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True
        ).to(device)
        
        # Final processing layers
        self.final_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the OHLCV data.
        
        Args:
            x: OHLCV tensor of shape (batch_size, n_assets, window_size, 5)
               or (batch_size, window_size, 5)
            
        Returns:
            Processed tensor
        """
        # Add batch dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        n_assets = x.shape[1] if len(x.shape) == 4 else 1
        
        # Reshape for CNN processing
        if len(x.shape) == 4:
            # Handle multiple assets case
            x = x.view(batch_size * n_assets, self.window_size, 5)
        
        # Transpose for CNN (batch, channels, sequence_length)
        x = x.transpose(1, 2)  # (batch*n_assets, 5, window_size)
        
        # Process through CNN
        temporal_features = self.temporal_cnn(x)  # (batch*n_assets, hidden_dim, window_size)
        
        # Reshape for attention
        temporal_features = temporal_features.transpose(1, 2)  # (batch*n_assets, window_size, hidden_dim)
        
        # Apply self-attention
        attn_output, _ = self.attention(
            temporal_features,
            temporal_features,
            temporal_features
        )
        
        # Global average pooling
        pooled = attn_output.mean(dim=1)  # (batch*n_assets, hidden_dim)
        
        # Reshape back to batch format if needed
        if n_assets > 1:
            pooled = pooled.view(batch_size, n_assets, -1)
            pooled = pooled.mean(dim=1)  # Average across assets
        
        # Final processing
        return self.final_processor(pooled)
    
    def get_output_dim(self) -> int:
        return self.hidden_dim 