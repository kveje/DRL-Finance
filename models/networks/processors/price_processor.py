from typing import Dict, Any
import torch
import torch.nn as nn
from .base_processor import BaseProcessor

class PriceProcessor(BaseProcessor):
    """Processor for price data using CNN architecture."""
    
    def __init__(
        self,
        window_size: int,
        hidden_dim: int,
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
        
        self.processor = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),  # Input: (batch, 1, window_size)
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(hidden_dim),  # Add final normalization
            nn.Tanh()  # Bound the output values
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the price data.
        
        Args:
            x: Price tensor of shape (batch_size, n_assets, window_size)
               or (window_size,) for single price series
            
        Returns:
            Processed tensor of shape (batch_size, n_assets, hidden_dim)
        """
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        batch_size = x.shape[0]
        n_assets = x.shape[1] if len(x.shape) == 3 else 1
        
        # Reshape for CNN processing
        if len(x.shape) == 3:
            # Handle multiple assets case
            x = x.view(batch_size * n_assets, 1, self.window_size)
        
        # Process through CNN
        output = self.processor(x)  # (batch*n_assets, hidden_dim)
        
        # Reshape back to batch format
        output = output.view(batch_size, n_assets, -1)
        
        return output
    
    def get_output_dim(self) -> int:
        return self.hidden_dim 