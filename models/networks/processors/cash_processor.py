from typing import Dict, Any
import torch
import torch.nn as nn
from .base_processor import BaseProcessor

class CashProcessor(BaseProcessor):
    """Processor for cash data (balance and portfolio value)."""
    
    def __init__(
        self,
        input_dim: int,  # Usually 2 for [cash_balance, portfolio_value]
        hidden_dim: int,
        device: str = "cuda"
    ):
        """
        Initialize the cash processor.
        
        Args:
            input_dim: Input dimension (number of cash-related features)
            hidden_dim: Hidden dimension for processing
            device: Device to run the processor on
        """
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
        
        self.processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the cash data.
        
        Args:
            x: Cash tensor of shape (batch_size, input_dim)
            
        Returns:
            Processed tensor
        """
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.processor(x)
    
    def get_output_dim(self) -> int:
        return self.hidden_dim 