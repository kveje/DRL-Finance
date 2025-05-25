from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from .base_head import BaseHead
from .sampling import BayesianSampler

class ParametricValueHead(BaseHead):
    """Parametric value head that outputs a single value estimate."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_assets: int,
        device: str = "cuda"
    ):
        """
        Initialize the parametric value head.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for processing
            n_assets: Number of assets (used for compatibility)
            device: Device to run the head on
        """
        super().__init__(input_dim, hidden_dim, device)
        self.n_assets = n_assets
        
        # Value estimation layer - output single value
        self.value_layer = nn.Linear(hidden_dim, 1).to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input features and output value estimate.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Dictionary containing value estimate
        """
        # Add batch dimension if not present
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.processor(x)
        value = self.value_layer(features)
        
        # Remove batch dimension if input was single sample
        if x.shape[0] == 1 and x.dim() == 2:
            value = value.squeeze(0)
        
        return {
            "value": value  # (batch_size, 1) or (1,)
        }
    
    def get_output_dim(self) -> int:
        return 1  # Single value estimate

class BayesianValueHead(BaseHead):
    """Bayesian value head that outputs value distribution parameters."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_assets: int,
        device: str = "cuda"
    ):
        """
        Initialize the Bayesian value head.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for processing
            n_assets: Number of assets (used for compatibility)
            device: Device to run the head on
        """
        super().__init__(input_dim, hidden_dim, device)
        self.n_assets = n_assets
        
        # Distribution parameter layers - output single value
        self.mean_layer = nn.Linear(hidden_dim, 1).to(device)
        self.log_std_layer = nn.Linear(hidden_dim, 1).to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input features and output value estimate.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Dictionary containing value estimate
        """
        # Add batch dimension if not present
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.processor(x)
        means = self.mean_layer(features)
        log_stds = self.log_std_layer(features)
        
        # Ensure positive standard deviation
        stds = torch.exp(log_stds)
        
        # Remove batch dimension if input was single sample
        if x.shape[0] == 1 and x.dim() == 2:
            means = means.squeeze(0)
            stds = stds.squeeze(0)
        
        return {
            "value": means,  # (batch_size, 1) or (1,)
            "mean": means,   # (batch_size, 1) or (1,)
            "std": stds     # (batch_size, 1) or (1,)
        }
    
    def sample(
        self,
        x: torch.Tensor,
        strategy: str = "thompson",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Sample values from the distribution using specified strategy.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            strategy: Sampling strategy ("thompson", "optimistic", "ucb")
            **kwargs: Additional arguments for the sampling strategy
            
        Returns:
            Dictionary containing sampled values
        """
        dist_params = self.forward(x)
        means = dist_params["mean"]
        stds = dist_params["std"]
        
        if strategy == "thompson":
            samples = BayesianSampler.thompson_sample(means, stds, **kwargs)
        elif strategy == "optimistic":
            samples = BayesianSampler.optimistic_sample(means, stds, **kwargs)
        elif strategy == "ucb":
            samples = BayesianSampler.ucb_sample(means, stds, **kwargs)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        return {
            "value": samples,  # (batch_size, 1) or (1,)
            "mean": means,     # (batch_size, 1) or (1,)
            "std": stds       # (batch_size, 1) or (1,)
        }
    
    def get_output_dim(self) -> int:
        return 2  # Mean and std for single value 