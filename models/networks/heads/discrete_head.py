from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from .base_head import BaseHead
from .sampling import BayesianSampler

class ParametricDiscreteHead(BaseHead):
    """Parametric discrete head that outputs action probabilities."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_assets: int,
        device: str = "cuda"
    ):
        """
        Initialize the parametric discrete head.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for processing
            n_assets: Number of assets
            device: Device to run the head on
        """
        super().__init__(input_dim, hidden_dim, device)
        self.n_assets = n_assets
        
        # Action probability layers (3 actions: buy, sell, hold)
        self.action_layer = nn.Linear(hidden_dim, n_assets * 3).to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input features and output action probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing action probabilities
        """
        features = self.processor(x)
        logits = self.action_layer(features)
        
        # Reshape to (batch_size, n_assets, 3)
        batch_size = x.shape[0]
        logits = logits.view(batch_size, self.n_assets, 3)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        return {
            "action_probs": probs  # (batch_size, n_assets, 3)
        }
    
    def get_output_dim(self) -> int:
        return self.n_assets * 3

class BayesianDiscreteHead(BaseHead):
    """Bayesian discrete head that outputs action distribution parameters."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_assets: int,
        device: str = "cuda"
    ):
        """
        Initialize the Bayesian discrete head.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for processing
            n_assets: Number of assets
            device: Device to run the head on
        """
        super().__init__(input_dim, hidden_dim, device)
        self.n_assets = n_assets
        
        # Distribution parameter layers for each action
        self.alpha_layer = nn.Linear(hidden_dim, n_assets * 3).to(device)
        self.beta_layer = nn.Linear(hidden_dim, n_assets * 3).to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input features and output action probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing action probabilities
        """
        features = self.processor(x)
        alphas = self.alpha_layer(features)
        betas = self.beta_layer(features)
        
        # Ensure positive parameters
        alphas = torch.exp(alphas)
        betas = torch.exp(betas)
        
        # Reshape to (batch_size, n_assets, 3)
        batch_size = x.shape[0]
        alphas = alphas.view(batch_size, self.n_assets, 3)
        betas = betas.view(batch_size, self.n_assets, 3)
        
        # Convert to probabilities using softmax
        probs = torch.softmax(alphas, dim=-1)
        
        return {
            "action_probs": probs,  # (batch_size, n_assets, 3)
            "alphas": alphas,       # (batch_size, n_assets, 3)
            "betas": betas         # (batch_size, n_assets, 3)
        }
    
    def sample(
        self,
        x: torch.Tensor,
        strategy: str = "thompson",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Sample action probabilities from the distribution using specified strategy.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            strategy: Sampling strategy ("thompson", "entropy")
            **kwargs: Additional arguments for the sampling strategy
            
        Returns:
            Dictionary containing sampled action probabilities
        """
        dist_params = self.forward(x)
        alphas = dist_params["alphas"]
        betas = dist_params["betas"]
        
        if strategy == "thompson":
            samples = BayesianSampler.beta_sample(alphas, betas, **kwargs)
        elif strategy == "entropy":
            samples = BayesianSampler.entropy_sample(alphas, betas, **kwargs)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        # Apply softmax to get valid action probabilities
        samples = torch.softmax(samples, dim=-1)
        
        return {
            "action_probs": samples,  # (batch_size, n_assets, 3)
            "alphas": alphas,         # (batch_size, n_assets, 3)
            "betas": betas           # (batch_size, n_assets, 3)
        }
    
    def get_output_dim(self) -> int:
        return self.n_assets * 3 * 2  # Alpha and beta for each action 