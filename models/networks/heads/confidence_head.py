from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from .base_head import BaseHead
from .sampling import BayesianSampler

class ParametricConfidenceHead(BaseHead):
    """Parametric confidence head that outputs a single confidence value per asset."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_assets: int,
        device: str = "cuda"
    ):
        """
        Initialize the parametric confidence head.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for processing
            n_assets: Number of assets
            device: Device to run the head on
        """
        super().__init__(input_dim, hidden_dim, device)
        self.n_assets = n_assets
        
        # Confidence estimation layer
        self.confidence_layer = nn.Sequential(
            nn.Linear(hidden_dim, n_assets),
            nn.Sigmoid()  # Ensure confidence is between 0 and 1
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input features and output confidence values.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Dictionary containing confidence values
        """
        # Add batch dimension if not present
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.processor(x)
        confidences = self.confidence_layer(features)
        
        # Remove batch dimension if input was single sample
        if x.shape[0] == 1 and x.dim() == 2:
            confidences = confidences.squeeze(0)
        
        return {
            "confidences": confidences  # (batch_size, n_assets) or (n_assets,)
        }
    
    def get_output_dim(self) -> int:
        return self.n_assets

class BayesianConfidenceHead(BaseHead):
    """Bayesian confidence head that outputs confidence distribution parameters."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_assets: int,
        device: str = "cuda"
    ):
        """
        Initialize the Bayesian confidence head.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for processing
            n_assets: Number of assets
            device: Device to run the head on
        """
        super().__init__(input_dim, hidden_dim, device)
        self.n_assets = n_assets
        
        # Distribution parameter layers
        self.alpha_layer = nn.Linear(hidden_dim, n_assets).to(device)
        self.beta_layer = nn.Linear(hidden_dim, n_assets).to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input features and output confidence values.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Dictionary containing confidence values
        """
        # Add batch dimension if not present
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.processor(x)
        alphas = self.alpha_layer(features)
        betas = self.beta_layer(features)
        
        # Ensure positive parameters
        alphas = torch.exp(alphas)
        betas = torch.exp(betas)
        
        # Calculate mean of Beta distribution
        confidences = alphas / (alphas + betas)
        
        # Remove batch dimension if input was single sample
        if x.shape[0] == 1 and x.dim() == 2:
            confidences = confidences.squeeze(0)
            alphas = alphas.squeeze(0)
            betas = betas.squeeze(0)
        
        return {
            "confidences": confidences,  # (batch_size, n_assets) or (n_assets,)
            "alphas": alphas,           # (batch_size, n_assets) or (n_assets,)
            "betas": betas             # (batch_size, n_assets) or (n_assets,)
        }
    
    def sample(
        self,
        x: torch.Tensor,
        strategy: str = "thompson",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Sample confidence values from the distribution using specified strategy.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            strategy: Sampling strategy ("thompson", "entropy")
            **kwargs: Additional arguments for the sampling strategy
            
        Returns:
            Dictionary containing sampled confidence values
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
        
        # Ensure confidence values are bounded between 0 and 1
        samples = torch.clamp(samples, 0, 1)
        
        return {
            "confidences": samples,  # (batch_size, n_assets) or (n_assets,)
            "alphas": alphas,        # (batch_size, n_assets) or (n_assets,)
            "betas": betas          # (batch_size, n_assets) or (n_assets,)
        }
    
    def get_output_dim(self) -> int:
        return self.n_assets * 2  # Alpha and beta for each asset 