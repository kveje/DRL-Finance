from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.distributions as dist

class BayesianSampler:
    """Sampling strategies for Bayesian heads."""
    
    @staticmethod
    def thompson_sample(
        means: torch.Tensor,
        stds: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Thompson sampling for Normal distributions.
        
        Args:
            means: Mean values (batch_size, n_assets)
            stds: Standard deviations (batch_size, n_assets)
            temperature: Controls exploration (higher = more exploration)
            
        Returns:
            Sampled values
        """
        normal = dist.Normal(means, stds * temperature)
        return normal.sample()
    
    @staticmethod
    def beta_sample(
        alphas: torch.Tensor,
        betas: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Thompson sampling for Beta distributions.
        
        Args:
            alphas: Alpha parameters (batch_size, n_assets)
            betas: Beta parameters (batch_size, n_assets)
            temperature: Controls exploration (higher = more exploration)
            
        Returns:
            Sampled values
        """
        # Scale parameters by temperature
        scaled_alphas = alphas / temperature
        scaled_betas = betas / temperature
        
        beta = dist.Beta(scaled_alphas, scaled_betas)
        return beta.sample()
    
    @staticmethod
    def optimistic_sample(
        means: torch.Tensor,
        stds: torch.Tensor,
        optimism_factor: float = 1.0
    ) -> torch.Tensor:
        """
        Optimistic sampling that biases towards higher values.
        
        Args:
            means: Mean values (batch_size, n_assets)
            stds: Standard deviations (batch_size, n_assets)
            optimism_factor: How much to bias towards higher values
            
        Returns:
            Sampled values
        """
        normal = dist.Normal(means + stds * optimism_factor, stds)
        return normal.sample()
    
    @staticmethod
    def entropy_sample(
        alphas: torch.Tensor,
        betas: torch.Tensor,
        entropy_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Entropy-based sampling that balances exploration and exploitation.
        
        Args:
            alphas: Alpha parameters (batch_size, n_assets)
            betas: Beta parameters (batch_size, n_assets)
            entropy_weight: Weight for entropy term
            
        Returns:
            Sampled values
        """
        # Calculate entropy
        entropy = torch.digamma(alphas + betas) - torch.digamma(alphas) - torch.digamma(betas)
        
        # Sample from Beta distribution
        beta = dist.Beta(alphas, betas)
        samples = beta.sample()
        
        # Adjust samples based on entropy
        return samples + entropy * entropy_weight
    
    @staticmethod
    def ucb_sample(
        means: torch.Tensor,
        stds: torch.Tensor,
        exploration_factor: float = 2.0
    ) -> torch.Tensor:
        """
        Upper Confidence Bound (UCB) sampling.
        
        Args:
            means: Mean values (batch_size, n_assets)
            stds: Standard deviations (batch_size, n_assets)
            exploration_factor: Controls exploration-exploitation trade-off
            
        Returns:
            Sampled values
        """
        return means + exploration_factor * stds 