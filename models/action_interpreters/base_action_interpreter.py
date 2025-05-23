"""Base class for action interpreters."""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any, Tuple
import numpy as np
import torch

class BaseActionInterpreter(ABC):
    """Base class for action interpreters that convert network outputs to trading actions."""
    
    def __init__(
        self,
        n_assets: int,
        max_position_size: float = 1.0,
        temperature: float = 1.0,
        temperature_decay: float = 0.995,
        min_temperature: float = 0.1
    ):
        """
        Initialize the base action interpreter.
        
        Args:
            n_assets: Number of assets to trade
            max_position_size: Maximum position size for any asset
            temperature: Initial temperature for Bayesian sampling
            temperature_decay: Rate at which temperature decays
            min_temperature: Minimum temperature value
        """
        self.n_assets = n_assets
        self.max_position_size = max_position_size
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
    
    @abstractmethod
    def interpret(
        self,
        network_outputs: Dict[str, Union[np.ndarray, torch.Tensor]],
        current_position: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Convert network outputs to actual actions.
        
        Args:
            network_outputs: Dictionary of network outputs
            current_position: Current position array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Scaled action array
        """
        pass
    
    @abstractmethod
    def interpret_with_log_prob(
        self,
        network_outputs: Dict[str, Union[np.ndarray, torch.Tensor]],
        current_position: np.ndarray
    ) -> Tuple[np.ndarray, Union[np.ndarray, torch.Tensor]]:
        """
        Convert network outputs to actions and compute log probabilities.
        
        Args:
            network_outputs: Dictionary of network outputs
            current_position: Current position array
            
        Returns:
            Tuple of (scaled_action, log_prob)
        """
        pass
    
    @abstractmethod
    def get_q_values(
        self,
        network_outputs: Dict[str, Union[torch.Tensor, np.ndarray]],
        actions: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Extract Q-values for specific actions from network outputs.
        
        Args:
            network_outputs: Dictionary of network outputs
            actions: Action indices or values
            
        Returns:
            Tensor of Q-values for the given actions
        """
        pass
    
    @abstractmethod
    def get_max_q_values(
        self,
        network_outputs: Dict[str, Union[torch.Tensor, np.ndarray]]
    ) -> torch.Tensor:
        """
        Get maximum Q-values from network outputs.
        
        Args:
            network_outputs: Dictionary of network outputs
            
        Returns:
            Tensor of maximum Q-values
        """
        pass
    
    def update_temperature(self) -> None:
        """Update the temperature value for Bayesian sampling."""
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
    
    def get_temperature(self) -> float:
        """Get the current temperature value."""
        return self.temperature
    
    def set_temperature(self, temperature: float) -> None:
        """Set the temperature value."""
        self.temperature = max(self.min_temperature, min(temperature, 1.0))
    
    @abstractmethod
    def get_action_choice(
        self,
        network_outputs: Dict[str, Union[np.ndarray, torch.Tensor]],
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get the unscaled action choice from network outputs.
        
        Args:
            network_outputs: Dictionary of network outputs
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Unscaled action choice array
        """
        pass
    
    @abstractmethod
    def evaluate_actions_log_probs(
        self,
        network_outputs: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probabilities of given actions under the current policy.
        
        Args:
            network_outputs: Dictionary of network outputs from the policy network
            actions: Tensor of actions to evaluate (batch_size, action_dim)
            
        Returns:
            Tuple of (scaled_actions, log_probs) where:
            - scaled_actions are the actual actions to execute
            - log_probs are the log probabilities of the actions
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        current_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """
        Compute the loss for training.
        
        Args:
            current_outputs: Current network outputs
            target_outputs: Target network outputs
            actions: Actions taken
            rewards: Rewards received
            dones: Done flags
            gamma: Discount factor
            
        Returns:
            Loss tensor
        """
        pass

