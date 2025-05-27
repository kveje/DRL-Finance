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
        max_trade_size: float = 1.0,
        interpreter_type: str = "discrete"
    ):
        """
        Initialize the base action interpreter.
        
        Args:
            n_assets: Number of assets to trade
            max_trade_size: Maximum trade size for any asset
            interpreter_type: Type of interpreter to use
        """
        self.n_assets = n_assets
        self.max_trade_size = max_trade_size
        self.interpreter_type = interpreter_type
    
    @abstractmethod
    def interpret(
        self,
        network_outputs: Dict[str, torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert network outputs to actual actions.
        
        Args:
            network_outputs: Dictionary of network outputs
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (scaled_action, action_choices)
        """
        pass
    
    @abstractmethod
    def interpret_with_log_prob(
        self,
        network_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert network outputs to actions and compute log probabilities.
        
        Args:
            network_outputs: Dictionary of network outputs
            
        Returns:
            Tuple of (scaled_action, action_choices, log_probs)
        """
        pass
    
    @abstractmethod
    def get_q_values(
        self,
        network_outputs: Dict[str, torch.Tensor],
        action_choices: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract Q-values for specific actions from network outputs.
        
        Args:
            network_outputs: Dictionary of network outputs
            action_choicess: Action choice tensor (-1, 0, 1)
            
        Returns:
            Tensor of Q-values for the given actions
        """
        pass
    
    @abstractmethod
    def get_max_q_values(
        self,
        network_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get maximum Q-values from network outputs.
        
        Args:
            network_outputs: Dictionary of network outputs
            
        Returns:
            Tensor of maximum Q-values
        """
        pass
    
    
    @abstractmethod
    def evaluate_actions_log_probs(
        self,
        network_outputs: Dict[str, torch.Tensor],
        action_choices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probabilities of given actions under the current policy.
        
        Args:
            network_outputs: Dictionary of network outputs from the policy network
            action_choices: Tensor of actions to evaluate (batch_size, action_dim)
            
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
        action_choices: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        method: str = "mse"
    ) -> torch.Tensor:
        """
        Compute the loss for training.
        
        Args:
            current_outputs: Current network outputs
            target_outputs: Target network outputs
            action_choices: Actions taken
            rewards: Rewards received
            dones: Done flags
            gamma: Discount factor
            method: Method to use for loss computation
            
        Returns:
            Loss tensor
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the action interpreter."""
        pass

    def get_type(self) -> str:
        """Get the type of the action interpreter."""
        return self.interpreter_type

