"""Discrete action interpreter implementation."""
import numpy as np
import torch
from typing import Dict, Union, Optional, Tuple
from .base_action_interpreter import BaseActionInterpreter

class DiscreteInterpreter(BaseActionInterpreter):
    """Interpreter for discrete action spaces."""
    
    def __init__(
        self,
        n_assets: int,
        max_trade_size: float = 10,
        temperature: float = 1.0,
        temperature_decay: float = 0.995,
        min_temperature: float = 0.1,
        interpreter_type: str = "discrete"
    ):
        """
        Initialize the discrete action interpreter.
        
        Args:
            n_assets: Number of assets to trade
            max_trade_size: Maximum position size for any asset
            temperature: Initial temperature for Bayesian sampling
            temperature_decay: Rate at which temperature decays
            min_temperature: Minimum temperature value
            interpreter_type: Type of interpreter to use
        """
        super().__init__(
            interpreter_type=interpreter_type,
            n_assets=n_assets,
            max_trade_size=max_trade_size,
        )

    def _get_action_probs(
        self,
        network_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get network outputs from dictionary.
        """
        return network_outputs['action_probs']
    
    def _get_probs_batch_input(
            self,
            x: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        If the input is a single sample, make it a batch.

        Args:
            x: Tensor of shape (n_assets, n_actions) or (batch_size, n_assets, n_actions)

        Returns:
            Tensor of shape (batch_size, n_assets, n_actions)
            bool: True if the input is batch, False otherwise
        """
        # If single sample, make it a batch
        is_batch = True

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            is_batch = False

        return x, is_batch
    
    def interpret(
        self,
        network_outputs: Dict[str, torch.Tensor],
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpret network outputs into discrete trading actions.
        
        Args:
            network_outputs: Dictionary containing:
                - 'action_probs': Action probabilities from head
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of:
                - Tensor of action choices (-1: sell, 0: hold, 1: buy)
                - Tensor of scaled actions (-1, 0, 1)
        """
        # Get action probabilities
        probs = self._get_action_probs(network_outputs)
        probs, original_shape_is_batch = self._get_probs_batch_input(probs) # shape: (batch_size, n_assets, n_actions)
        batch_size, n_assets, _ = probs.shape
        
        # Deterministic action selection
        if deterministic:
            # Use argmax to get the action choice
            action_choices = torch.argmax(probs, dim=-1) - 1 # shape: (batch_size, n_assets)
        # Random action selection
        else: 
            # Sample from parametric head's learned probabilities
            categorical_dist = torch.distributions.Categorical(probs.view(-1, 3))
            action_choices = categorical_dist.sample().view(batch_size, n_assets) - 1 # shape: (batch_size, n_assets)
        
        # Squeeze batch dimension if present
        if not original_shape_is_batch:
            action_choices = action_choices.squeeze(0) # shape: (1, n_assets) -> (n_assets)
        
        return self.scale_actions(action_choices), action_choices.type(torch.int64)
    
    def get_q_values(
        self,
        network_outputs: Dict[str, torch.Tensor],
        action_choices: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract Q-values for specific actions from network outputs.
        
        Args:
            network_outputs: Dictionary containing action probabilities
            actions: Action choices (-1, 0, 1)
            
        Returns:
            Tensor of Q-values for the given actions
        """
        # Get action probabilities
        probs = self._get_action_probs(network_outputs)
        probs, probs_original_shape_is_batch = self._get_probs_batch_input(probs) # shape: (batch_size, n_assets, n_actions)

        if len(action_choices.shape) == 1:
            action_choices = action_choices.unsqueeze(0) # shape: (n_assets) -> (1, n_assets)

        if probs.shape[0] != action_choices.shape[0]:
            raise ValueError("Probs and action_choices must have the same batch size!")

        # Convert actions from [-1,0,1] to [0,1,2] for indexing
        action_indices = action_choices + 1 # shape: (batch_size, n_assets)
        
        q_values = torch.gather(probs, 2, action_indices.unsqueeze(-1)).squeeze(-1) # shape: (batch_size, n_assets)

        if not probs_original_shape_is_batch:
            q_values = q_values.squeeze(0) # shape: (1, n_assets) -> (n_assets)

        return q_values
    
    def get_max_q_values(
        self,
        network_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get maximum Q-values from network outputs.
        
        Args:
            network_outputs: Dictionary containing action probabilities
            
        Returns:
            Tensor of maximum Q-values
        """
        probs = self._get_action_probs(network_outputs)
        probs, original_shape_is_batch = self._get_probs_batch_input(probs) # shape: (batch_size, n_assets, n_actions)

        # Get maximum Q-values
        max_q_values = torch.max(probs, dim=-1).values # shape: (batch_size, n_assets)
        
        # Squeeze batch dimension if present
        if not original_shape_is_batch:
            max_q_values = max_q_values.squeeze(0) # shape: (1, n_assets) -> (n_assets)
        
        return max_q_values

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
        Compute loss for discrete actions.
        """
        if method == "mse":
            return self.compute_mse_loss(
                current_outputs=current_outputs,
                target_outputs=target_outputs,
                action_choices=action_choices,
                rewards=rewards,
                dones=dones,
                gamma=gamma
            )

    def compute_mse_loss(
        self,
        current_outputs: Dict[str, torch.Tensor],
        target_outputs: Dict[str, torch.Tensor],
        action_choices: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """
        Compute standard Q-learning loss for discrete actions.
        
        Args:
            current_outputs: Current network outputs
            target_outputs: Target network outputs
            actions: Action choices (-max_trade_size, 0, max_trade_size)
            rewards: Reward tensor
            dones: Done flag tensor
            gamma: Discount factor
            
        Returns:
            Loss tensor
        """
        # Get current Q-values for taken actions
        current_q_values = self.get_q_values(current_outputs, action_choices)
        
        # Get next Q-values from target network
        next_q_values = self.get_max_q_values(target_outputs)
        
        # Compute target Q-values
        # Case 1: Single sample
        if len(next_q_values.shape) == 1:
            target_q_values = rewards + (1 - dones) * gamma * next_q_values
        # Case 2: Batch of samples
        else:
            target_q_values = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * gamma * next_q_values
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        
        return loss

    def interpret_with_log_prob(
        self,
        network_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interpret network outputs and compute log probabilities.
        
        Args:
            network_outputs: Dictionary containing network outputs
            
        Returns:
            Tuple of:
                - Tensor of scaled actions (-max_trade_size, 0, max_trade_size)
                - Tensor of action choices (-1: sell, 0: hold, 1: buy)
                - Tensor of log probabilities of the actions
        """
        # Get action probabilities
        action_probs = self._get_action_probs(network_outputs)
        action_probs, action_probs_original_shape_is_batch = self._get_probs_batch_input(action_probs) # shape: (batch_size, n_assets, n_actions)
        batch_size, n_assets, n_actions = action_probs.shape
            
        # Sample actions for each asset
        categorical_dist = torch.distributions.Categorical(action_probs.view(-1, n_actions))
        action_indices = categorical_dist.sample().view(batch_size, n_assets) # shape: (batch_size, n_assets)
        
        # Compute log probabilities (vectorized)
        log_probs = categorical_dist.log_prob(action_indices.view(-1)).view(batch_size, n_assets) # shape: (batch_size, n_assets)
        
        # Convert action indices (0, 1, 2) to actual actions (-1, 0, 1)
        action_choices = (action_indices - 1) # shape: (batch_size, n_assets)
        
        if not action_probs_original_shape_is_batch:
            action_choices = action_choices.squeeze(0) # shape: (1, n_assets) -> (n_assets)
            log_probs = log_probs.squeeze(0) # shape: (1, n_assets) -> (n_assets)
        
        return self.scale_actions(action_choices), action_choices.type(torch.int64), log_probs

    def scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Scale actions from choices (-1, 0, 1) to actual trading decisions.
        
        Args:
            actions: Tensor of action choices (-1, 0, 1)
            
        Returns:
            Tensor of scaled integer actions (-max_trade_size, 0, max_trade_size)
        """
        # Round is just to ensure that the actions are integers!
        scaled_actions = actions * self.max_trade_size
        return scaled_actions.type(torch.int64)

    def evaluate_actions_log_probs(
        self,
        network_outputs: Dict[str, torch.Tensor],
        action_choices: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the log probabilities of given actions under the current policy.
        Args:
            network_outputs: Dictionary of network outputs from the policy network
            action_choices: Tensor of action choices to evaluate (-1,0,1), shape: (batch_size, n_assets)
        Returns:
            Tensor of log_probs
        """
        action_probs = self._get_action_probs(network_outputs)
        action_probs, action_probs_original_shape_is_batch = self._get_probs_batch_input(action_probs) # shape: (batch_size, n_assets, n_actions)
        batch_size, n_assets, n_actions = action_probs.shape

        if len(action_choices.shape) == 1:
            action_choices = action_choices.unsqueeze(0) # shape: (n_assets) -> (1, n_assets)

        # Convert actions from [-1,0,1] to [0,1,2] choices for gathering
        action_indices = action_choices + 1  # shape: (batch_size, n_assets)

        # Get log probabilities for each action
        categorical_dist = torch.distributions.Categorical(action_probs.view(-1, n_actions))
        log_probs = categorical_dist.log_prob(action_indices.view(-1)).view(batch_size, n_assets)
        
        # Squeeze batch dimension if present
        if not action_probs_original_shape_is_batch:
            log_probs = log_probs.squeeze(0) # shape: (1, n_assets) -> (n_assets)
        
        return log_probs 
    
    def get_config(self):
        return {
            "n_assets": self.n_assets,
            "max_trade_size": self.max_trade_size,
        }
