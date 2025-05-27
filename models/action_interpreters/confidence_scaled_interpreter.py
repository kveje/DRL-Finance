"""Confidence scaled action interpreter implementation."""
import numpy as np
import torch
from typing import Dict, Union, Optional, Tuple
from .base_action_interpreter import BaseActionInterpreter

class ConfidenceScaledInterpreter(BaseActionInterpreter):
    """Interpreter that scales actions based on confidence values."""
    
    def __init__(
        self,
        n_assets: int,
        max_trade_size: float = 10,
        interpreter_type: str = "confidence_scaled"
    ):
        """
        Initialize the confidence scaled interpreter.
        
        Args:
            n_assets: Number of assets to trade
            max_trade_size: Maximum trade size for any asset
            interpreter_type: Type of interpreter to use
        """
        super().__init__(
            n_assets=n_assets,
            max_trade_size=max_trade_size,
            interpreter_type=interpreter_type
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
    
    def _get_confidences(
        self,
        network_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get confidences from dictionary.
        """
        return network_outputs['confidences']
    
    def _get_confidences_batch_input(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        If the input is a single sample, make it a batch.
        """
        is_batch = True

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            is_batch = False

        return x, is_batch

    def interpret(
        self,
        network_outputs: Dict[str, torch.Tensor],
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpret network outputs into confidence-scaled trading actions.
        
        Args:
            network_outputs: Dictionary containing network outputs
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (scaled_actions, action_choices)
        """
        probs = self._get_action_probs(network_outputs)
        probs, probs_is_batch = self._get_probs_batch_input(probs) # (batch_size, n_assets, n_actions)
        batch_size, n_assets, _ = probs.shape

        confidences = self._get_confidences(network_outputs)
        confidences, confidences_is_batch = self._get_confidences_batch_input(confidences) # (batch_size, n_assets)

        if confidences.shape[0] != probs.shape[0]:
            raise ValueError("Probs and confidences must have the same batch size")

        if deterministic:
            action_indices = torch.argmax(probs, dim=-1) # (batch_size, n_assets)
        else:
            # Sample from parametric head's learned probabilities
            categorical_dist = torch.distributions.Categorical(probs.view(-1, 3))
            action_indices = categorical_dist.sample().view(batch_size, n_assets) # (batch_size, n_assets)

        action_choices = action_indices - 1 # (batch_size, n_assets)
        scaled_actions = self.scale_actions_with_confidence(action_choices, confidences) # (batch_size, n_assets)

        if not probs_is_batch:
            action_choices = action_choices.squeeze(0) # (n_assets)
            scaled_actions = scaled_actions.squeeze(0) # (n_assets)

        # Convert to integers and return
        return scaled_actions, action_choices.type(torch.int64)
    
    def get_q_values(
        self,
        network_outputs: Dict[str, torch.Tensor],
        action_choices: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract Q-values for specific actions from network outputs.
        
        Args:
            network_outputs: Dictionary containing network outputs
            actions: Action indices (-1, 0, 1)
            
        Returns:
            Tensor of Q-values for the given actions
        """
        probs = self._get_action_probs(network_outputs)
        probs, probs_is_batch = self._get_probs_batch_input(probs) # (batch_size, n_assets, n_actions)

        confidences = self._get_confidences(network_outputs)
        confidences, confidences_is_batch = self._get_confidences_batch_input(confidences) # (batch_size, n_assets)

        if probs_is_batch != confidences_is_batch:
            raise ValueError("If one of probs or confidences is a batch, the other must also be a batch")
        
        if confidences.shape[0] != probs.shape[0]:
            raise ValueError("Probs and confidences must have the same batch size")
        
        if len(action_choices.shape) == 1:
            action_choices = action_choices.unsqueeze(0) # (batch_size, n_assets)

        if probs.shape[0] != action_choices.shape[0]:
            raise ValueError("Probs and action_choices must have the same batch size")
        
        # Convert actions from [-1,0,1] to [0,1,2] for indexing
        action_indices = action_choices + 1 # (batch_size, n_assets)

        # Get Q-values for the given actions
        q_values = torch.gather(probs, 2, action_indices.unsqueeze(-1)).squeeze(-1) * confidences # (batch_size, n_assets)

        if not probs_is_batch:
            q_values = q_values.squeeze(0)

        return q_values

    
    def get_max_q_values(
        self,
        network_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get maximum Q-values from network outputs.
        
        Args:
            network_outputs: Dictionary containing action probabilities and confidences
            
        Returns:
            Tensor of maximum Q-values scaled by confidence
        """
        # Get action probabilities
        probs = self._get_action_probs(network_outputs)
        probs, probs_is_batch = self._get_probs_batch_input(probs) # (batch_size, n_assets, n_actions)

        # Get confidences
        confidences = self._get_confidences(network_outputs)
        confidences, confidences_is_batch = self._get_confidences_batch_input(confidences) # (batch_size, n_assets)
        
        if probs_is_batch != confidences_is_batch:
            raise ValueError("If one of probs or confidences is a batch, the other must also be a batch")
        
        if confidences.shape[0] != probs.shape[0]:
            raise ValueError("Probs and confidences must have the same batch size")

        # Get maximum Q-values
        max_q_values = torch.max(probs, dim=-1).values # (batch_size, n_assets)

        # Scale by confidence
        max_q_values = max_q_values * confidences # (batch_size, n_assets)
        
        if not probs_is_batch:
            max_q_values = max_q_values.squeeze(0) # (n_assets)

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
        Compute loss for confidence-scaled actions.
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
        else:
            raise ValueError(f"Method {method} not supported")
    

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
        Compute combined loss for confidence-scaled actions.
        Includes both Q-learning loss and confidence regularization.
        
        Args:
            current_outputs: Current network outputs
            target_outputs: Target network outputs
            action_choices: Action choices (-1, 0, 1)
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


        # Compute Q-learning loss
        q_loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        
        return q_loss

    def interpret_with_log_prob(
        self,
        network_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpret network outputs and compute log probabilities.
        
        Args:
            network_outputs: Dictionary containing network outputs
        Returns:
            Tuple of (scaled_actions, action_choices, log_probs)
        """
        action_probs = self._get_action_probs(network_outputs)
        action_probs, action_probs_is_batch = self._get_probs_batch_input(action_probs) # (batch_size, n_assets, n_actions)

        confidences = self._get_confidences(network_outputs)
        confidences, confidences_is_batch = self._get_confidences_batch_input(confidences) # (batch_size, n_assets)

        batch_size, n_assets, n_actions = action_probs.shape

        if action_probs_is_batch != confidences_is_batch:
            raise ValueError("If one of probs or confidences is a batch, the other must also be a batch")
        
        if confidences.shape[0] != action_probs.shape[0]:
            raise ValueError("Probs and confidences must have the same batch size")
            
        # Sample actions for each asset
        categorical_dist = torch.distributions.Categorical(action_probs.view(-1, n_actions))
        action_indices = categorical_dist.sample().view(batch_size, n_assets) # (batch_size, n_assets)
        
        # Compute log probabilities (vectorized)
        log_probs = categorical_dist.log_prob(action_indices.view(-1)).view(batch_size, n_assets) # (batch_size, n_assets)
        
        # Convert action indices (0, 1, 2) to actual actions (-1, 0, 1)
        action_choices = (action_indices - 1) # (batch_size, n_assets)

        scaled_actions = self.scale_actions_with_confidence(action_choices, confidences) # (batch_size, n_assets)
        
        if not action_probs_is_batch:
            scaled_actions = scaled_actions.squeeze(0)
            action_choices = action_choices.squeeze(0)
            log_probs = log_probs.squeeze(0)

        return scaled_actions, action_choices.type(torch.int64), log_probs

    def scale_actions_with_confidence(
        self,
        action_choices: torch.Tensor,
        confidences: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale actions using both max_trade_size and confidence values.
        Args:
            action_choices: Tensor of action indices (-1, 0, 1)
            confidences: Tensor of confidence values (0 to 1)
        Returns:
            Tensor of scaled integer actions (-max_trade_size * confidence to max_trade_size * confidence)
        """
        scaled_actions = action_choices * self.max_trade_size * confidences
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
            action_choices: Tensor of action indices (-1, 0, 1)
        Returns:
            Tensor of log probabilities of the actions
        """
        probs = self._get_action_probs(network_outputs)
        probs, probs_is_batch = self._get_probs_batch_input(probs) # (batch_size, n_assets, n_actions)
        batch_size, n_assets, n_actions = probs.shape

        confidences = self._get_confidences(network_outputs)
        confidences, confidences_is_batch = self._get_confidences_batch_input(confidences) # (batch_size, n_assets)
        
        if probs_is_batch != confidences_is_batch:
            raise ValueError("If one of probs or confidences is a batch, the other must also be a batch")
        
        if confidences.shape[0] != probs.shape[0]:
            raise ValueError("Probs and confidences must have the same batch size")
        
        # Convert actions from [-1,0,1] to [0,1,2] for indexing
        action_indices = action_choices + 1 # (batch_size, n_assets)

        categorical_dist = torch.distributions.Categorical(probs.view(-1, n_actions))
        log_probs = categorical_dist.log_prob(action_indices.view(-1)).view(batch_size, n_assets) # (batch_size, n_assets)

        if not probs_is_batch:
            log_probs = log_probs.squeeze(0) # (n_assets)

        return log_probs
    
    def get_config(self):
        return {
            "n_assets": self.n_assets,
            "max_trade_size": self.max_trade_size,
        }
