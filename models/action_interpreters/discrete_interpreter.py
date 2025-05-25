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
        max_position_size: float = 10,
        temperature: float = 1.0,
        temperature_decay: float = 0.995,
        min_temperature: float = 0.1,
        interpreter_type: str = "discrete"
    ):
        """
        Initialize the discrete action interpreter.
        
        Args:
            n_assets: Number of assets to trade
            max_position_size: Maximum position size for any asset
            temperature: Initial temperature for Bayesian sampling
            temperature_decay: Rate at which temperature decays
            min_temperature: Minimum temperature value
            interpreter_type: Type of interpreter to use
        """
        super().__init__(
            interpreter_type=interpreter_type,
            n_assets=n_assets,
            max_position_size=max_position_size,
            temperature=temperature,
            temperature_decay=temperature_decay,
            min_temperature=min_temperature
        )
    
    def interpret(
        self,
        network_outputs: Dict[str, Union[torch.Tensor, np.ndarray]],
        current_position: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Interpret network outputs into discrete trading actions.
        
        Args:
            network_outputs: Dictionary containing:
                - 'action_probs': Action probabilities from parametric head
                - 'alphas' and 'betas': Beta distribution parameters from Bayesian head
            current_position: Current position array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Array of trading actions scaled by max_position_size (-max_position_size: sell, 0: hold, max_position_size: buy)
        """
        # Handle random action generation
        if network_outputs is None:
            return np.random.choice([-self.max_position_size, 0, self.max_position_size], size=self.n_assets)
        
        # Convert inputs to numpy if needed
        if isinstance(network_outputs, dict):
            if 'action_probs' in network_outputs:
                probs = network_outputs['action_probs']
                if isinstance(probs, torch.Tensor):
                    probs = probs.cpu().numpy()
                # Take first batch if batched
                if len(probs.shape) == 3:
                    probs = probs[0]
            elif 'alphas' in network_outputs and 'betas' in network_outputs:
                alphas = network_outputs['alphas']
                betas = network_outputs['betas']
                if isinstance(alphas, torch.Tensor):
                    alphas = alphas.cpu().numpy()
                    betas = betas.cpu().numpy()
                # Take first batch if batched
                if len(alphas.shape) == 3:
                    alphas = alphas[0]
                    betas = betas[0]
                
                if deterministic:
                    # Use mean of Beta distribution
                    probs = alphas / (alphas + betas)
                else:
                    # Sample from Beta distribution with temperature
                    scaled_alphas = alphas / self.temperature
                    scaled_betas = betas / self.temperature
                    probs = np.random.beta(scaled_alphas, scaled_betas)
            else:
                raise ValueError("Network outputs must contain either 'action_probs' or 'alphas' and 'betas'")
        else:
            raise ValueError("Network outputs must be a dictionary")
        
        # Ensure probabilities sum to 1 for each asset
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        
        # Convert probabilities to actions
        if deterministic:
            # Take argmax for deterministic actions
            actions = np.argmax(probs, axis=-1) - 1  # Convert [0,1,2] to [-1,0,1]
        else:
            # Sample actions based on probabilities
            actions = np.array([
                np.random.choice([-1, 0, 1], p=probs[i])
                for i in range(self.n_assets)
            ])
        
        # Scale actions by max_position_size
        return actions * self.max_position_size
    
    def get_q_values(
        self,
        network_outputs: Dict[str, Union[torch.Tensor, np.ndarray]],
        actions: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Extract Q-values for specific actions from network outputs.
        
        Args:
            network_outputs: Dictionary containing action probabilities
            actions: Action indices (-max_position_size, 0, max_position_size)
            
        Returns:
            Tensor of Q-values for the given actions
        """
        if isinstance(network_outputs, dict):
            if 'action_probs' in network_outputs:
                probs = network_outputs['action_probs']
                if isinstance(probs, np.ndarray):
                    probs = torch.from_numpy(probs)
            elif 'alphas' in network_outputs and 'betas' in network_outputs:
                alphas = network_outputs['alphas']
                betas = network_outputs['betas']
                if isinstance(alphas, np.ndarray):
                    alphas = torch.from_numpy(alphas)
                    betas = torch.from_numpy(betas)
                probs = alphas / (alphas + betas)
            else:
                raise ValueError("Network outputs must contain either 'action_probs' or 'alphas' and 'betas'")
        else:
            raise ValueError("Network outputs must be a dictionary")
        
        # Convert actions from [-max_position_size,0,max_position_size] to [0,1,2] for indexing
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        action_indices = (actions / self.max_position_size).long() + 1
        
        # Get Q-values for the given actions
        batch_size = probs.shape[0] if len(probs.shape) > 2 else 1
        if batch_size == 1:
            # Handle single batch case
            return torch.tensor([probs[0, i, idx.item()] for i, idx in enumerate(action_indices)])
        else:
            # Handle multiple batches case
            # Reshape action_indices to match probs dimensions
            action_indices = action_indices.view(-1, self.n_assets)
            # Use gather to get Q-values for each action
            return torch.gather(probs, 2, action_indices.unsqueeze(-1)).squeeze(-1)
    
    def get_max_q_values(
        self,
        network_outputs: Dict[str, Union[torch.Tensor, np.ndarray]]
    ) -> torch.Tensor:
        """
        Get maximum Q-values from network outputs.
        
        Args:
            network_outputs: Dictionary containing action probabilities
            
        Returns:
            Tensor of maximum Q-values
        """
        if isinstance(network_outputs, dict):
            if 'action_probs' in network_outputs:
                probs = network_outputs['action_probs']
                if isinstance(probs, np.ndarray):
                    probs = torch.from_numpy(probs)
            elif 'alphas' in network_outputs and 'betas' in network_outputs:
                alphas = network_outputs['alphas']
                betas = network_outputs['betas']
                if isinstance(alphas, np.ndarray):
                    alphas = torch.from_numpy(alphas)
                    betas = torch.from_numpy(betas)
                probs = alphas / (alphas + betas)
            else:
                raise ValueError("Network outputs must contain either 'action_probs' or 'alphas' and 'betas'")
        else:
            raise ValueError("Network outputs must be a dictionary")
        
        # Get maximum Q-values
        max_q_values = torch.max(probs, dim=-1)[0]
        
        # Squeeze batch dimension if present
        if len(max_q_values.shape) > 1:
            max_q_values = max_q_values.squeeze(0)
        
        return max_q_values

    def get_action_choice(
        self,
        network_outputs: Dict[str, Union[torch.Tensor, np.ndarray]],
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get the unscaled action choice (-1, 0, 1) from network outputs.
        This is used for Q-learning, while interpret() is used for execution.
        
        Args:
            network_outputs: Dictionary containing network outputs
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Array of unscaled action choices (-1, 0, 1)
        """
        # Handle random action generation
        if network_outputs is None:
            return np.random.choice([-1, 0, 1], size=self.n_assets)
        
        # Convert inputs to numpy if needed
        if isinstance(network_outputs, dict):
            if 'action_probs' in network_outputs:
                probs = network_outputs['action_probs']
                if isinstance(probs, torch.Tensor):
                    probs = probs.cpu().numpy()
                # Take first batch if batched
                if len(probs.shape) == 3:
                    probs = probs[0]
            elif 'alphas' in network_outputs and 'betas' in network_outputs:
                alphas = network_outputs['alphas']
                betas = network_outputs['betas']
                if isinstance(alphas, torch.Tensor):
                    alphas = alphas.cpu().numpy()
                    betas = betas.cpu().numpy()
                # Take first batch if batched
                if len(alphas.shape) == 3:
                    alphas = alphas[0]
                    betas = betas[0]
                
                if deterministic:
                    # Use mean of Beta distribution
                    probs = alphas / (alphas + betas)
                else:
                    # Sample from Beta distribution with temperature
                    scaled_alphas = alphas / self.temperature
                    scaled_betas = betas / self.temperature
                    probs = np.random.beta(scaled_alphas, scaled_betas)
            else:
                raise ValueError("Network outputs must contain either 'action_probs' or 'alphas' and 'betas'")
        else:
            raise ValueError("Network outputs must be a dictionary")
        
        # Ensure probabilities sum to 1 for each asset
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        
        # Convert probabilities to base actions
        if deterministic:
            # Take argmax for deterministic actions
            base_actions = np.argmax(probs, axis=-1) - 1  # Convert [0,1,2] to [-1,0,1]
        else:
            # Sample actions based on probabilities
            base_actions = np.array([
                np.random.choice([-1, 0, 1], p=probs[i])
                for i in range(self.n_assets)
            ])
        
        return base_actions

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
        Compute standard Q-learning loss for discrete actions.
        
        Args:
            current_outputs: Current network outputs
            target_outputs: Target network outputs
            actions: Action indices (-max_position_size, 0, max_position_size)
            rewards: Reward tensor
            dones: Done flag tensor
            gamma: Discount factor
            
        Returns:
            Loss tensor
        """
        # Get current Q-values for taken actions
        current_q_values = self.get_q_values(current_outputs, actions)
        
        # Get next Q-values from target network
        next_q_values = self.get_max_q_values(target_outputs)
        
        # Ensure next_q_values has the right shape for broadcasting
        if len(next_q_values.shape) == 1:
            next_q_values = next_q_values.unsqueeze(0)  # Add batch dimension
        
        # Compute target Q-values
        target_q_values = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * gamma * next_q_values
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        
        return loss

    def interpret_with_log_prob(self, network_outputs, current_position):
        """Interpret network outputs and compute log probabilities.
        
        Args:
            network_outputs: Dictionary containing network outputs
            current_position: Current position array
            
        Returns:
            Tuple of (scaled_actions, log_probs)
        """
        # Handle both parametric and Bayesian outputs
        if 'action_probs' in network_outputs:
            action_probs = network_outputs['action_probs'][0]  # Shape: (n_assets, n_actions)
        elif 'alphas' in network_outputs and 'betas' in network_outputs:
            alphas = network_outputs['alphas'][0]  # Shape: (n_assets, n_actions)
            betas = network_outputs['betas'][0]    # Shape: (n_assets, n_actions)
            action_probs = alphas / (alphas + betas)  # Convert Beta parameters to probabilities
        else:
            raise ValueError("Network outputs must contain either 'action_probs' or 'alphas'/'betas'")
            
        # Ensure probabilities sum to 1 for each asset
        action_probs = action_probs / np.sum(action_probs, axis=-1, keepdims=True)
            
        # Sample actions for each asset
        actions = []
        log_probs = []
        for i in range(len(current_position)):
            probs = action_probs[i]  # Get probabilities for this asset
            action_idx = np.random.choice(3, p=probs)  # Sample action index
            action = (action_idx - 1) * self.max_position_size  # Scale by max position size
            log_prob = np.log(probs[action_idx])  # Get log probability of chosen action
            
            actions.append(action)
            log_probs.append(log_prob)
            
        return np.array(actions), np.array(log_probs)

    def scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Scale actions from indices (-1, 0, 1) to actual trading decisions.
        
        Args:
            actions: Tensor of action indices (-1, 0, 1)
            
        Returns:
            Tensor of scaled actions (-max_position_size, 0, max_position_size)
        """
        # Convert actions to float if they're not already
        actions = actions.float()
        
        # Scale actions by max_position_size
        scaled_actions = actions * self.max_position_size
        
        return scaled_actions

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
        if 'action_probs' in network_outputs:
            action_probs = network_outputs['action_probs']
        elif 'alphas' in network_outputs and 'betas' in network_outputs:
            alphas = network_outputs['alphas']
            betas = network_outputs['betas']
            action_probs = alphas / (alphas + betas)
        else:
            raise ValueError("Network outputs must contain either 'action_probs' or 'alphas'/'betas'")
        
        # Store original actions shape to handle single vs batch cases
        original_actions_shape = actions.shape
        
        # Ensure actions are 2D (batch, n_assets)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        # Repeat network outputs if batch size is 1 but actions batch size is >1
        if action_probs.shape[0] == 1 and actions.shape[0] > 1:
            action_probs = action_probs.repeat(actions.shape[0], 1, 1)
        
        # Get log probabilities for each action
        log_probs = torch.log(action_probs + 1e-10)
        
        # Ensure actions and log_probs have matching dimensions
        if log_probs.dim() == 3:  # (batch, n_assets, n_actions)
            actions_for_gather = actions.unsqueeze(-1).long()  # (batch, n_assets, 1)
            action_log_probs = torch.gather(log_probs, -1, actions_for_gather).squeeze(-1)
        else:
            raise ValueError(f"Unexpected log_probs dimensions: {log_probs.shape}")
        
        # Convert actions from [0,1,2] to [-1,0,1] and scale by max_position_size
        actual_actions = torch.zeros_like(actions.float())
        for i in range(actions.shape[0]):
            for j in range(actions.shape[1]):
                if actions[i, j] == 0:  # sell
                    actual_actions[i, j] = -1.0
                elif actions[i, j] == 1:  # hold
                    actual_actions[i, j] = 0.0
                elif actions[i, j] == 2:  # buy
                    actual_actions[i, j] = 1.0
        
        # Scale actions by max_position_size
        scaled_actions = actual_actions * self.max_position_size
        
        # Handle single action case
        if len(original_actions_shape) == 1:
            scaled_actions = scaled_actions.squeeze(0)
            action_log_probs = action_log_probs.squeeze(0)
        
        return scaled_actions, action_log_probs 
    
    def get_config(self):
        return {
            "n_assets": self.n_assets,
            "max_position_size": self.max_position_size,
            "temperature": self.temperature,
            "temperature_decay": self.temperature_decay,
            "min_temperature": self.min_temperature
        }
