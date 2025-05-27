import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
import random
import os

from utils.logger import Logger
logger = Logger.get_logger()

from models.agents.base_agent import BaseAgent
from models.action_interpreters.base_action_interpreter import BaseActionInterpreter
from models.agents.temperature_manager import TemperatureManager

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent implementation.
    Uses clipped surrogate objective to stabilize training.
    """
    
    def __init__(
        self,
        network_config: Dict[str, Any],
        interpreter: BaseActionInterpreter,
        temperature_manager: TemperatureManager,
        update_frequency: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        **kwargs
    ):
        """
        Initialize the PPO agent.
        
        Args:
            network_config: Configuration for the unified network
            interpreter: Action interpreter instance
            temperature_manager: Temperature manager instance
            update_frequency: Update frequency for the temperature manager
            device: Device to run the agent on
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            gae_lambda: Lambda for Generalized Advantage Estimation
            clip_ratio: Clip ratio for PPO update
            entropy_coef: Coefficient for entropy loss
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs for each update
            batch_size: Mini-batch size for PPO update
            **kwargs: Additional arguments
        """
        super().__init__(network_config, interpreter, temperature_manager, update_frequency, device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Storage for rollout data
        self.rollout_states = []
        self.rollout_actions = []
        self.rollout_log_probs = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.rollout_values = []
        
        # Steps counter
        self.steps = 0
    
    def get_intended_action(
        self,
        observations: Dict[str, torch.Tensor],
        current_position: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the intended action using the policy network.
        
        Args:
            observations: Dictionary of observation tensors
            current_position: Current position of the agent
            deterministic: Whether to select actions deterministically
            
        Returns:
            Tuple of (scaled_action, action_choice) where:
            - scaled_action is the actual action to execute
            - action_choice is the unscaled action for learning
        """
        with torch.no_grad():
            outputs = self.network(observations)
            
            # Store value for training
            self.last_value = outputs["value"]

            # Convert outputs to numpy and remove batch dimension if present
            cpu_outputs = {}
            for key, value in outputs.items():
                if key == "entropy":
                    continue  # Skip entropy which is a scalar
                if isinstance(value, torch.Tensor):
                    cpu_outputs[key] = value.cpu().numpy()
                else:
                    cpu_outputs[key] = value
            
            # Take deterministic action
            if deterministic:
                scaled_action, action_choice = self.interpreter.interpret(cpu_outputs, current_position, deterministic=True)
            else:
                scaled_action, action_choice, log_probs = self.interpreter.interpret_with_log_prob(cpu_outputs, current_position)
                
                # Store log probabilities for training
                self.last_log_probs = log_probs
            
        return scaled_action, action_choice
    
    def add_to_rollout(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        action_choice: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool
    ) -> None:
        """
        Add a transition to the current rollout.
        
        Args:
            observation: Current state observation
            action: Scaled action taken
            action_choice: Unscaled action choice for learning
            reward: Reward received
            next_observation: Next state observation
            done: Whether the episode ended
        """
        # Convert observations to tensors
        obs_tensors = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in observation.items()
        }
        
        # Store transition data
        self.rollout_states.append(obs_tensors)
        self.rollout_actions.append(torch.as_tensor(action_choice, device=self.device))
        self.rollout_log_probs.append(self.last_log_probs)
        self.rollout_rewards.append(torch.tensor([reward], device=self.device))
        self.rollout_dones.append(torch.tensor([done], device=self.device))
        self.rollout_values.append(self.last_value)
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's network using collected rollout data with PPO.
        
        Args:
            batch: Dictionary containing experience data (not used in PPO)
            
        Returns:
            Dictionary of training metrics
        """
        # Check if we have data to train on
        if len(self.rollout_states) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0, "total_loss": 0}
        
        # Convert lists to tensors
        states = {k: torch.stack([s[k] for s in self.rollout_states]) for k in self.rollout_states[0].keys()}
        actions = torch.stack(self.rollout_actions)
        old_log_probs = torch.stack(self.rollout_log_probs)
        rewards = torch.stack(self.rollout_rewards)
        dones = torch.stack(self.rollout_dones).float()  # Convert bool to float
        values = torch.stack(self.rollout_values)
        
        # Calculate returns and advantages using GAE
        returns = []
        advantages = []
        next_value = 0
        next_advantage = 0
        
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            delta = r + self.gamma * next_value * (1 - d) - v
            next_advantage = delta + self.gamma * self.gae_lambda * next_advantage * (1 - d)
            next_value = v
            
            returns.insert(0, next_advantage + v)
            advantages.insert(0, next_advantage)
        
        returns = torch.cat(returns).detach()
        advantages = torch.cat(advantages).detach()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        # PPO update
        metrics = {
            "policy_loss": 0,
            "value_loss": 0,
            "entropy_loss": 0,
            "total_loss": 0,
            "clip_fraction": 0
        }
        
        # Prepare dataset
        dataset_size = len(states[list(states.keys())[0]])
        indices = np.arange(dataset_size)
        
        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = {k: v[batch_indices] for k, v in states.items()}
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current outputs
                outputs = self.network(batch_states)
                batch_values = outputs["value"]
                
                # Get current log probabilities
                batch_new_log_probs = self.interpreter.evaluate_actions_log_probs(
                    outputs, batch_actions
                )
                
                # Compute ratio (policy / old policy)
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = self.value_coef * ((batch_returns - batch_values) ** 2).mean()
                
                # Entropy loss
                entropy_loss = -self.entropy_coef * outputs.get("entropy", torch.zeros(1).to(self.device)).mean()
                
                # Total loss
                total_loss = policy_loss + value_loss + entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["total_loss"] += total_loss.item()
                metrics["clip_fraction"] += ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
        
        # Average metrics
        num_updates = self.ppo_epochs * ((dataset_size + self.batch_size - 1) // self.batch_size)
        for key in metrics:
            metrics[key] /= num_updates
        
        # Clear rollout buffers
        self.rollout_states = []
        self.rollout_actions = []
        self.rollout_log_probs = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.rollout_values = []
        
        # Increment steps
        self.steps += 1
        
        return metrics

    def save(self, path: str) -> None:
        """
        Save the agent's networks and optimizer state.
        
        Args:
            path: Base path to save the models
        """
        os.makedirs(path, exist_ok=True)
        
        # Save network state
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps
        }, os.path.join(path, 'agent_state.pth'))

    def load(self, path: str) -> None:
        """
        Load the agent's networks and optimizer state.
        
        Args:
            path: Base path to load the models from
        """
        checkpoint = torch.load(os.path.join(path, 'agent_state.pth'), weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information for visualization.
        
        Returns:
            Dictionary containing agent information:
            - policy_loss: Last policy loss
            - value_loss: Last value loss
            - entropy_loss: Last entropy loss
            - total_loss: Last total loss
            - clip_fraction: Last clip fraction
        """
        return {
            "policy_loss": getattr(self, 'last_policy_loss', 0),
            "value_loss": getattr(self, 'last_value_loss', 0),
            "entropy_loss": getattr(self, 'last_entropy_loss', 0),
            "total_loss": getattr(self, 'last_total_loss', 0),
            "clip_fraction": getattr(self, 'last_clip_fraction', 0)
        }

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the agent."""
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "ppo_epochs": self.ppo_epochs,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "network_config": self.network_config
        }

    def _get_agent_specific_state(self) -> Dict[str, Any]:
        """
        Get PPO-specific state for checkpointing.
        
        Returns:
            Dictionary containing PPO-specific state
        """
        return {
            'last_value': self.last_value.detach().cpu() if hasattr(self, 'last_value') else None,
            'last_log_probs': self.last_log_probs.detach().cpu() if hasattr(self, 'last_log_probs') else None,
            'last_entropy': self.last_entropy.detach().cpu() if hasattr(self, 'last_entropy') else None,
            'last_policy_loss': getattr(self, 'last_policy_loss', 0.0),
            'last_value_loss': getattr(self, 'last_value_loss', 0.0),
            'last_entropy_loss': getattr(self, 'last_entropy_loss', 0.0),
            'last_total_loss': getattr(self, 'last_total_loss', 0.0),
            'clip_ratio': self.clip_ratio,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'gae_lambda': self.gae_lambda,
            'normalize_advantages': self.normalize_advantages,
            'clip_grad_norm': self.clip_grad_norm
        }
    
    def _load_agent_specific_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load PPO-specific state from checkpoint.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        if 'last_value' in state_dict and state_dict['last_value'] is not None:
            self.last_value = state_dict['last_value'].to(self.device)
        if 'last_log_probs' in state_dict and state_dict['last_log_probs'] is not None:
            self.last_log_probs = state_dict['last_log_probs'].to(self.device)
        if 'last_entropy' in state_dict and state_dict['last_entropy'] is not None:
            self.last_entropy = state_dict['last_entropy'].to(self.device)
            
        self.last_policy_loss = state_dict.get('last_policy_loss', 0.0)
        self.last_value_loss = state_dict.get('last_value_loss', 0.0)
        self.last_entropy_loss = state_dict.get('last_entropy_loss', 0.0)
        self.last_total_loss = state_dict.get('last_total_loss', 0.0)
        
        # Load hyperparameters
        self.clip_ratio = state_dict.get('clip_ratio', self.clip_ratio)
        self.value_coef = state_dict.get('value_coef', self.value_coef)
        self.entropy_coef = state_dict.get('entropy_coef', self.entropy_coef)
        self.gae_lambda = state_dict.get('gae_lambda', self.gae_lambda)
        self.normalize_advantages = state_dict.get('normalize_advantages', self.normalize_advantages)
        self.clip_grad_norm = state_dict.get('clip_grad_norm', self.clip_grad_norm) 