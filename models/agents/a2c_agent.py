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
from environments.trading_env import TradingEnv
from ..networks.unified_network import UnifiedNetwork

class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) agent implementation.
    Uses a unified network for policy and value prediction.
    """
    
    def __init__(
        self,
        env: TradingEnv,
        network_config: Dict[str, Any],
        interpreter: BaseActionInterpreter,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.0007,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        **kwargs
    ) -> None:
        """
        Initialize the A2C agent.
        
        Args:
            env: Trading environment instance
            network_config: Configuration for the unified network
            interpreter: Action interpreter instance
            device: Device to run the agent on
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            entropy_coef: Coefficient for entropy loss
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            **kwargs: Additional arguments
        """
        super().__init__(env, network_config, interpreter, device)
        
        # Hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
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
        
        # Track metrics for visualization
        self.last_loss = 0.0
    
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
            
            # Sample from policy or take most likely action
            if deterministic:
                scaled_action = self.interpreter.interpret(cpu_outputs, current_position, deterministic=True)
                action_choice = self.interpreter.get_action_choice(cpu_outputs, deterministic=True)
            else:
                scaled_action, log_probs = self.interpreter.interpret_with_log_prob(cpu_outputs, current_position)
                action_choice = self.interpreter.get_action_choice(cpu_outputs, deterministic=False)
                # Convert log_probs to a PyTorch tensor if it's not already one
                if not isinstance(log_probs, torch.Tensor):
                    self.last_log_probs = torch.tensor(log_probs, device=self.device)
                else:
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
        Update the agent's network using collected rollout data.
        
        Args:
            batch: Dictionary containing experience data (not used in A2C)
            
        Returns:
            Dictionary of training metrics
        """
        # Check if we have data to train on
        if len(self.rollout_states) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0, "total_loss": 0}
        
        # Convert lists to tensors
        states = {k: torch.stack([s[k] for s in self.rollout_states]) for k in self.rollout_states[0].keys()}
        actions = torch.stack(self.rollout_actions)
        log_probs = torch.stack(self.rollout_log_probs)
        rewards = torch.stack(self.rollout_rewards)
        dones = torch.stack(self.rollout_dones).float()  # Convert bool to float
        values = torch.stack(self.rollout_values)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        next_return = 0
        next_value = 0
        
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            next_return = r + self.gamma * next_return * (1 - d)
            next_advantage = next_return - v
            returns.insert(0, next_return)
            advantages.insert(0, next_advantage)
        
        returns = torch.cat(returns).detach()
        advantages = torch.cat(advantages).detach()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        # Compute losses
        outputs = self.network(states)
        value_loss = self.value_coef * ((returns - outputs["value"]) ** 2).mean()
        
        # Policy loss
        policy_loss = -(advantages * log_probs).mean()
        
        # Entropy loss (encourage exploration)
        entropy_loss = -self.entropy_coef * outputs.get("entropy", torch.zeros(1).to(self.device)).mean()
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Store loss values for visualization
        self.last_policy_loss = policy_loss.item()
        self.last_value_loss = value_loss.item()
        self.last_entropy_loss = entropy_loss.item()
        self.last_total_loss = total_loss.item()
        
        # Clear rollout buffers
        self.rollout_states = []
        self.rollout_actions = []
        self.rollout_log_probs = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.rollout_values = []
        
        # Increment steps
        self.steps += 1
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item()
        }

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
        checkpoint = torch.load(os.path.join(path, 'agent_state.pth'))
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
        """
        return {
            "policy_loss": getattr(self, 'last_policy_loss', 0),
            "value_loss": getattr(self, 'last_value_loss', 0),
            "entropy_loss": getattr(self, 'last_entropy_loss', 0),
            "total_loss": getattr(self, 'last_total_loss', 0)
        }

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the agent."""
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "device": self.device
        }
