import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import random

from utils.logger import Logger
logger = Logger.get_logger()

from models.agents.base_agent import BaseAgent
from models.networks.base_network import BaseNetwork
from models.action_interpreters.allocation_interpreter import AllocationInterpreter
from models.processors.base_processor import ObservationFormat, BaseObservationProcessor
from environments.trading_env import TradingEnv
from models.networks.heads import DirectionHead, LinearHead

class PPONetwork(nn.Module):
    """Network for PPO agent that outputs policy distribution and value function."""
    ACTIVATIONS = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "softmax": nn.Softmax(dim=-1),
    }
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [1024, 512, 256, 128, 64, 32],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        # Initialize parent class
        super().__init__()

        # Initialize attributes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.device = device

        # Initialize activation function
        self.activation = self.ACTIVATIONS[activation]

        # Initialize shared layers (feature extractor)
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.shared_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if dropout > 0:
                self.shared_layers.append(nn.Dropout(dropout))
            self.shared_layers.append(self.ACTIVATIONS[activation])

        # Initialize policy heads (actor)
        self.direction_head = DirectionHead(hidden_dims[-1], output_dim)
        self.alpha_head = LinearHead(hidden_dims[-1], output_dim)
        self.beta_head = LinearHead(hidden_dims[-1], output_dim)
        
        # Initialize value head (critic)
        self.value_head = nn.Linear(hidden_dims[-1], 1)

        # Move to device
        self.to(device)

        # Log the network architecture
        logger.info(f"PPO Network architecture: {self}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Processes state through shared layers and heads.
        Args:
            state: Input state tensor (batch_size, state_dim)
        Returns:
            dict: Contains outputs from each head including value
        """
        # Apply shared layers sequentially
        for layer in self.shared_layers:
            x = layer(x)

        # Apply heads
        return {
            "direction": self.direction_head(x),
            "alpha": self.alpha_head(x),
            "beta": self.beta_head(x),
            "value": self.value_head(x)
        }
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the value prediction from the network."""
        for layer in self.shared_layers:
            x = layer(x)
        return self.value_head(x)

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent implementation.
    Uses clipped surrogate objective to stabilize training.
    """
    
    def __init__(
        self,
        env: TradingEnv,  # TradingEnv instance
        observation_processor_class: BaseObservationProcessor,  # Observation processor class
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the PPO agent.
        
        Args:
            env: Trading environment instance
            observation_processor_class: Class for processing observations
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            gae_lambda: Lambda for Generalized Advantage Estimation
            clip_ratio: Clip ratio for PPO update
            entropy_coef: Coefficient for entropy loss
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs for each update
            batch_size: Mini-batch size for PPO update
            device: Device to run the agent on
        """
        # Initialize observation processor
        self.processor: BaseObservationProcessor = observation_processor_class(env, device)
        
        # Initialize action interpreter
        self.action_interpreter = AllocationInterpreter(env)
        
        # Get state and action dimensions
        state_dim = self.processor.get_observation_dim(ObservationFormat.FLAT)
        action_dim = env.get_action_dim()
        
        super().__init__(state_dim, action_dim, device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Settings for random action selection
        self.position_limits = env.get_position_limits()
        self.random_bound = self.position_limits["max"] - self.position_limits["min"]

        # Initialize network
        self.network = PPONetwork(state_dim, action_dim, device=device)
        
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
    
    def get_intended_action(self, observation: Dict[str, np.ndarray], current_position: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get the intended action using the policy network.
        
        Args:
            observation: Current state observation
            current_position: Current position of the agent
            deterministic: Whether to select actions deterministically
            
        Returns:
            Selected action (number of shares to buy/sell for each asset)
        """
        # Process observation
        state = self.processor.process(observation, format=ObservationFormat.FLAT)
        
        with torch.no_grad():
            outputs = self.network.forward(state)
            
            # Store value for training
            self.last_value = outputs["value"]
            
            # Sample from policy or take most likely action
            if deterministic:
                action = self.action_interpreter.interpret(outputs, current_position, deterministic=True)
            else:
                action, log_probs = self.action_interpreter.interpret_with_log_prob(outputs, current_position)
                self.last_log_probs = log_probs
            
        return action
    
    def add_to_rollout(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool
    ) -> None:
        """
        Add a transition to the current rollout.
        
        Args:
            observation: Current state observation
            action: Action taken (number of shares to buy/sell)
            reward: Reward received
            next_observation: Next state observation
            done: Whether the episode ended
        """
        # Process observations
        state = self.processor.process(observation, format=ObservationFormat.FLAT)
        
        # Store transition data
        self.rollout_states.append(state)
        self.rollout_actions.append(torch.IntTensor(action))
        self.rollout_log_probs.append(self.last_log_probs)
        self.rollout_rewards.append(torch.FloatTensor([reward]))
        self.rollout_dones.append(torch.FloatTensor([done]))
        self.rollout_values.append(self.last_value)
    
    def update(self) -> Dict[str, float]:
        """
        Update the agent's network using collected rollout data with PPO.
        
        Returns:
            Dictionary of training metrics
        """
        # Check if we have data to train on
        if len(self.rollout_states) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0, "total_loss": 0}
        
        # Convert lists to tensors
        states = torch.stack(self.rollout_states)
        actions = torch.stack(self.rollout_actions)
        old_log_probs = torch.stack(self.rollout_log_probs)
        rewards = torch.stack(self.rollout_rewards)
        dones = torch.stack(self.rollout_dones)
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
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current outputs
                outputs = self.network.forward(batch_states)
                batch_values = outputs["value"]
                
                # Get current log probabilities
                _, batch_new_log_probs = self.action_interpreter.evaluate_actions_log_probs(
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

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return "PPO"

    def get_model(self) -> nn.Module:
        """Get the network."""
        return self.network

    def train(self, mode: bool = True) -> None:
        """Set the model to training mode."""
        self.network.train(mode)

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        self.train(mode=False)

    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)

    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps'] 