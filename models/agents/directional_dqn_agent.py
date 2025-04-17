import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
import random

from utils.logger import Logger
logger = Logger.get_logger()

from models.agents.base_agent import BaseAgent
from models.action_interpreters.directional_interpreter import DirectionalActionInterpreter
from models.processors.base_processor import ObservationFormat, BaseObservationProcessor
from environments.trading_env import TradingEnv
from models.networks.heads import DirectionHead

class DirectionalDQNNetwork(nn.Module):
    """Network for DirectionalDQN agent that outputs action probabilities for each asset."""
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
        output_dim: int,  # Number of assets
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

        # Initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            self.layers.append(self.ACTIVATIONS[activation])

        # Initialize direction head (sell, hold, buy for each asset)
        self.direction_head = DirectionHead(hidden_dims[-1], output_dim)

        # Move to device
        self.to(device)

        # Log the network architecture
        logger.info(f"DirectionalDQN Network architecture: {self}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Processes state through network and returns direction logits.
        Args:
            state: Input state tensor (batch_size, state_dim)
        Returns:
            Dictionary with direction logits for each asset
        """
        # Apply layers sequentially
        for layer in self.layers:
            x = layer(x)

        # Apply direction head
        direction_logits = self.direction_head(x)
        
        return {"direction": direction_logits}

class DirectionalDQNAgent(BaseAgent):
    """
    Directional Deep Q-Network (DQN) agent implementation.
    Uses a direction head for 3-way classification (sell/hold/buy) and scales actions by confidence.
    """
    
    def __init__(
        self,
        env: TradingEnv,
        observation_processor_class: BaseObservationProcessor, 
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update: int = 10,
        memory_size: int = 10000,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the DirectionalDQN agent.
        
        Args:
            env: Trading environment instance
            observation_processor_class: Class for processing observations
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Starting value for epsilon-greedy exploration
            epsilon_end: Final value for epsilon-greedy exploration
            epsilon_decay: Decay rate for epsilon
            target_update: Number of steps between target network updates
            memory_size: Size of the replay memory
            batch_size: Size of batches for training
            device: Device to run the agent on
        """
        # Initialize observation processor
        self.processor: BaseObservationProcessor = observation_processor_class(env, device)
        
        # Initialize action interpreter
        self.action_interpreter = DirectionalActionInterpreter(env)
        
        # Get state and action dimensions
        state_dim = self.processor.get_observation_dim(ObservationFormat.FLAT)
        action_dim = env.get_action_dim()  # Number of assets
        
        super().__init__(state_dim, action_dim, device)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.num_assets = action_dim

        # Save config
        self.config = {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "target_update": target_update,
            "memory_size": memory_size,
            "batch_size": batch_size,
            "device": device,
        }

        # Settings for the environment
        self.position_limits = env.get_position_limits()

        # Initialize networks
        self.policy_net = DirectionalDQNNetwork(state_dim, action_dim, device=device)
        self.target_net = DirectionalDQNNetwork(state_dim, action_dim, device=device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Training step counter
        self.steps = 0
        
        # Initialize loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Track metrics for visualization
        self.last_loss = 0.0
    
    def get_intended_action(self, observation: Dict[str, np.ndarray], current_position: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get the intended action using epsilon-greedy policy.
        
        Args:
            observation: Current state observation
            current_position: Current position of the agent
            deterministic: Whether to select actions deterministically
            
        Returns:
            Selected action (number of shares to buy/sell for each asset)
        """
        # Process observation
        state = self.processor.process(observation, format=ObservationFormat.FLAT)
        
        # Decide between exploration and exploitation
        if deterministic or random.random() > self.epsilon:
            # Deterministic action using policy net
            with torch.no_grad():
                model_output = self.policy_net(state)
                q_values = model_output["direction"].cpu().numpy()
                actions = self.action_interpreter.interpret(q_values, current_position, deterministic=True)
        else:
            # Random action - generate random logits
            random_logits = np.random.randn(self.num_assets, 3)
            actions = self.action_interpreter.interpret(random_logits, current_position, deterministic=False)
        
        return actions
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's networks using a batch of experience.
        
        Uses CrossEntropyLoss to train the direction head for classification.
        
        Args:
            batch: Dictionary containing experience data (states, actions, rewards, next_states, dones)
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        batch_size = states.shape[0]
        
        # Get current direction logits
        current_outputs = self.policy_net(states)
        current_direction_logits = current_outputs["direction"]  # [batch_size, n_assets, 3]
        
        # Get next direction logits using target network
        with torch.no_grad():
            next_outputs = self.target_net(next_states)
            next_direction_logits = next_outputs["direction"]  # [batch_size, n_assets, 3]
            
            # Find the max Q-value from each asset's direction logits
            next_q_values = torch.max(next_direction_logits, dim=2)[0]  # [batch_size, n_assets]
            
            # Calculate target Q-values
            # This is the standard DQN update: reward + gamma * max_a Q(s', a)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Convert actions to direction labels for CrossEntropyLoss
        # We need to determine if each action was a sell, hold, or buy
        direction_labels = torch.zeros(batch_size, self.num_assets, dtype=torch.long, device=self.device)
        
        # Actions < 0 are sell (0), actions == 0 are hold (1), actions > 0 are buy (2)
        direction_labels[actions < 0] = 0  # Sell
        direction_labels[actions == 0] = 1  # Hold
        direction_labels[actions > 0] = 2  # Buy
        
        # Reshape for CrossEntropyLoss
        current_direction_logits_view = current_direction_logits.view(-1, 3)  # [batch_size * n_assets, 3]
        direction_labels_view = direction_labels.view(-1)  # [batch_size * n_assets]
        
        # Compute classification loss
        loss = self.loss_fn(current_direction_logits_view, direction_labels_view)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store loss for visualization
        self.last_loss = loss.item()
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon
        }
    
    def add_to_memory(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool
    ) -> None:
        """
        Add a transition to the replay memory.
        
        Args:
            observation: Current state observation
            action: Action taken (number of shares to buy/sell)
            reward: Reward received
            next_observation: Next state observation
            done: Whether the episode ended
        """
        # Process observations
        state = self.processor.process(observation, format=ObservationFormat.FLAT)
        next_state = self.processor.process(next_observation, format=ObservationFormat.FLAT)
        
        self.memory.append((
            state,
            torch.FloatTensor(action),
            torch.FloatTensor([reward]),
            next_state,
            torch.FloatTensor([done])
        ))
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        Sample a batch from the replay memory.
        
        Returns:
            Dictionary containing the batch of experience
        """
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return {
            "states": torch.stack(states),
            "actions": torch.stack(actions),
            "rewards": torch.stack(rewards),
            "next_states": torch.stack(next_states),
            "dones": torch.stack(dones)
        }
    
    def get_info(self) -> Dict:
        """
        Get additional information for visualization and logging.
        
        Returns:
            Dictionary with agent information
        """
        return {
            "loss": self.last_loss,
            "epsilon": self.epsilon,
        }

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return "DirectionalDQN"

    def get_model(self) -> nn.Module:
        """Get the policy network."""
        return self.policy_net

    def train(self, mode: bool = True) -> None:
        """Set the model to training mode."""
        self.policy_net.train(mode)
        self.target_net.train(mode)

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
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state dictionary directly into the model.
        Used for backtesting when models are passed directly without saving to disk.
        
        Args:
            state_dict: State dictionary containing model weights
        """
        # If we're given the full model state dict (expected format from get_model().state_dict())
        if isinstance(state_dict, dict):
            self.policy_net.load_state_dict(state_dict)
            # Also update target network for consistency
            self.target_net.load_state_dict(state_dict)
            logger.info("Loaded model state dictionary directly into agent")
        else:
            logger.error(f"Invalid state dict format: {type(state_dict)}")
            raise ValueError("Invalid state dictionary format") 
        
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the agent."""
        return self.config

