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
from environments.trading_env import TradingEnv

class ActorNetwork(nn.Module):
    """Actor network for DDPG that outputs deterministic actions."""
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
        hidden_dims: List[int] = [256, 128, 64],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.activation = self.ACTIVATIONS[activation]
        
        # Initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            self.layers.append(self.ACTIVATIONS[activation])
        
        # Output layers for direction and magnitude
        self.direction_head = DirectionHead(hidden_dims[-1], output_dim)
        self.alpha_head = LinearHead(hidden_dims[-1], output_dim)
        self.beta_head = LinearHead(hidden_dims[-1], output_dim)
        
        # Move to device
        self.to(device)
        
        # Log the network architecture
        logger.info(f"DDPG Actor Network architecture: {self}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the actor network."""
        # Apply shared layers
        for layer in self.layers:
            x = layer(x)
        
        # Get outputs from each head
        direction_output = self.direction_head(x)
        alpha_output = self.alpha_head(x)
        beta_output = self.beta_head(x)
        
        return {
            "direction": direction_output,
            "alpha": alpha_output,
            "beta": beta_output
        }

class CriticNetwork(nn.Module):
    """Critic network for DDPG that estimates Q-values."""
    ACTIVATIONS = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "softmax": nn.Softmax(dim=-1),
    }
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.activation = self.ACTIVATIONS[activation]
        
        # Initialize layers
        self.layers = nn.ModuleList()
        
        # First layer combines state and action
        self.layers.append(nn.Linear(state_dim + action_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            self.layers.append(self.ACTIVATIONS[activation])
        
        # Output layer for Q-value
        self.q_head = nn.Linear(hidden_dims[-1], 1)
        
        # Move to device
        self.to(device)
        
        # Log the network architecture
        logger.info(f"DDPG Critic Network architecture: {self}")
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network."""
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Get Q-value
        q_value = self.q_head(x)
        
        return q_value

class DDPGAgent(BaseAgent):
    """
    Deep Deterministic Policy Gradient (DDPG) agent implementation.
    Uses separate actor and critic networks with target networks for stability.
    """
    
    def __init__(
        self,
        env: TradingEnv,
        observation_processor_class,
        learning_rate_actor: float = 0.0001,
        learning_rate_critic: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.001,  # Target network update rate
        memory_size: int = 100000,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """
        Initialize the DDPG agent.
        
        Args:
            env: Trading environment instance
            observation_processor_class: Class for processing observations
            learning_rate_actor: Learning rate for the actor network
            learning_rate_critic: Learning rate for the critic network
            gamma: Discount factor
            tau: Target network update rate
            memory_size: Size of the replay buffer
            batch_size: Batch size for training
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
        self.tau = tau
        self.batch_size = batch_size
        
        # Store configuration
        self.config = {
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "gamma": gamma,
            "tau": tau,
            "memory_size": memory_size,
            "batch_size": batch_size,
            "device": device
        }
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, device=device)
        self.critic = CriticNetwork(state_dim, action_dim, device=device)
        
        # Initialize target networks
        self.actor_target = ActorNetwork(state_dim, action_dim, device=device)
        self.critic_target = CriticNetwork(state_dim, action_dim, device=device)
        
        # Copy weights from main networks to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        
        # Initialize replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Steps counter
        self.steps = 0
        
        # Track metrics for visualization
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
    
    def get_intended_action(self, observation: Dict[str, np.ndarray], current_position: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get the intended action using the actor network.
        
        Args:
            observation: Current state observation
            current_position: Current position of the agent
            deterministic: Whether to select actions deterministically (always True for DDPG)
            
        Returns:
            Selected action (number of shares to buy/sell for each asset)
        """
        # Process observation
        state = self.processor.process(observation, format=ObservationFormat.FLAT)
        
        with torch.no_grad():
            # Get action from actor network
            outputs = self.actor.forward(state)
            
            # Convert outputs to numpy and remove batch dimension if present
            cpu_outputs = {}
            for key, value in outputs.items():
                if value.dim() > 1:
                    cpu_outputs[key] = value[0].cpu().numpy()
                else:
                    cpu_outputs[key] = value.cpu().numpy()
            
            # Get deterministic action
            action = self.action_interpreter.interpret(cpu_outputs, current_position, deterministic=True)
            
        return action
    
    def add_to_memory(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool
    ) -> None:
        """
        Add a transition to the replay buffer.
        
        Args:
            observation: Current state observation
            action: Action taken
            reward: Reward received
            next_observation: Next state observation
            done: Whether the episode ended
        """
        # Process observations
        state = self.processor.process(observation, format=ObservationFormat.FLAT)
        next_state = self.processor.process(next_observation, format=ObservationFormat.FLAT)
        
        # Convert to tensors
        state = state.cpu().numpy()
        next_state = next_state.cpu().numpy()
        
        # Store transition
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self) -> Dict[str, float]:
        """
        Update the agent's networks using experience replay.
        
        Returns:
            Dictionary of training metrics
        """
        # Check if we have enough samples
        if len(self.memory) < self.batch_size:
            return {"actor_loss": 0, "critic_loss": 0}
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert lists to numpy arrays first
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update critic
        with torch.no_grad():
            # Get next actions from target actor
            next_outputs = self.actor_target.forward(next_states)
            next_actions = self.action_interpreter.interpret(
                {k: v.detach().cpu().numpy() for k, v in next_outputs.items()},
                actions.detach().cpu().numpy(),
                deterministic=True
            )
            next_actions = torch.FloatTensor(next_actions).to(self.device)
            
            # Get target Q-values
            target_q = self.critic_target.forward(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q
        
        # Get current Q-values
        current_q = self.critic.forward(states, actions)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        # Get actions from actor
        actor_outputs = self.actor.forward(states)
        actor_actions = self.action_interpreter.interpret(
            {k: v.detach().cpu().numpy() for k, v in actor_outputs.items()},
            actions.detach().cpu().numpy(),
            deterministic=True
        )
        actor_actions = torch.FloatTensor(actor_actions).to(self.device)
        
        # Compute actor loss
        actor_loss = -self.critic.forward(states, actor_actions).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._update_target_networks()
        
        # Store loss values
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        
        # Increment steps
        self.steps += 1
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }
    
    def _update_target_networks(self):
        """Soft update target networks."""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_model_name(self) -> str:
        """Get the name of the model."""
        return "DDPG"
    
    def get_model(self) -> nn.Module:
        """Get the actor network."""
        return self.actor
    
    def train(self, mode: bool = True) -> None:
        """Set the model to training mode."""
        self.actor.train(mode)
        self.critic.train(mode)
        self.actor_target.train(mode)
        self.critic_target.train(mode)
    
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
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'steps': self.steps
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.steps = checkpoint['steps']
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state dictionary directly into the model.
        Used for backtesting when models are passed directly without saving to disk.
        
        Args:
            state_dict: State dictionary containing model weights
        """
        if isinstance(state_dict, dict):
            self.actor.load_state_dict(state_dict)
            logger.info("Loaded model state dictionary directly into agent")
        else:
            logger.error(f"Invalid state dict format: {type(state_dict)}")
    
    def get_info(self) -> Dict:
        """
        Get additional information for visualization and logging.
        
        Returns:
            Dictionary with agent information
        """
        return {
            "actor_loss": self.last_actor_loss if hasattr(self, 'last_actor_loss') else 0,
            "critic_loss": self.last_critic_loss if hasattr(self, 'last_critic_loss') else 0
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the agent."""
        return self.config 