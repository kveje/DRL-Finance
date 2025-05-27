import numpy as np
import torch
from tensordict import TensorDict
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any
from torchrl.data import ReplayBuffer, LazyTensorStorage
import random
import os

from utils.logger import Logger
logger = Logger.get_logger()


from models.agents.base_agent import BaseAgent
from models.networks.unified_network import UnifiedNetwork
from models.action_interpreters.base_action_interpreter import BaseActionInterpreter
from models.agents.temperature_manager import TemperatureManager

class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent implementation.
    Uses a unified network for Q-value prediction and works with any action interpreter.
    """
    def __init__(
        self,
        network_config: Dict[str, Any],
        interpreter: BaseActionInterpreter,
        temperature_manager: TemperatureManager,
        update_frequency: int,
        device: str = "cuda",
        memory_size: int = 10000,
        batch_size: int = 128,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.999,
        target_update: int = 10,
        learning_rate: float = 0.001,
        use_bayesian: bool = False,
        **kwargs
    ):
        """
        Initialize the DQN agent.
        
        Args:
            network_config: Configuration for the unified network
            interpreter: Action interpreter instance
            temperature_manager: Temperature manager instance
            update_frequency: Update frequency for the temperature manager
            device: Device to run the agent on
            memory_size: Size of the replay memory
            batch_size: Batch size for training
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration rate decay
            target_update: Frequency of target network updates
            learning_rate: Learning rate for the optimizer
            use_bayesian: Whether to use Bayesian heads for exploration
            **kwargs: Additional arguments
        """
        super().__init__(network_config, interpreter, temperature_manager, update_frequency, device)
        
        # DQN specific parameters
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.learning_rate = learning_rate
        self.use_bayesian = use_bayesian
        
        # Initialize replay memory
        self.memory_device = "cpu"
        self.memory = ReplayBuffer(
            storage=LazyTensorStorage(max_size=memory_size,
                                      device=self.memory_device),
            batch_size=batch_size,
        )
        
        # Create target network
        self.target_network = UnifiedNetwork(network_config, device=device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training step counter
        self.steps = 0
    
    def get_intended_action(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = True,
        sample: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observations: Dictionary of observation numpy arrays
            deterministic: Whether to use deterministic action selection
            sample: Whether to sample from the distribution

        Returns:
            Tuple of (scaled_action, action_choices) where:
            - scaled_action is the actual action to execute
            - action_choices is the unscaled action (-1,0,1) for learning
        """
        # Convert observations to tensors
        obs_tensors = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in observations.items()
        }
        
        use_sampling = self.use_bayesian and sample
        # Get network outputs
        with torch.no_grad():
            # Use sampling if not deterministic and Bayesian is enabled
            network_outputs = self.network(obs_tensors, use_sampling=use_sampling, temperature = self.temperature_manager.get_all_temperatures())
        
        # Move the network outputs to cpu
        network_outputs = {k: v.cpu() for k, v in network_outputs.items()}

        # Epsilon-greedy action selection
        if not deterministic and random.random() < self.epsilon:
            scaled_action, action_choice = self.interpreter.interpret(
                network_outputs=network_outputs,
                deterministic=False
            )
        else:
            # Use interpreter to convert network outputs to actions
            scaled_action, action_choice = self.interpreter.interpret(
                network_outputs=network_outputs,
                deterministic=True
            )

        # Convert to numpy
        scaled_action = scaled_action.numpy()
        action_choice = action_choice.numpy()
        
        return scaled_action, action_choice
    
    def update(self) -> Dict[str, float]:
        """
        Update the network using a batch of experience.
        
        Args:
            batch: Dictionary containing:
                - observations: Dict of observation tensors
                - actions: Action choices tensor
                - rewards: Reward tensor
                - next_observations: Dict of next observation tensors
                - dones: Done flag tensor
                
        Returns:
            Dictionary of training metrics
        """

        # Get batch from memory
        batch = self.get_batch()

        # Extract batch data
        obs = batch['observations'] # Dict {"process_str": tensor[batch_size, ...], ...}
        action_choices = batch['actions'] # Tensor [batch_size, 1]
        rewards = batch['rewards'] # Tensor [batch_size, 1]
        next_obs = batch['next_observations'] # Dict {"process_str": tensor[batch_size, ...], ...}
        dones = batch['dones'] # Tensor [batch_size, 1]

        # Move the batch to the device
        obs = {k: v.to(self.device) for k, v in obs.items()}
        action_choices = action_choices.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = {k: v.to(self.device) for k, v in next_obs.items()}
        dones = dones.to(self.device)
        
        # Get current network outputs
        current_outputs = self.network(obs, use_sampling=False, temperature = self.temperature_manager.get_all_temperatures())

        # Get target network outputs
        with torch.no_grad():
            target_outputs = self.target_network(next_obs)
            
        # Compute loss using the interpreter
        loss = self.interpreter.compute_loss(
            current_outputs=current_outputs,
            target_outputs=target_outputs,
            action_choices=action_choices,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            method="mse"
        )
        
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store loss for visualization
        self.last_loss = loss.item()

        metrics = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
        }

        if self.use_bayesian:
            metrics.update(self.temperature_manager.get_all_temperatures_printerfriendly())

        return metrics
    
    def save(self, path: str) -> None:
        """
        Save the agent's networks and optimizer state.
        
        Args:
            path: Base path to save the models
        """
        os.makedirs(path, exist_ok=True)
        
        # Save main network
        torch.save(self.network.state_dict(), os.path.join(path, 'network.pth'))
        
        # Save target network
        torch.save(self.target_network.state_dict(), os.path.join(path, 'target_network.pth'))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pth'))
        
        # Save agent state
        agent_state = {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'temperature': self.interpreter.get_temperature() if self.use_bayesian else None
        }
        torch.save(agent_state, os.path.join(path, 'agent_state.pth'))
    
    def load(self, path: str) -> None:
        """
        Load the agent's networks and optimizer state.
        
        Args:
            path: Base path to load the models from
        """
        # Load network weights
        self.network.load_state_dict(torch.load(os.path.join(path, 'network.pth'), weights_only=False))
        
        # Load target network weights
        self.target_network.load_state_dict(torch.load(os.path.join(path, 'target_network.pth'), weights_only=False))
        
        # Load optimizer state
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth'), weights_only=False))
        
        # Load agent state
        agent_state = torch.load(os.path.join(path, 'agent_state.pth'), weights_only=False)
        self.epsilon = agent_state['epsilon']
        self.steps = agent_state['steps']
        if self.use_bayesian and 'temperature' in agent_state:
            self.interpreter.set_temperature(agent_state['temperature'])
    
    def get_model_name(self) -> str:
        """Get the name of the model."""
        return "DQN"
    
    def get_model(self) -> nn.Module:
        """Get the policy network."""
        return self.network
    
    def train(self, mode: bool = True) -> None:
        """Set the model to training mode."""
        self.network.train(mode)
        self.target_network.train(mode)
    
    def eval(self) -> None:
        """Set the model to evaluation mode."""
        self.train(mode=False)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the agent."""
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "target_update": self.target_update,
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "use_bayesian": self.use_bayesian,
            "network_config": self.network_config
        }

    def add_to_memory(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        action_choice: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool
    ) -> None:
        """
        Add a transition to the replay memory.
        
        Args:
            observation: Current observation dictionary
            action: Scaled action taken (from interpret())
            action_choices: Unscaled action choice (-1,0,1) for learning
            reward: Reward received
            next_observation: Next observation dictionary
            done: Whether the episode is done
        """

        # Convert numpy arrays to tensors
        obs_tensors = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in observation.items()
        }
        next_obs_tensors = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in next_observation.items()
        }

        transition = TensorDict({
            'observations': obs_tensors,
            'actions': torch.as_tensor(action_choice,  dtype=torch.long),
            'rewards': torch.as_tensor(reward, dtype=torch.float32),
            'next_observations': next_obs_tensors,
            'dones': torch.as_tensor(done, dtype=torch.float32)
        }, device=self.memory_device)

        self.memory.add(transition)
    
    def get_batch(self) -> TensorDict:
        """
        Sample a batch of transitions from the replay memory.
        
        Returns:
            TensorDict containing batched tensors for:
            - observations: Dict of observation tensors
            - actions: Action indices tensor
            - rewards: Reward tensor
            - next_observations: Dict of next observation tensors
            - dones: Done flag tensor
        """
        if len(self.memory) < self.batch_size:
            raise ValueError(f"Not enough samples in memory. Need {self.batch_size}, have {len(self.memory)}")

        return self.memory.sample()

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information for visualization.
        
        Returns:
            Dictionary containing agent information:
            - epsilon: Current exploration rate
            - loss: Last training loss
            - temperature: Current temperature for Bayesian exploration (if enabled)
        """
        info = {
            'epsilon': self.epsilon,
            'loss': getattr(self, 'last_loss', None)
        }
        
        if self.use_bayesian:
            temperatures = self.temperature_manager.get_all_temperatures_printerfriendly()
            for key, value in temperatures.items():
                info[key] = value
        
        return info

    def _get_agent_specific_state(self) -> Dict[str, Any]:
        """
        Get DQN-specific state for checkpointing.
        
        Returns:
            Dictionary containing DQN-specific state
        """
        state = {}
        state['epsilon'] = self.epsilon
        state['target_network'] = self.target_network.state_dict()
        if self.use_bayesian:
            steps_info = self.temperature_manager.get_step_info()
            for key, value in steps_info.items():
                state[key] = value
        return state
    
    def _load_agent_specific_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load DQN-specific state from checkpoint.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        self.epsilon = state_dict.get('epsilon', self.epsilon)
        if 'target_network_state_dict' in state_dict:
            self.target_network.load_state_dict(state_dict['target_network_state_dict'])

        if self.use_bayesian:
            steps_info = self.temperature_manager.get_step_info()
            for key, value in steps_info.items():
                state_dict[key] = value
            
            self.temperature_manager.set_step_info(**steps_info)

    def sufficient_memory(self) -> bool:
        """Check if the agent has enough memory to perform an update."""
        return len(self.memory) >= self.batch_size