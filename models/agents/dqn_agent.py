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
from models.action_interpreters.discrete_interpreter import DiscreteInterpreter
from models.action_interpreters.confidence_scaled_interpreter import ConfidenceScaledInterpreter
from environments.trading_env import TradingEnv
from .base_agent import BaseAgent
from ..networks.unified_network import UnifiedNetwork
from ..action_interpreters.base_action_interpreter import BaseActionInterpreter

class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent implementation.
    Uses a unified network for Q-value prediction and works with any action interpreter.
    """
    def __init__(
        self,
        env: Any,
        network_config: Dict[str, Any],
        interpreter: BaseActionInterpreter,
        device: str = "cuda",
        memory_size: int = 100000,
        batch_size: int = 64,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update: int = 10,
        learning_rate: float = 0.001,
        use_bayesian: bool = False,
        **kwargs
    ):
        """
        Initialize the DQN agent.
        
        Args:
            env: Trading environment instance
            network_config: Configuration for the unified network
            interpreter: Action interpreter instance
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
        super().__init__(env, network_config, interpreter, device)
        
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
        self.memory = deque(maxlen=memory_size)
        
        # Create target network
        self.target_network = UnifiedNetwork(network_config, device=device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training step counter
        self.steps = 0
    
    def get_intended_action(
        self,
        observations: Dict[str, torch.Tensor],
        current_position: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observations: Dictionary of observation tensors
            current_position: Current position array
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (scaled_action, action_choice) where:
            - scaled_action is the actual action to execute
            - action_choice is the unscaled action (-1,0,1) for learning
        """
        # Convert observations to tensors if needed
        obs_tensors = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in observations.items()
        }
        
        # Get network outputs
        with torch.no_grad():
            network_outputs = self.network(obs_tensors, use_sampling=not deterministic and self.use_bayesian)
        
        # Epsilon-greedy action selection
        if not deterministic and random.random() < self.epsilon:
            # Let the interpreter handle random action generation
            scaled_action = self.interpreter.interpret(
                network_outputs=None,  # Signal to interpreter to generate random actions
                current_position=current_position,
                deterministic=False
            )
            action_choice = self.interpreter.get_action_choice(
                network_outputs=None,
                deterministic=False
            )
        else:
            # Use interpreter to convert network outputs to actions
            scaled_action = self.interpreter.interpret(
                network_outputs=network_outputs,
                current_position=current_position,
                deterministic=True
            )
            action_choice = self.interpreter.get_action_choice(
                network_outputs=network_outputs,
                deterministic=True
            )
        
        return scaled_action, action_choice
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the network using a batch of experience.
        
        Args:
            batch: Dictionary containing:
                - observations: Dict of observation tensors
                - actions: Action indices tensor
                - rewards: Reward tensor
                - next_observations: Dict of next observation tensors
                - dones: Done flag tensor
                
        Returns:
            Dictionary of training metrics
        """
        # Extract batch data
        obs = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_observations']
        dones = batch['dones']
        
        # Get current network outputs
        current_outputs = self.network(obs)
        
        # Validate network outputs
        if not isinstance(current_outputs, dict):
            raise ValueError("Network outputs must be a dictionary")
        
        # Check for required outputs based on interpreter type
        if isinstance(self.interpreter, DiscreteInterpreter):
            if 'action_probs' not in current_outputs and not ('alphas' in current_outputs and 'betas' in current_outputs):
                raise ValueError("Network outputs must contain either 'action_probs' or 'alphas' and 'betas'")
        elif isinstance(self.interpreter, ConfidenceScaledInterpreter):
            if not (('action_probs' in current_outputs and 'confidences' in current_outputs) or 
                   ('alphas' in current_outputs and 'betas' in current_outputs)):
                print(current_outputs)
                raise ValueError("Network outputs must contain either ('action_probs', 'confidences') or ('alphas', 'betas')")
        
        # Get target network outputs
        with torch.no_grad():
            target_outputs = self.target_network(next_obs)
            
            # Validate target outputs
            if not isinstance(target_outputs, dict):
                raise ValueError("Target network outputs must be a dictionary")
            
            # Check for required outputs based on interpreter type
            if isinstance(self.interpreter, DiscreteInterpreter):
                if 'action_probs' not in target_outputs and not ('alphas' in target_outputs and 'betas' in target_outputs):
                    raise ValueError("Target network outputs must contain either 'action_probs' or 'alphas' and 'betas'")
            elif isinstance(self.interpreter, ConfidenceScaledInterpreter):
                if not (('action_probs' in target_outputs and 'confidences' in target_outputs) or 
                       ('alphas' in target_outputs and 'betas' in target_outputs)):
                    raise ValueError("Target network outputs must contain either ('action_probs', 'confidences') or ('alphas', 'betas')")
        
        # Compute loss using the interpreter
        loss = self.interpreter.compute_loss(
            current_outputs=current_outputs,
            target_outputs=target_outputs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma
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
        
        # Update temperature for Bayesian exploration
        if self.use_bayesian:
            self.interpreter.update_temperature()
        
        # Store loss for visualization
        self.last_loss = loss.item()

        metrics = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
        }

        if self.use_bayesian:
            metrics["temperature"] = self.interpreter.get_temperature()

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
        # Load main network
        self.network.load_state_dict(torch.load(os.path.join(path, 'network.pth')))
        
        # Load target network
        self.target_network.load_state_dict(torch.load(os.path.join(path, 'target_network.pth')))
        
        # Load optimizer state
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth')))
        
        # Load agent state
        agent_state = torch.load(os.path.join(path, 'agent_state.pth'))
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
            action_choice: Unscaled action choice (-1,0,1) for learning
            reward: Reward received
            next_observation: Next observation dictionary
            done: Whether the episode is done
        """
        # Convert numpy arrays to tensors
        obs_tensors = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in observation.items()
        }
        next_obs_tensors = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in next_observation.items()
        }
        action_tensor = torch.as_tensor(action_choice, device=self.device, dtype=torch.long)
        reward_tensor = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        done_tensor = torch.as_tensor(done, device=self.device, dtype=torch.float32)
        
        # Store transition
        self.memory.append({
            'observations': obs_tensors,
            'actions': action_tensor,  # Store unscaled action choice
            'rewards': reward_tensor,
            'next_observations': next_obs_tensors,
            'dones': done_tensor
        })
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the replay memory.
        
        Returns:
            Dictionary containing batched tensors for:
            - observations: Dict of observation tensors
            - actions: Action indices tensor
            - rewards: Reward tensor
            - next_observations: Dict of next observation tensors
            - dones: Done flag tensor
        """
        if len(self.memory) < self.batch_size:
            raise ValueError(f"Not enough samples in memory. Need {self.batch_size}, have {len(self.memory)}")
        
        # Sample batch
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        # Stack tensors
        batched = {
            'observations': {},
            'next_observations': {}
        }
        
        # Stack observations and next_observations
        for key in batch[0]['observations'].keys():
            batched['observations'][key] = torch.stack([t['observations'][key] for t in batch])
            batched['next_observations'][key] = torch.stack([t['next_observations'][key] for t in batch])
        
        # Stack other tensors
        batched['actions'] = torch.stack([t['actions'] for t in batch])
        batched['rewards'] = torch.stack([t['rewards'] for t in batch])
        batched['dones'] = torch.stack([t['dones'] for t in batch])
        
        return batched

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
            info['temperature'] = self.interpreter.get_temperature()
        
        return info

    def _get_agent_specific_state(self) -> Dict[str, Any]:
        """
        Get DQN-specific state for checkpointing.
        
        Returns:
            Dictionary containing DQN-specific state
        """
        return {
            'epsilon': self.epsilon,
            'target_network': self.target_network.state_dict(),
            'temperature': self.interpreter.get_temperature() if self.use_bayesian else None
        }
    
    def _load_agent_specific_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load DQN-specific state from checkpoint.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        self.epsilon = state_dict.get('epsilon', self.epsilon)
        if 'target_network_state_dict' in state_dict:
            self.target_network.load_state_dict(state_dict['target_network_state_dict'])
        if self.use_bayesian and 'temperature' in state_dict:
            self.interpreter.set_temperature(state_dict['temperature'])