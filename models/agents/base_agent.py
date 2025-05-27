"""Base class for all agents"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from models.networks.unified_network import UnifiedNetwork
from models.action_interpreters.base_action_interpreter import BaseActionInterpreter
from models.agents.temperature_manager import TemperatureManager

class BaseAgent(ABC):
    """
    Base class for all DRL agents.
    Provides common functionality and interface for all agent implementations.
    """
    
    def __init__(
        self,
        network_config: Dict[str, Any],
        interpreter: BaseActionInterpreter,
        temperature_manager: TemperatureManager,
        update_frequency: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            network_config: Configuration for the unified network
            interpreter: Action interpreter instance to use
            device: Device to run the agent on
            **kwargs: Additional arguments specific to the agent implementation
        """
        self.device = device
        self.interpreter = interpreter
        self.temperature_manager = temperature_manager
        self.update_frequency = update_frequency
        
        # Initialize unified network
        self.network = UnifiedNetwork(network_config, device=device)
        
        # Store network configuration
        self.network_config = network_config
        
        # Track training start time
        self.start_time = None
    
    @abstractmethod
    def get_intended_action(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = True,
        sample: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select an action based on the current state.
        
        Args:
            observations: Dictionary of observation numpy arrays for each processor
            deterministic: Whether to use deterministic action selection
            sample: Whether to sample from the distribution
            
        Returns:
            Tuple of (scaled_action, action_choice) where:
            - scaled_action is the actual action to execute
            - action_choice is the unscaled action (-1,0,1) for learning
        """
        pass
    
    @abstractmethod
    def update(self, **kwargs) -> Dict[str, float]:
        """
        Update the agent's networks using a batch of experience.
        
        Args:
            **kwargs: Additional arguments specific to the agent implementation
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get the agent's state dictionary for checkpointing.
        
        Returns:
            Dictionary containing the agent's state
        """
        state = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': getattr(self, 'steps', 0),
            'start_time': self.start_time
        }
        
        # Add agent-specific state
        agent_state = self._get_agent_specific_state()
        if agent_state:
            state.update(agent_state)
            
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the agent's state from a checkpoint.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        # Load common state
        self.network.load_state_dict(state_dict['network_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.steps = state_dict.get('steps', 0)
        self.start_time = state_dict.get('start_time')
        
        # Load agent-specific state
        self._load_agent_specific_state(state_dict)
    
    @abstractmethod
    def _get_agent_specific_state(self) -> Dict[str, Any]:
        """
        Get agent-specific state for checkpointing.
        Override this method in agent implementations to add custom state.
        
        Returns:
            Dictionary containing agent-specific state
        """
        pass
    
    @abstractmethod
    def _load_agent_specific_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load agent-specific state from checkpoint.
        Override this method in agent implementations to load custom state.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the agent.
        """
        pass
    
    def train(self) -> None:
        """Set the agent's networks to training mode."""
        self.network.train()
    
    def eval(self) -> None:
        """Set the agent's networks to evaluation mode."""
        self.network.eval()
    
    def get_model(self) -> UnifiedNetwork:
        """Get the unified network instance."""
        return self.network
    
    def get_model_name(self) -> str:
        """Get the name of the agent's model."""
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration."""
        return {
            "network_config": self.network_config,
            "device": self.device,
            "update_frequency": self.update_frequency,
        }
    
    def add_to_rollout(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        action_choice: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool) -> None:
        """
        Add experience to the rollout buffer.
        """
        raise NotImplementedError("This method should be implemented by the agent subclass.")
    
    def add_to_memory(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        action_choice: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool) -> None:
        """
        Add experience to the memory buffer.
        """
        raise NotImplementedError("This method should be implemented by the agent subclass.")
    
    def sufficient_memory(self) -> bool:
        """Check if the agent has enough memory."""
        raise NotImplementedError("This method should be implemented by the agent subclass.")
    
    