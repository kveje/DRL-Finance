"""Base class for all agents"""

from abc import ABC, abstractmethod
from environments.base_env import BaseTradingEnv
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..networks.unified_network import UnifiedNetwork
from ..action_interpreters.base_action_interpreter import BaseActionInterpreter

class BaseAgent(ABC):
    """
    Base class for all DRL agents.
    Provides common functionality and interface for all agent implementations.
    """
    
    def __init__(
        self,
        env: BaseTradingEnv,
        network_config: Dict[str, Any],
        interpreter: BaseActionInterpreter,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            env: Trading environment instance
            network_config: Configuration for the unified network
            interpreter: Action interpreter instance to use
            device: Device to run the agent on
            **kwargs: Additional arguments specific to the agent implementation
        """
        self.env = env
        self.device = device
        self.interpreter = interpreter
        
        # Initialize unified network
        self.network = UnifiedNetwork(network_config, device=device)
        
        # Store network configuration
        self.network_config = network_config
        
        # Track training start time
        self.start_time = None
    
    @abstractmethod
    def get_intended_action(
        self,
        observations: Dict[str, torch.Tensor],
        current_position: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select an action based on the current state.
        
        Args:
            observations: Dictionary of observation tensors for each processor
            current_position: Current position of the agent
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (scaled_action, action_choice) where:
            - scaled_action is the actual action to execute
            - action_choice is the unscaled action (-1,0,1) for learning
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's networks using a batch of experience.
        
        Args:
            batch: Dictionary containing experience data
            
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
            "device": self.device
        }
