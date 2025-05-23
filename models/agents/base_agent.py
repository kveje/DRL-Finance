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
    
    @abstractmethod
    def get_intended_action(
        self,
        observations: Dict[str, torch.Tensor],
        current_position: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Select an action based on the current state.
        
        Args:
            observations: Dictionary of observation tensors for each processor
            current_position: Current position of the agent
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action array
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
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent's networks.
        
        Args:
            path: Base path to save the models
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the agent's networks.
        
        Args:
            path: Base path to load the models from
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
