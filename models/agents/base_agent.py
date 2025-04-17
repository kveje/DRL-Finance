"""Base class for all agents"""

from abc import ABC, abstractmethod
from environments.base_env import BaseTradingEnv
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..networks.base_network import BaseNetwork

class BaseAgent(ABC):
    """
    Base class for all DRL agents.
    Provides common functionality and interface for all agent implementations.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            device: Device to run the agent on
            **kwargs: Additional arguments specific to the agent implementation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
    
    @abstractmethod
    def get_intended_action(self, **kwargs) -> np.ndarray:
        """
        Select an action based on the current state.
        
        Args:
            **kwargs: Additional arguments specific to the agent implementation
            
        Returns:
            Selected action
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
    
    @abstractmethod
    def train(self) -> None:
        """Set the agent's networks to training mode."""
        pass
    
    @abstractmethod
    def eval(self) -> None:
        """Set the agent's networks to evaluation mode."""
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_config(self):
        pass
