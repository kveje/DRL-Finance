from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from enum import Enum

class ObservationFormat(Enum):
    """Different formats for processing observations."""
    FLAT = "flat"  # Flatten everything into a single vector
    STACKED = "stacked"  # Keep the 3D structure of market data

class BaseObservationProcessor(ABC):
    """
    Base class for all observation processors.
    Defines the interface that all environment-specific processors must implement.
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the base processor.
        
        Args:
            env: The environment instance
            device: Device to place tensors on
        """
        self.device = device
    
    @abstractmethod
    def process(
        self,
        observation: Dict[str, Any],
        format: ObservationFormat = ObservationFormat.FLAT
    ) -> torch.Tensor:
        """
        Process the observation into the specified tensor format.
        
        Args:
            observation: Observation from the environment
            format: Desired output format
            
        Returns:
            Processed tensor in the specified format
        """
        pass
    
    @abstractmethod
    def get_observation_dim(self, format: ObservationFormat) -> int:
        """
        Get the dimension of the processed observation for the specified format.
        
        Args:
            format: Observation format
            
        Returns:
            Dimension of the processed observation
        """
        pass 