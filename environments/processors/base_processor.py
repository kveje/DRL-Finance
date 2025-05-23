from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from gym import spaces

class BaseProcessor(ABC):
    """
    Base class for all data processors.
    Each processor should implement methods for processing data and defining observation spaces.
    """
    
    @abstractmethod
    def process(self, data: Any) -> np.ndarray:
        """
        Process the input data into features.
        
        Args:
            data: Input data to process
            
        Returns:
            np.ndarray: Processed features
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the observation space for this processor.
        
        Returns:
            Dict containing the observation space specification with:
                - 'low': float - Lower bound of the space
                - 'high': float - Upper bound of the space
                - 'shape': tuple - Shape of the space
                - 'dtype': type - Data type of the space
        """
        pass
    
    @abstractmethod
    def get_input_dim(self) -> Tuple[int, ...]:
        """
        Get the input dimension for the network.
        
        Returns:
            tuple[int, ...]: Input dimensions
        """
        pass 