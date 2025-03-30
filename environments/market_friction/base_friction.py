from abc import ABC, abstractmethod
import numpy as np

class BaseMarketFriction(ABC):
    """Base class for all market frictions."""

    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def apply(self, action: np.ndarray, price: np.ndarray) -> np.ndarray:
        """
        Apply the market friction to the action.
        
        Args:
            action: Numpy array of shape (n_assets,)
            price: Numpy array of shape (n_assets,)
        Returns:    
            np.ndarray: New prices after applying the friction
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the friction."""
        return f"Friction {self.name}"
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the friction."""
        return self.__str__()
