from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseReward(ABC):
    """Base class for all reward functions."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        positions: np.ndarray,
        price_changes: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """
        Calculate the reward for the current step.

        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Portfolio value from previous step
            positions: Current positions in each asset
            price_changes: Price changes for each asset
            info: Additional information about the environment state

        Returns:
            float: The calculated reward
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the reward function."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the reward function."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the reward function."""
        pass
