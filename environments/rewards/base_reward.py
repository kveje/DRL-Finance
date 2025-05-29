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
        **kwargs
    ) -> float:
        """
        Calculate the reward for the current step.

        Args:
            **kwargs: Arguments for the reward function

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
