import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward


class ReturnsBasedReward(BaseReward):
    """Reward function based on portfolio returns."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the returns-based reward function.
        
        Args:
            config: Configuration dictionary containing reward parameters.
                parameters:
                    scale: Scale factor for the reward
                
                Example:
                {"scale": 1.0}
        """
        super().__init__(name="returns_based")
        self.scale = config.get("scale", 1.0)

    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate reward based on portfolio returns.

        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Portfolio value from previous step
            **kwargs: Additional arguments

        Returns:
            float: The calculated reward
        """
        # Calculate portfolio return
        portfolio_return = (
            portfolio_value - previous_portfolio_value
        ) / previous_portfolio_value

        return portfolio_return * self.scale # Scale the reward

    def __str__(self) -> str:
        """Return a string representation of the reward function."""
        return f"ReturnsBasedReward()"

    def __repr__(self) -> str:
        """Return a string representation of the reward function."""
        return self.__str__()

    def reset(self):
        """Reset the reward function."""
        pass

