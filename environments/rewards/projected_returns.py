import numpy as np
from typing import Dict, Any, List
from .base_reward import BaseReward
import pandas as pd

class ProjectedReturnsReward(BaseReward):
    """Reward function based on projected returns (holding position for N days)."""

    def __init__(self, config: Dict[str, Any], raw_data_feature_indices: Dict[str, int]):
        """
        Initialize the projected returns-based reward function.
        Args:
            config: Configuration dictionary containing reward parameters.
                parameters:
                    projection_period: Number of steps to project the return (default: 5)
                    scale: Scale factor for the reward (default: 1.0)
                Example:
                {"projection_period": 5, "scale": 1.0}
        """
        super().__init__(name="projected_returns")
        self.projection_period = config.get("projection_period", 5)
        self.scale = config.get("scale", 1.0)
        self.price_column = config.get("price_column", "close")
        self.raw_data_feature_indices = raw_data_feature_indices

    def calculate(
        self,
        raw_data: np.ndarray,
        current_day: int,
        current_position: np.ndarray,
        previous_portfolio_value: float,
        cash_balance: float,
        **kwargs
    ) -> float:
        """
        Calculate projected return reward: the return if you hold the current_position from current_day for projection_period days.
        The reward is scaled by the scale factor.

        Args:
            raw_data: Raw data for the environment
            current_day: Current day index
            current_position: Current positions in each asset (after the trade)
            previous_portfolio_value: Previous portfolio value (before the trade)
            cash_balance: Cash balance (after the trade)
            **kwargs: Additional arguments
        Returns:
            float: The calculated projected return reward
        """
        # Get the max day
        if self.max_day is None:
            self.max_day = raw_data.shape[0]

        # Adjust projection_period if we don't have enough data
        current_projection_period = min(self.projection_period, self.max_day - current_day - 1)

        # Get the prices for the current day and the day + current_projection_period
        prices_at_projection_end = raw_data[current_day + current_projection_period, :, self.raw_data_feature_indices[self.price_column]] # shape: (n_assets,)

        # Get the current position
        current_position = current_position # shape: (n_assets,)

        # Get the previous portfolio value
        baseline_portfolio_value = previous_portfolio_value

        # Compute the portfolio value for the current day + current_projection_period
        portfolio_value_at_projection_end = np.sum(current_position * prices_at_projection_end) + cash_balance

        # Compute the projected return
        if baseline_portfolio_value == 0:
            projected_return = 0.0
        else:
            projected_return = (portfolio_value_at_projection_end - baseline_portfolio_value) / baseline_portfolio_value

        # Return the projected return scaled by the scale factor
        return projected_return * self.scale

    def __str__(self) -> str:
        return f"ProjectedReturnsReward(projection_period={self.projection_period}, scale={self.scale})"

    def __repr__(self) -> str:
        return self.__str__()

    def reset(self):
        self.max_day = None

    def get_parameters(self) -> Dict[str, Any]:
        return {"projection_period": self.projection_period, "scale": self.scale} 