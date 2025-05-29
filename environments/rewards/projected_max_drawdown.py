import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward
# import pandas as pd # pandas is not used

class ProjectedMaxDrawdownReward(BaseReward):
    """Reward function based on the max drawdown over a projected forward window."""

    def __init__(self, config: Dict[str, Any], raw_data_feature_indices: Dict[str, int]):
        super().__init__(name="projected_max_drawdown")
        self.projection_period = config.get("projection_period", 20) # Number of steps in the projection window
        self.price_column = config.get("price_column", "close")
        self.scale = config.get("scale", 1.0)
        self.max_day = None
        self.raw_data_feature_indices = raw_data_feature_indices

    def calculate(
        self,
        raw_data: np.ndarray,
        current_day: int,
        current_position: np.ndarray,
        cash_balance: float,
        **kwargs
    ) -> float:
        # Get the max day
        if self.max_day is None:
            self.max_day = raw_data.shape[0]
        
        future_days_for_prices = [i for i in range(current_day, current_day + self.projection_period) if i < self.max_day]

        # Need at least 2 portfolio values to calculate a drawdown (1 period)
        if len(future_days_for_prices) <= 1:
            return 0.0

        # Get the prices for the future days
        prices = raw_data[future_days_for_prices, :, self.raw_data_feature_indices[self.price_column]]

        # Compute the projected portfolio values for the future days
        projected_portfolio_values = np.sum(current_position * prices, axis=1) + cash_balance

        # Compute max drawdown for the projected period
        running_max = np.maximum.accumulate(projected_portfolio_values)
        
        # Initialize drawdowns to 0 (no drawdown if running_max is non-positive or for the first element implicitly)
        drawdowns = np.zeros_like(projected_portfolio_values, dtype=float)

        
        drawdowns = (projected_portfolio_values - running_max) / running_max
            
        max_drawdown = np.min(drawdowns)  # This will be negative or zero

        # Reward is negative drawdown (so less drawdown is better, max is 0)
        # A max_drawdown of -0.1 (10% drop) results in a reward of 0.1.
        return max_drawdown * self.scale

    def __str__(self) -> str:
        return f"ProjectedMaxDrawdownReward(window_size={self.projection_period}, scale={self.scale})"

    def __repr__(self) -> str:
        return self.__str__()

    def reset(self):
        self.max_day = None

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "window_size": self.projection_period,
            "price_column": self.price_column, # Retaining as it was there
            "scale": self.scale
        } 