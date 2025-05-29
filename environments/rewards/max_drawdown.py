import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward
import pandas as pd

class MaxDrawdownReward(BaseReward):
    """Reward function based on the max drawdown over a backward window (past only)."""

    def __init__(self, config: Dict[str, Any], raw_data_feature_indices: Dict[str, int]):
        super().__init__(name="max_drawdown")
        self.window_size = config.get("window_size", 20)
        self.price_column = config.get("price_column", "close")
        self.scale = config.get("scale", 1.0)
        self.raw_data_feature_indices = raw_data_feature_indices

    def calculate(
        self,
        raw_data: np.ndarray,
        current_day: int,
        current_position: np.ndarray,
        **kwargs
    ) -> float:
        
        # Compute the portfolio values for the previous window_size+1 days
        start_day = current_day - self.window_size
        if start_day < 0:
            return 0.0
        
        window_days = [start_day + i for i in range(self.window_size + 1)]
        if window_days[-1] > raw_data.shape[0]:
            return 0.0

        prices = raw_data[window_days, :, self.raw_data_feature_indices[self.price_column]]
        portfolio_values = np.sum(current_position * prices, axis=1)

        # Compute max drawdown (past only)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns)  # This will be negative or zero

        # Reward is negative drawdown (so less drawdown is better, max is 0)
        return max_drawdown * self.scale

    def __str__(self) -> str:
        return f"MaxDrawdownReward(window_size={self.window_size}, scale={self.scale})"

    def __repr__(self) -> str:
        return self.__str__()

    def reset(self):
        pass

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "price_column": self.price_column,
            "scale": self.scale
        } 