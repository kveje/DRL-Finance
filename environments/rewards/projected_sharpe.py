import numpy as np
from typing import Dict, Any, List
from .base_reward import BaseReward
# import pandas as pd # pandas is not used

class ProjectedSharpeReward(BaseReward):
    """Reward function based on the Sharpe ratio of projected returns over a rolling window."""

    def __init__(self, config: Dict[str, Any], raw_data_feature_indices: Dict[str, int]):
        super().__init__(name="projected_sharpe")
        self.scale = config.get("scale", 1.0)
        self.price_column = config.get("price_column", "close")
        self.projection_period = config.get("projection_period", 20) # Number of projected returns to calculate Sharpe over
        self.annualization_factor = config.get("annualization_factor", 252)
        self.annual_risk_free_rate = config.get("annual_risk_free_rate", 0.00)
        self.daily_risk_free_rate = (1 + self.annual_risk_free_rate) ** (1/self.annualization_factor) - 1
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
 
        if len(future_days_for_prices) < 5:
            return 0.0

        # Fetch prices for the identified future days
        prices = raw_data[future_days_for_prices, :, self.raw_data_feature_indices[self.price_column]] # shape: (num_available_price_points, n_assets)
        
        # Compute projected portfolio values for these days
        # current_position shape: (n_assets,), prices shape: (num_available_price_points, n_assets)
        projected_portfolio_values = np.sum(current_position * prices, axis=1) + cash_balance

        projected_daily_returns = (projected_portfolio_values[1:] - projected_portfolio_values[:-1]) / projected_portfolio_values[:-1]

        # Compute the Sharpe ratio
        excess_returns = projected_daily_returns - self.daily_risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.annualization_factor)

        return sharpe_ratio * self.scale

    def __str__(self) -> str:
        return (f"ProjectedSharpeReward(projection_delay={self.projection_delay}, "
                f"window_size={self.window_size}, scale={self.scale}, "
                f"min_history_size={self.min_history_size})")

    def __repr__(self) -> str:
        return self.__str__()

    def reset(self):
        self.max_day = None
        # self.daily_returns = [] # Removed

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "projection_delay": self.projection_delay,
            "scale": self.scale,
            "window_size": self.window_size,
            "annualization_factor": self.annualization_factor,
            "annual_risk_free_rate": self.annual_risk_free_rate,
            "min_history_size": self.min_history_size
        } 