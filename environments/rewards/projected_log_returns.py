import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward
# import pandas as pd # pandas is not used

class ProjectedLogReturnsReward(BaseReward):
    """Reward function based on projected log returns (holding position for N days)."""

    def __init__(self, config: Dict[str, Any], raw_data_feature_indices: Dict[str, int]):
        super().__init__(name="projected_log_returns")
        self.projection_period = config.get("projection_period", 5)
        self.scale = config.get("scale", 1.0)
        self.price_column = config.get("price_column", "close")
        self.max_day = None
        self.raw_data_feature_indices = raw_data_feature_indices

    def calculate(
        self,
        raw_data: np.ndarray,
        current_day: int,
        current_position: np.ndarray,
        previous_portfolio_value: float, # Asset component of portfolio value before trade / after trade at current prices
        cash_balance: float, # Cash balance after trade
        **kwargs
    ) -> float:
        """Calculate the reward based on the projected log returns."""
        baseline_portfolio_value = previous_portfolio_value
        
        # Determine the projection period 
        if self.max_day is None:
            self.max_day = raw_data.shape[0]
        
        current_projection_period = min(self.projection_period, self.max_day - current_day - 1)
        prices_at_projection_end = raw_data[current_day + current_projection_period, :, self.raw_data_feature_indices[self.price_column]]

        # Calculate the portfolio value at the end of the projection period
        portfolio_value_at_projection_end = np.sum(current_position * prices_at_projection_end) + cash_balance
        
        # Calculate the projected log return
        projected_log_return = np.log(portfolio_value_at_projection_end / baseline_portfolio_value)

        return projected_log_return * self.scale

    def __str__(self) -> str:
        return f"ProjectedLogReturnsReward(projection_period={self.projection_period}, scale={self.scale})"

    def __repr__(self) -> str:
        return self.__str__()

    def reset(self):
        self.max_day = None

    def get_parameters(self) -> Dict[str, Any]:
        return {"projection_period": self.projection_period, "scale": self.scale} 