import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward


class SharpeReward(BaseReward):
    """Reward function based on Sharpe ratio."""

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """
        Initialize the Sharpe ratio-based reward function.

        Args:
            config: Configuration dictionary containing reward parameters.
                parameters:
                    annual_risk_free_rate: Annual risk-free rate
                    annualization_factor: Annualization factor for the Sharpe ratio (default is 252 for daily data)
                    window_size: Window size for calculating returns and volatility
                    min_history_size: Minimum history size for calculating the Sharpe ratio
                    scale: Scale factor for the reward
                
                Example:
                {"annual_risk_free_rate": 0.02, "annualization_factor": 252, "window_size": 20, "min_history_size": 10, "scale": 1.0}
        """
        super().__init__(name="sharpe_based")
        self.annual_risk_free_rate = config.get("annual_risk_free_rate", 0.02)
        self.annualization_factor = config.get("annualization_factor", 252)
        self.daily_risk_free_rate = (1 + self.annual_risk_free_rate) ** (1/self.annualization_factor) - 1
        self.window_size = config.get("window_size", 20)
        self.min_history_size = config.get("min_history_size", 10)
        self.returns_history = []
        self.scale = config.get("scale", 1.0)

    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate reward based on Sharpe ratio.

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

        # Update returns history
        self.returns_history.append(portfolio_return)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # Calculate Sharpe ratio if we have enough data
        if len(self.returns_history) >= self.min_history_size:
            returns_array = np.array(self.returns_history)
            excess_returns = returns_array - self.daily_risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(self.annualization_factor)
        # If we have less than min_history_size returns, use the std of the returns history
        elif len(self.returns_history) > 1:
            sharpe_ratio = (portfolio_return - self.daily_risk_free_rate) / (np.std(self.returns_history) + 1e-9) * np.sqrt(self.annualization_factor)
        # If only one period (no history), use the excess return of the portfolio return annualized
        else:
            sharpe_ratio = (portfolio_return - self.daily_risk_free_rate) * np.sqrt(self.annualization_factor)
        
        # Final reward is Sharpe ratio scaled by the scale factor
        return sharpe_ratio * self.scale
    
    def reset(self):
        """Reset the reward function."""
        self.returns_history = []
    
    def __str__(self) -> str:
        """Return a string representation of the reward function."""
        return f"SharpeBasedReward(annual_risk_free_rate={self.annual_risk_free_rate}, window_size={self.window_size}, scale={self.scale})"
    
    def __repr__(self) -> str:
        """Return a string representation of the reward function."""
        return self.__str__()
