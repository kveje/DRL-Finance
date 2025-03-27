import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward


class SharpeBasedReward(BaseReward):
    """Reward function based on Sharpe ratio."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        window_size: int = 20,
        transaction_cost_penalty: float = 0.1,
    ):
        """
        Initialize the Sharpe ratio-based reward function.

        Args:
            risk_free_rate: Annual risk-free rate
            window_size: Window size for calculating returns and volatility
            transaction_cost_penalty: Penalty factor for transaction costs
        """
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.transaction_cost_penalty = transaction_cost_penalty
        self.returns_history = []
        self.previous_positions = None

    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        positions: np.ndarray,
        price_changes: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """
        Calculate reward based on Sharpe ratio and transaction costs.

        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Portfolio value from previous step
            positions: Current positions in each asset
            price_changes: Price changes for each asset
            info: Additional information about the environment state

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
        if len(self.returns_history) >= self.window_size:
            returns_array = np.array(self.returns_history)
            excess_returns = returns_array - (
                self.risk_free_rate / 252
            )  # Daily risk-free rate
            sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)
        else:
            sharpe_ratio = (
                portfolio_return  # Fallback to simple return if not enough data
            )

        # Calculate transaction cost penalty
        transaction_cost = 0.0
        if self.previous_positions is not None:
            position_changes = np.abs(positions - self.previous_positions)
            transaction_cost = np.sum(position_changes) * self.transaction_cost_penalty

        # Update previous positions
        self.previous_positions = positions.copy()

        # Final reward is Sharpe ratio minus transaction cost penalty
        return sharpe_ratio - transaction_cost
