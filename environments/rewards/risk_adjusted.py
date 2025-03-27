import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward


class RiskAdjustedReward(BaseReward):
    """Reward function based on multiple risk-adjusted metrics."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        window_size: int = 20,
        transaction_cost_penalty: float = 0.1,
        max_drawdown_penalty: float = 0.5,
        volatility_penalty: float = 0.3,
    ):
        """
        Initialize the risk-adjusted reward function.

        Args:
            risk_free_rate: Annual risk-free rate
            window_size: Window size for calculating metrics
            transaction_cost_penalty: Penalty factor for transaction costs
            max_drawdown_penalty: Penalty factor for maximum drawdown
            volatility_penalty: Penalty factor for portfolio volatility
        """
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.transaction_cost_penalty = transaction_cost_penalty
        self.max_drawdown_penalty = max_drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.returns_history = []
        self.portfolio_values = []
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
        Calculate reward based on multiple risk-adjusted metrics.

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

        # Update history
        self.returns_history.append(portfolio_return)
        self.portfolio_values.append(portfolio_value)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)
            self.portfolio_values.pop(0)

        # Calculate various risk metrics if we have enough data
        if len(self.returns_history) >= self.window_size:
            returns_array = np.array(self.returns_history)
            portfolio_values_array = np.array(self.portfolio_values)

            # 1. Sharpe Ratio
            excess_returns = returns_array - (self.risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)

            # 2. Maximum Drawdown
            peak = np.maximum.accumulate(portfolio_values_array)
            drawdown = (peak - portfolio_values_array) / peak
            max_drawdown = np.max(drawdown)

            # 3. Volatility
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized volatility

            # 4. Sortino Ratio (downside risk only)
            downside_returns = returns_array[returns_array < 0]
            sortino_ratio = np.mean(excess_returns) / (np.std(downside_returns) + 1e-9)

            # Combine metrics with penalties
            reward = (
                sharpe_ratio * 0.4  # 40% weight on Sharpe ratio
                + sortino_ratio * 0.3  # 30% weight on Sortino ratio
                + portfolio_return * 0.3  # 30% weight on current return
            )

            # Apply penalties
            reward -= max_drawdown * self.max_drawdown_penalty
            reward -= volatility * self.volatility_penalty

        else:
            # Fallback to simple return if not enough data
            reward = portfolio_return

        # Calculate transaction cost penalty
        transaction_cost = 0.0
        if self.previous_positions is not None:
            position_changes = np.abs(positions - self.previous_positions)
            transaction_cost = np.sum(position_changes) * self.transaction_cost_penalty

        # Update previous positions
        self.previous_positions = positions.copy()

        # Final reward is risk-adjusted return minus penalties
        return reward - transaction_cost
