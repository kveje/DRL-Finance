import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward


class ReturnsBasedReward(BaseReward):
    """Reward function based on portfolio returns."""

    def __init__(self, transaction_cost_penalty: float = 0.1):
        """
        Initialize the returns-based reward function.

        Args:
            transaction_cost_penalty: Penalty factor for transaction costs
        """
        self.transaction_cost_penalty = transaction_cost_penalty
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
        Calculate reward based on portfolio returns and transaction costs.

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

        # Calculate transaction cost penalty
        transaction_cost = 0.0
        if self.previous_positions is not None:
            position_changes = np.abs(positions - self.previous_positions)
            transaction_cost = np.sum(position_changes) * self.transaction_cost_penalty

        # Update previous positions
        self.previous_positions = positions.copy()

        # Final reward is return minus transaction cost penalty
        return portfolio_return - transaction_cost
