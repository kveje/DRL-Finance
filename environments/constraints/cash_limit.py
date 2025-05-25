from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
from .base_constraint import BaseConstraint

# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()




class CashLimit(BaseConstraint):
    """
    Constraint that enforces a cash limit.
    """

    def __init__(self, config: Dict[str, Any], name: str = "cash_limit"):
        """
        Initialize cash limit constraint.

        Args:
            config: Dictionary containing min and max cash limits
            name: Name of the constraint
        """
        super().__init__(name)
        self.min_cash = config["min"]
        self.max_cash = config["max"]

    def validate_and_adjust_action(
        self,
        action: np.ndarray,
        current_positions: np.ndarray,
        current_cash: float,
        current_prices: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validate and adjust the action to comply with cash limits.

        Args:
            action: Array of quantities to trade (positive for buy, negative for sell)
            current_positions: Current positions in shares
            current_cash: Current cash balance
            current_prices: Current prices

        Returns:
            Tuple of (adjusted_action, violation_info)
        """
        self._clear_violation()
        feasible_action = action.copy()
        violation_info = {}

        # Calculate net cash flow required
        net_cash_flow_required = np.sum(feasible_action * current_prices)
        available_cash_for_trade = current_cash - self.min_cash

        if net_cash_flow_required > available_cash_for_trade:
            violation_info = {
                "required_cash": net_cash_flow_required,
                "available_cash": available_cash_for_trade,
                "min_cash_limit": self.min_cash
            }
            self._set_violation(
                f"Cash limit violated: {violation_info}",
                violation_info
            )

            # Scale down buys to fit within cash constraints
            scaling_factor = available_cash_for_trade / (net_cash_flow_required + 1e-9)
            buy_mask = feasible_action > 0
            feasible_action[buy_mask] = np.floor(feasible_action[buy_mask] * scaling_factor)

        return feasible_action, violation_info

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters of the constraint."""
        return {"min": self.min_cash, "max": self.max_cash}

    def __str__(self) -> str:
        """Return a string representation of the cash limit constraint."""
        return f"CashLimit(min={self.min_cash}, max={self.max_cash})"

    def __repr__(self) -> str:
        """Return a string representation of the cash limit constraint."""
        return self.__str__()
