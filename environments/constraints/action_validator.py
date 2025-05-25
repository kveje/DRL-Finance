from typing import Dict, Any, Tuple
import numpy as np
from .manager import ConstraintManager

class ActionValidator:
    """
    Validates and adjusts trading actions to comply with various constraints.
    This class handles position limits, cash limits, and other trading constraints.
    """

    def __init__(self, constraint_manager: ConstraintManager):
        """
        Initialize the action validator.

        Args:
            constraint_manager: The constraint manager instance that holds all constraints
        """
        self.constraint_manager = constraint_manager

    def validate_and_adjust_action(
        self,
        intended_action: np.ndarray,
        current_positions: np.ndarray,
        current_cash: float,
        adjusted_prices: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validates and adjusts the intended action to comply with all constraints.

        Args:
            intended_action: Desired change in shares for each asset
            current_positions: Current shares held for each asset
            current_cash: Current cash balance
            adjusted_prices: Prices after accounting for slippage

        Returns:
            Tuple of (feasible_action, violation_info)
            - feasible_action: The adjusted action that complies with all constraints
            - violation_info: Dictionary containing information about any constraint violations
        """
        feasible_action = intended_action.copy()
        violation_info = {}

        # Apply each constraint in sequence
        for constraint_name, constraint in self.constraint_manager.constraints.items():
            # Get adjusted action and violation info from this constraint
            feasible_action, constraint_violations = constraint.validate_and_adjust_action(
                action=feasible_action,
                current_positions=current_positions,
                current_cash=current_cash,
                current_prices=adjusted_prices
            )

            # If there were violations, add them to the violation info
            if constraint_violations:
                violation_info[constraint_name] = constraint_violations

        return feasible_action.astype(int), violation_info

    def get_constraint_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all constraint parameters.

        Returns:
            Dictionary containing all constraint parameters
        """
        return {
            name: constraint.get_parameters()
            for name, constraint in self.constraint_manager.constraints.items()
        } 