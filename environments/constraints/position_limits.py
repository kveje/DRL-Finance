from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
from .base_constraint import BaseConstraint

# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()




class PositionLimits(BaseConstraint):
    """
    Constraint that enforces position limits for each asset.
    Position limits can be specified as absolute number of shares or as cash value.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        name: str = "position_limits",
    ):
        """
        Initialize position limits constraint.

        Args:
            config: Dictionary containing min and max position limits
            name: Name of the constraint
        """
        super().__init__(name)
        self.min_position = config["min"]
        self.max_position = config["max"]

    def validate_and_adjust_action(
        self,
        action: np.ndarray,
        current_positions: np.ndarray,
        current_cash: float,
        current_prices: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validate and adjust the action to comply with position limits.
        
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
        
        # Calculate resulting positions after executing the action
        resulting_positions = current_positions + feasible_action
        
        # Check for violations
        min_violations = resulting_positions < self.min_position
        max_violations = resulting_positions > self.max_position
        
        if np.any(min_violations) or np.any(max_violations):
            violation_info = {
                "min_violations": np.where(min_violations)[0].tolist(),
                "max_violations": np.where(max_violations)[0].tolist(),
                "min_limit": self.min_position,
                "max_limit": self.max_position
            }
            self._set_violation(
                f"Position limits violated: {violation_info}",
                violation_info
            )
        
        # Clip to valid ranges
        feasible_action = np.clip(
            feasible_action,
            self.min_position - current_positions,
            self.max_position - current_positions
        )
        
        return feasible_action, violation_info

    def update_limits(
        self, new_limits: Tuple[Union[float, int], Union[float, int]]
    ) -> None:
        """
        Update the position limits.

        Args:
            new_limits: New position limits dictionary
        """
        self.min_position = new_limits[0]
        self.max_position = new_limits[1]

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters of the constraint."""
        return {"min": self.min_position, "max": self.max_position}

    def __str__(self) -> str:
        """Return a string representation of the position limits constraint."""
        return f"PositionLimits(min={self.min_position}, max={self.max_position})"

    def __repr__(self) -> str:
        """Return a string representation of the position limits constraint."""
        return self.__str__()
