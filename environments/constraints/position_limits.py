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
            limits: Tuple of two floats/ints representing the minimum and maximum position limits.
            name: Name of the constraint
        """
        super().__init__(name)
        self.min_position = config["min"]
        self.max_position = config["max"]

    def check(
        self,
        action: np.ndarray,
        current_positions: np.ndarray,
        current_cash: float,
        current_prices: np.ndarray
    ) -> bool:
        """
        Check if the action satisfies position limits.
        
        For quantity-based actions, we need to check if the resulting position
        (current position + action) satisfies the position limits.

        Args:
            action: Array of quantities to trade (positive for buy, negative for sell)
            current_positions: Current positions in shares (required for quantity-based actions)
            current_cash: Current cash
            current_prices: Current prices

        Returns:
            bool: True if all position limits are satisfied
        """
        self._clear_violation()
        
        # Calculate resulting positions after executing the action
        resulting_positions = current_positions + action
        
        for asset_index, position in enumerate(resulting_positions):
            if position < self.min_position:
                self._set_violation(
                    f"Resulting position {position:.2f} for asset {asset_index} below minimum limit {self.min_position:.2f}",
                    {"constraint": self.name, "violation_details": {"position": position, "min_position": self.min_position}}
                )
                return False
            elif position > self.max_position:
                self._set_violation(
                    f"Resulting position {position:.2f} for asset {asset_index} above maximum limit {self.max_position:.2f}",
                    {"constraint": self.name, "violation_details": {"position": position, "max_position": self.max_position}}
                )
                return False

        return True

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

    def __str__(self) -> str:
        """Return a string representation of the position limits constraint."""
        return f"PositionLimits(limits={self.min_position}, {self.max_position})"

    def __repr__(self) -> str:
        """Return a string representation of the position limits constraint."""
        return self.__str__()
