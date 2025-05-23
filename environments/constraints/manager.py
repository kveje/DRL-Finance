"""Manager for constraints."""

from typing import Dict
from .base_constraint import BaseConstraint
from .position_limits import PositionLimits
from .cash_limit import CashLimit
import numpy as np

class ConstraintManager:
    """Manager for constraints."""

    CONSTRAINT_TYPES = {
        "position_limits": PositionLimits,
        "cash_limit": CashLimit
    }

    def __init__(self, config: dict):
        """Initialize the constraint manager.
        
        Args:
            config: Configuration dictionary containing constraints.
        """
        self.constraints: Dict[str, BaseConstraint] = {}
        self._setup_constraints(config)

    def _setup_constraints(self, config: dict) -> None:
        """Setup the constraints."""
        # Define available constraint types
        enabled_constraints = config.keys()
        
        # Initialize enabled constraints
        for constraint_name in enabled_constraints:
            if constraint_name in self.CONSTRAINT_TYPES:
                constraint_config = config.get(constraint_name, {})
                self.constraints[constraint_name] = self.CONSTRAINT_TYPES[constraint_name](constraint_config)
            else:
                raise ValueError(f"Invalid constraint type: {constraint_name}")

    def check_constraints(self, action: np.ndarray, current_positions: np.ndarray, current_cash: float, current_prices: np.ndarray) -> bool:
        """
        Check if the action satisfies all constraints.
        
        Args:
            action: Array of quantities to trade
            current_positions: Current positions in shares
            current_cash: Current cash
            current_prices: Current prices
            
        Returns:
            bool: True if all constraints are satisfied
        """
        for constraint in self.constraints.values():
            if not constraint.check(
                action = action,
                current_positions = current_positions,
                current_cash = current_cash,
                current_prices = current_prices
            ):
                return False
        return True
    
    def get_parameters(self, constraint_name: str) -> Dict[str, float]:
        """Get the parameters of a constraint."""
        if constraint_name not in self.constraints:
            raise ValueError(f"Constraint {constraint_name} not found in constraints")
        return self.constraints[constraint_name].get_parameters()

    def __str__(self) -> str:
        """Return a string representation of the constraint manager."""
        str = "ConstraintManager with \n"
        for constraint in self.constraints.values():
            str += f"- {print(constraint)} \n"

        str = str[:-2]
        return str

    def __repr__(self) -> str:
        """Return a string representation of the constraint manager."""
        return self.__str__()
