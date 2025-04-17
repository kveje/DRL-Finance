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
        super().__init__(name)
        self.min_cash = config["min"]
        self.max_cash = config["max"]
        self.cash = 0

    def check(self, action: np.ndarray, current_positions: np.ndarray, current_cash: float, current_prices: np.ndarray) -> bool:
        """
        Check if the cash is within the limits.
        """
        # Calculate the cash after the action
        self.cash = current_cash + np.sum(-1 * action * current_prices) # -1 because we are buying shares
        # Check if the cash is within the limits
        if self.cash < self.min_cash:
            self._set_violation(f"Cash {self.cash:.2f} is below the minimum limit {self.min_cash:.2f}",
                                {"constraint": self.name, "violation_details": {"cash": self.cash, "min_cash": self.min_cash}})
            return False
        elif self.cash > self.max_cash:
            self._set_violation(f"Cash {self.cash:.2f} is above the maximum limit {self.max_cash:.2f}",
                                {"constraint": self.name, "violation_details": {"cash": self.cash, "max_cash": self.max_cash}})
            return False
        
        return True
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters of the constraint."""
        return {"min": self.min_cash, "max": self.max_cash}
    
    def __str__(self) -> str:
        return f"CashLimit(min={self.min_cash}, max={self.max_cash})"
    
    def __repr__(self) -> str:
        return self.__str__()
