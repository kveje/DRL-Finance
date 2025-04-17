from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np

# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()



class BaseConstraint(ABC):
    """
    Base class for all trading constraints.
    Defines the interface that all constraint implementations must follow.
    """

    def __init__(self, name: str):
        """
        Initialize the constraint.

        Args:
            name: Name of the constraint for identification
        """
        self.name = name
        self.violated = False
        self.violation_message = ""
        self.violation_details = {}

    @abstractmethod
    def check(self, action: np.ndarray, current_positions: np.ndarray, **kwargs) -> bool:
        """
        Check if the action satisfies the constraint.

        Args:
            action: The action to check
            current_positions: The current positions of the portfolio
            **kwargs: Additional arguments needed for constraint checking

        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        pass

    def _set_violation(self, message: str, details: Dict[str, Any] = {}) -> None:
        """
        Set the violation state and message.

        Args:
            message: Description of the violation
        """
        self.violated = True
        self.violation_message = message
        self.violation_details = details

    def _clear_violation(self) -> None:
        """Clear the violation state and message."""
        self.violated = False
        self.violation_message = ""
        self.violation_details = {}

    def _log_violation(self, message: str) -> None:
        """Log the violation message."""
        logger.info(f"Constraint {self.name} violated: {message} with details {self.violation_details}")

    def is_violated(self) -> bool:
        """
        Check if the constraint is currently violated.

        Returns:
            bool: True if constraint is violated, False otherwise
        """
        return self.violated

    def get_violation_message(self) -> str:
        """
        Get the violation message.

        Returns:
            str: Message describing the violation
        """
        return self.violation_message 
    
    def get_violation_details(self) -> Dict[str, Any]:
        """
        Get the violation details.

        Returns:
            Dict[str, Any]: Dictionary containing violation details
        """
        return self.violation_details
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters of the constraint."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the constraint."""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the constraint."""
        pass
