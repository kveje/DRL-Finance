"""Reward class for handling constraint violation penalties."""

import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward

class ConstraintViolationReward(BaseReward):
    """Reward class that penalizes constraint violations."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the constraint violation reward.

        Args:
            config: Dictionary containing configuration parameters
                   - penalty_scale: float, scaling factor for the penalty (default: 0.001)
        """
        self.config = config or {}
        self.scale = self.config.get("scale", 0.01)

    def calculate(
        self,
        intended_action: np.ndarray,
        feasible_action: np.ndarray,
        **kwargs
    ) -> float:
        """
        Calculate the penalty for constraint violations.

        Args:
            intended_action: The original action proposed by the agent
            feasible_action: The action after applying constraints

        Returns:
            float: The penalty value (negative reward)
        """
        # Calculate the absolute difference between intended and feasible actions
        action_diff = np.abs(intended_action - feasible_action)
        
        # Sum up the differences and apply the penalty scale
        penalty = np.sum(action_diff) * self.scale
        
        return -penalty  # Return negative value as it's a penalty

    def get_parameters(self) -> Dict[str, Any]:
        """Get the reward parameters."""
        return {
            "scale": self.scale
        } 
    
    def reset(self):
        """Reset the reward function."""
        pass

    def __str__(self) -> str:
        """Return a string representation of the reward function."""
        return f"ConstraintViolationReward(scale={self.scale})"
    
    def __repr__(self) -> str:
        return self.__str__()