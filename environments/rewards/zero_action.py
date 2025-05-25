"""Reward class for penalizing zero actions over time."""

import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward

class ZeroActionReward(BaseReward):
    """Reward class that penalizes zero actions over time."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the zero action reward.

        Args:
            config: Dictionary containing configuration parameters
                   - scale: float, scaling factor for the penalty (default: 0.01)
                   - window_size: int, number of steps to look back for zero actions (default: 10)
                   - min_consecutive_days: int, minimum number of consecutive non-clipped zero actions before applying penalty (default: 3)
        """
        super().__init__(name="zero_action")
        self.config = config or {}
        self.scale = self.config.get("scale", 0.01)
        self.window_size = self.config.get("window_size", 10)
        self.min_consecutive_days = self.config.get("min_consecutive_days", 3)
        self.not_clipped_zero_action_history = []

    def calculate(self, intended_action: np.ndarray, feasible_action: np.ndarray) -> float:
        """
        Calculate the penalty for zero actions.

        Args:
            intended_action: The intended action taken by the agent
            feasible_action: The feasible action taken by the agent

        Returns:
            float: The penalty value (negative reward)
        """
        # Check if current action is zero
        is_zero_action = np.all(np.abs(feasible_action) < 1e-6)
        action_was_clipped = np.any(np.abs(intended_action - feasible_action) > 1e-6)
        
        # Add to history
        self.not_clipped_zero_action_history.append(is_zero_action and not action_was_clipped)
        
        # Keep only the last window_size entries
        if len(self.not_clipped_zero_action_history) > self.window_size:
            self.not_clipped_zero_action_history.pop(0)
        
        # Calculate penalty based on consecutive zero actions
        consecutive_zeros = 0
        for is_zero in reversed(self.not_clipped_zero_action_history):
            if is_zero:
                consecutive_zeros += 1
            else:
                break
        
        # Apply penalty if we have enough consecutive zero actions
        if consecutive_zeros >= self.min_consecutive_days:
            penalty = consecutive_zeros * self.scale
            return -penalty
        
        return 0.0

    def reset(self):
        """Reset the zero action history."""
        self.not_clipped_zero_action_history = []

    def get_parameters(self) -> Dict[str, Any]:
        """Get the reward parameters."""
        return {
            "scale": self.scale,
            "window_size": self.window_size,
            "min_consecutive_days": self.min_consecutive_days
        }
    
    def __str__(self) -> str:
        return f"ZeroActionReward(scale={self.scale}, window_size={self.window_size}, min_consecutive_days={self.min_consecutive_days})"
    
    def __repr__(self) -> str:
        return f"ZeroActionReward(scale={self.scale}, window_size={self.window_size}, min_consecutive_days={self.min_consecutive_days})"
    
    
