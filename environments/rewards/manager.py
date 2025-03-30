from .returns_based import ReturnsBasedReward
from .sharpe_based import SharpeBasedReward
from .base_reward import BaseReward

from typing import Tuple, Dict, Any

class RewardManager:
    """Manager for rewards."""

    REWARD_TYPES = {
        "returns_based": ReturnsBasedReward,
        "sharpe_based": SharpeBasedReward
    }

    def __init__(self, config: Tuple[str, Dict[str, Any]]):
        """
        Initialize the reward manager.

        Args:
            config: Configuration dictionary containing reward parameters.
            key: reward type

            example:
            ("returns_based", {"scale": 1.0})
        """
        # Get the reward type and config
        self.reward_type = config[0]
        self.reward_config = config[1]

        # Check if the reward type is valid
        if self.reward_type not in self.REWARD_TYPES:
            raise ValueError(f"Invalid reward type: {self.reward_type}")

        # Initialize the reward function
        self.reward_function: BaseReward = self.REWARD_TYPES[self.reward_type](self.reward_config)
        
    def calculate(self, **kwargs) -> float:
        """Calculate the reward."""
        return self.reward_function.calculate(**kwargs)
    
    def reset(self):
        """Reset the reward function."""
        self.reward_function.reset()
    
    def __str__(self) -> str:
        """Return a string representation of the reward manager."""
        return f"RewardManager(reward_function={self.reward_function})"
    
    def __repr__(self) -> str:
        """Return a string representation of the reward manager."""
        return self.__str__()

