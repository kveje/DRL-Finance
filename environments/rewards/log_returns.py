import numpy as np
from typing import Dict, Any
from .base_reward import BaseReward

class LogReturnsReward(BaseReward):
    """Reward function based on log returns."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the log returns reward.
        
        Args:
            config: Configuration dictionary containing reward parameters.
                parameters:
                    scale: Scale factor for the reward
                
                Example:
                {"scale": 10}
        """
        super().__init__(name="log_returns")
        self.scale = config.get("scale", 10)

    def calculate(self, portfolio_value: float, previous_portfolio_value: float, **kwargs) -> float:
        """Calculate the log returns reward.
        
        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Previous portfolio value
            **kwargs: Additional arguments

        Returns:
            float: The calculated reward
        """
        return (np.log(portfolio_value/previous_portfolio_value)) * self.scale

    def reset(self):
        """Reset the reward function."""
        pass
    
    def __str__(self) -> str:
        """Return a string representation of the reward function."""
        return f"LogReturnsReward(scale={self.scale})"
    
    def __repr__(self) -> str:
        """Return a string representation of the reward function."""
        return self.__str__()

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters of the reward function."""
        return {"scale": self.scale}
