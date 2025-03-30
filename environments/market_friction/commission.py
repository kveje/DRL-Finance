import numpy as np
from .base_friction import BaseMarketFriction


class CommissionFriction(BaseMarketFriction):
    """Handles commission in trading actions."""
    
    def __init__(self, config: dict):
        """
        Initialize commission friction.
        
        Args:
            config: Configuration dictionary containing commission parameters
        """
        super().__init__(name="commission")
        self.rate = config.get('commission_rate', 0.001)  # 0.1% default
    
    def apply(self, action: np.ndarray, price: np.ndarray) -> np.ndarray:
        """
        Apply commission to trade quantities.
        For quantity-based actions, commission is handled directly in the trading execution,
        so this method simply returns the original action without modification.
        
        Args:
            action: Numpy array of shape (n_assets,) representing quantities to trade
            price: Numpy array of shape (n_assets,) representing prices of the assets
        Returns:
            np.ndarray: New prices after applying commission
        """
        # Mask for actions
        buy_mask = action > 0 # Buy actions
        sell_mask = action < 0 # Sell actions

        # Create a multiplier for the prices
        multiplier = np.ones_like(price)
        multiplier[buy_mask] = 1 + self.rate
        multiplier[sell_mask] = 1 - self.rate

        return price * multiplier
    
    def __str__(self) -> str:
        """Return a string representation of the commission friction."""
        return f"CommissionFriction(rate={self.rate})"
    
    def __repr__(self) -> str:
        """Return a string representation of the commission friction."""
        return self.__str__()
