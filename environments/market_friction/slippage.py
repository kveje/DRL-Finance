import numpy as np
from .base_friction import BaseMarketFriction

class SlippageFriction(BaseMarketFriction):
    """Handles slippage in trading actions."""
    
    def __init__(self, config: dict):
        """
        Initialize slippage friction.
        
        Args:
            config: Configuration dictionary containing slippage parameters
        """
        super().__init__(name="slippage")
        self.mean = config.get('slippage_mean', 0.0)  # mean of the normal distribution
        self.std = config.get('slippage_std', 0.001)  # standard deviation of the normal distribution
    
    def apply(self, action: np.ndarray, price: np.ndarray) -> np.ndarray:
        """
        Apply slippage to the trading quantities.
        Adjusts prices based on random slippage.
        
        Args:
            action: Numpy array of shape (n_assets,) representing quantities to trade
            price: Numpy array of shape (n_assets,) representing prices of the assets
            
        Returns:
            np.ndarray: Prices after applying slippage
        """
        # Mask for any action
        mask = action != 0

        # Find the number of non-zero actions
        n_actions = np.sum(mask)

        # If there are no actions, return the original prices
        if n_actions == 0:
            return price
        
        # Generate the slippage values (default is 1 - multiplying by 1 does nothing)
        slippage = np.ones_like(price)

        # Only generate random values for the non-zero actions
        random_values = np.random.normal(self.mean, self.std, size=n_actions)

        # Apply the random values to the non-zero actions
        slippage[mask] += random_values

        # Apply slippage to the prices
        return price * slippage
    
    def __str__(self) -> str:
        """Return a string representation of the slippage friction."""
        return f"SlippageFriction(mean={self.mean}, std={self.std})"
    
    def __repr__(self) -> str:
        """Return a string representation of the slippage friction."""
        return self.__str__()
