# environments/market_friction/manager.py
from typing import Dict, Type, Tuple
import numpy as np
from .base_friction import BaseMarketFriction
from .slippage import SlippageFriction
from .commission import CommissionFriction

class MarketFrictionManager:
    """Manages multiple market frictions."""

    FRICITON_TYPES = {
        "slippage": SlippageFriction,
        "commission": CommissionFriction
    }
    
    def __init__(self, config: dict):
        """
        Initialize the market friction manager.
        
        Args:
            config: Configuration dictionary containing friction parameters.
            
            example:
            {
                'slippage': {
                    'slippage_mean': 0.0,
                    'slippage_std': 0.001
                },
                'commission': {
                    'commission_rate': 0.001
                }
            }
        """
        self.config = config
        self.frictions: Dict[str, BaseMarketFriction] = {}
        self._setup_frictions()
    
    def _setup_frictions(self):
        """Setup individual friction components."""
        # Define available friction types
        enabled_frictions = self.config.keys()
        
        # Initialize enabled frictions
        for friction_name in enabled_frictions:
            if friction_name in self.FRICITON_TYPES:
                friction_config = self.config.get(friction_name, {})
                self.frictions[friction_name] = self.FRICITON_TYPES[friction_name](friction_config)
            else:
                raise ValueError(f"Invalid friction type: {friction_name}")
    
    def apply_frictions(self, action: np.ndarray, price: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all enabled market frictions to the action.
        
        Args:
            action: Numpy array of shape (n_assets,) representing quantities to trade
            price: Numpy array of shape (n_assets,) representing prices of the assets   
        Returns:
            Tuple[np.ndarray, np.ndarray]: Prices after applying all frictions and the cost of the frictions (commission, slippage)
        """
        new_price = np.copy(price)
        
        # Apply each friction in sequence
        for friction in self.frictions.values():
            new_price = friction.apply(action, new_price)
        
        return new_price, price - new_price 
    
    def add_friction(self, name: str, friction: BaseMarketFriction):
        """Add a new friction to the manager."""
        self.frictions[name] = friction

    def __str__(self) -> str:
        """Return a string representation of the market friction manager."""
        str = "MarketFrictionManager with \n"
        for friction in self.frictions.values():
            str += f"- {repr(friction)} \n"
        str = str[:-2]
        return str
    
    def __repr__(self) -> str:
        """Return a string representation of the market friction manager."""
        return self.__str__()

    