import numpy as np
from typing import Dict, Any, Tuple
from .base_processor import BaseProcessor
import pandas as pd


class CurrentPriceProcessor(BaseProcessor):
    """
    Processor for current price information.
    Provides normalized current price information as features.
    """
    
    
    def __init__(self, min_cash_limit: float = 0, price_col: str = "close", transaction_cost: float = 0.001, slippage_mean: float = 0.0001, n_assets: int = 1):
        """
        Initialize the current price processor.
        
        Args:
            min_cash_limit (float): Minimum cash limit
            price_col (str): Column name for the price
            transaction_cost (float): Transaction cost
            slippage_mean (float): Mean slippage
            n_assets (int): Number of assets
        """
        super().__init__()
        self.price_col = price_col
        self.min_cash_limit = min_cash_limit
        self.transaction_cost = transaction_cost
        self.slippage_mean = slippage_mean
        self.n_assets = n_assets

    def process(self, raw_data: pd.DataFrame, current_cash: float, step: int) -> np.ndarray:
        """
        Process the current price information.
        """
        # Get the current price
        current_price = np.array(raw_data[self.price_col].loc[step], dtype=np.float32)

        # Calculate the adjusted price
        adjusted_price = current_price * (1 + self.transaction_cost + self.slippage_mean)

        # Calculate the available cash
        available_cash = current_cash - self.min_cash_limit

        # Calculate the normalized price
        normalized_price = adjusted_price / (available_cash + 1e-6)

        # Clip the price to be between 0 and 1 (If cash is close to 0, the normalized price will explode)
        # 1 = one share cost 100% of available cash
        normalized_price = np.clip(normalized_price, 0, 1)

        return normalized_price

    def get_observation_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            'current_price': {
                'low': 0,
                'high': 1,
                'shape': (self.n_assets,),
                'dtype': np.float32
            }
        }
    
    def get_input_dim(self) -> Tuple[int, ...]:
        return (self.n_assets,)

