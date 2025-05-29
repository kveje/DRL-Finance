import numpy as np
from typing import Dict, Any, Tuple
from .base_processor import BaseProcessor
import pandas as pd

class AffordabilityProcessor(BaseProcessor):
    """
    Processor for affordability information.
    Provides normalized affordability information as features.
    """
    
    def __init__(self, n_assets: int, 
                 min_cash_limit: float = 0.0, 
                 max_trade_size: int = 10, 
                 price_col: str = "close", 
                 transaction_cost: float = 0.001, 
                 slippage_mean: float = 0.0, 
                 raw_data_feature_indices: Dict[str, int] = None, 
                 processed_data_feature_indices: Dict[str, int] = None,
                 tech_col_indices: Dict[str, int] = None):
        """
        Initialize the affordability processor.
        
        Args:
            min_cash_limit (float): Minimum allowed cash balance
            max_trade_size (float): Maximum allowed trade size
            price_col (str): Column name for the price
            transaction_cost (float): Transaction cost
            slippage_mean (float): Mean slippage
            n_assets (int): Number of assets
        """
        super().__init__(raw_data_feature_indices, processed_data_feature_indices, tech_col_indices)
        self.min_cash_limit = min_cash_limit
        self.max_trade_size = max_trade_size
        self.price_col = price_col
        self.transaction_cost = transaction_cost
        self.slippage_mean = slippage_mean
        self.n_assets = n_assets

    def process(self, raw_data: np.ndarray, current_cash: float, step: int) -> np.ndarray:
        """
        Process the affordability information.
        
        Args:
            raw_data (np.ndarray): Numpy array containing the raw data with shape (n_steps, n_assets, n_features)
            current_cash (float): Current cash balance
            step (int): Current step
            
        Returns:
            np.ndarray: Affordability information
        """
        # Get the current prices
        current_prices = raw_data[step, :, self.raw_data_feature_indices[self.price_col]]
        adjusted_prices = current_prices * (1 + self.transaction_cost + self.slippage_mean)

        # Calculate the available cash
        available_cash = current_cash - self.min_cash_limit

        # Calculate the max shares we can afford to buy
        max_affordable_shares = np.floor(available_cash / (adjusted_prices + 1e-6))

        # Calculate the affordability
        max_shares = np.minimum(max_affordable_shares, self.max_trade_size)

        # Normalize the affordability
        affordability = np.clip(max_shares / self.max_trade_size, 0, 1)

        return affordability

    def get_observation_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            'affordability': {
                'low': 0,
                'high': 1,
                'shape': (self.n_assets,),
                'dtype': np.float32
            }
        }

    def get_input_dim(self) -> Tuple[int, ...]:
        return (self.n_assets,)

