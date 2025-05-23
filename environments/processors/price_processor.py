from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from .base_processor import BaseProcessor

class PriceProcessor(BaseProcessor):
    """
    A processor that only processes price data into a n_assets x window_size matrix.
    Specifically designed for the trading environment.
    """
    
    def __init__(
        self,
        price_col: str = "close",
        day_col: str = "day",
        window_size: int = 10,
        asset_list: List[str] = []
    ):
        """
        Initialize the price processor.
        
        Args:
            price_col: Name of the price column in the data
            window_size: Size of the lookback window
            asset_list: List of asset tickers to process
        """
        self.price_col = price_col
        self.window_size = window_size
        self.asset_list = asset_list
        self.n_assets = len(self.asset_list)
        self.day_col = day_col
    
    def process(self, data: pd.DataFrame, current_step: int) -> np.ndarray:
        """
        Process the data into a price matrix.
        
        Args:
            data: DataFrame containing the price data with day as index
            current_step: Current step in the environment
            
        Returns:
            np.ndarray: Price matrix of shape (n_assets, window_size)
        """
        # Initialize the price matrix
        price_matrix = np.zeros((self.n_assets, self.window_size))
        
        # Get the window of data
        window = list(range(current_step - self.window_size, current_step))
        if window[0] < 0:
            raise ValueError(f"Window is negative. Current step: {current_step}, Window size: {self.window_size}")

        # For each asset, extract its price data
        for i, asset in enumerate(self.asset_list):
            asset_data = data[data['ticker'] == asset].iloc[window][self.price_col]
            price_matrix[i, :] = asset_data.values
            
        return price_matrix
    
    def get_observation_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the observation space for this processor.
        
        Returns:
            Dict containing the observation space specification
        """
        return {
            'price': {
                'low': -np.inf,
                'high': np.inf,
                'shape': (self.n_assets, self.window_size),
                'dtype': np.float32
            }
        }
    
    def get_input_dim(self) -> Tuple[int, ...]:
        """
        Get the input dimension for the network.
        
        Returns:
            Tuple[int, ...]: Input dimension (n_assets, window_size)
        """
        return (self.n_assets, self.window_size) 