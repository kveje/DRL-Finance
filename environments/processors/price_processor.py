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
        asset_list: List[str] = [],
        raw_data_feature_indices: Dict[str, int] = None,
        processed_data_feature_indices: Dict[str, int] = None,
        tech_col_indices: Dict[str, int] = None
    ):
        """
        Initialize the price processor.
        
        Args:
            price_col: Name of the price column in the data
            window_size: Size of the lookback window
            asset_list: List of asset tickers to process
        """
        super().__init__(raw_data_feature_indices, processed_data_feature_indices, tech_col_indices)
        self.price_col = price_col
        self.window_size = window_size
        self.asset_list = asset_list
        self.n_assets = len(self.asset_list)
        self.day_col = day_col

    def process(self, processed_data: np.ndarray, current_step: int) -> np.ndarray:
        """
        Process the data into a price matrix.
        
        Args:
            raw_data (np.ndarray): Numpy array containing the raw data with shape (n_steps, n_assets, n_features)
            current_step (int): Current step in the environment
            
        Returns:
            np.ndarray: Price matrix of shape (n_assets, window_size)
        """
        data = processed_data[current_step - self.window_size:current_step, :, self.processed_data_feature_indices[self.price_col]] # shape: (window_size, n_assets)
        data = data.transpose(1, 0) # shape: (n_assets, window_size)

        return data
    
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