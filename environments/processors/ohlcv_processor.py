from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from .base_processor import BaseProcessor

class OHLCVProcessor(BaseProcessor):
    """
    A processor that handles OHLCV data with a specified window size.
    """
    
    def __init__(
        self,
        ohlcv_cols: List[str],
        window_size: int = 10,
        asset_list: List[str] = [],
        raw_data_feature_indices: Dict[str, int] = None,
        processed_data_feature_indices: Dict[str, int] = None,
        tech_col_indices: Dict[str, int] = None
    ):
        """
        Initialize the OHLCV processor.
        
        Args:
            ohlcv_cols: List of OHLCV column names
            window_size: Size of the lookback window
            asset_list: List of asset tickers to process
        """
        super().__init__(raw_data_feature_indices, processed_data_feature_indices, tech_col_indices)
        self.ohlcv_cols = ohlcv_cols
        self.window_size = window_size
        self.asset_list = asset_list
        self.n_assets = len(self.asset_list)
        self.ohlcv_dim = len(self.ohlcv_cols)
        self.ohlcv_indices = [self.processed_data_feature_indices[col] for col in self.ohlcv_cols]

    def process(self, processed_data: np.ndarray, current_step: int) -> np.ndarray:
        """
        Process the OHLCV data into a matrix.
        
        Args:
            processed_data: Numpy array containing the processed data with shape (n_steps, n_assets, n_features)
            current_step: Current step in the environment
            
        Returns:
            np.ndarray: OHLCV data matrix of shape (n_assets, ohlcv_dim, window_size)
        """
        data = processed_data[current_step - self.window_size:current_step, :, self.ohlcv_indices] # shape: (window_size, n_assets, ohlcv_dim)
        return data.transpose(1, 0, 2) # shape: (n_assets, window_size, ohlcv_dim)
    
    def get_observation_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the observation space for this processor.
        
        Returns:
            Dict containing the observation space specification
        """
        return {
            'ohlcv': {
                'low': -np.inf,
                'high': np.inf,
                'shape': (self.n_assets, self.ohlcv_dim, self.window_size),
                'dtype': np.float32
            }
        }
    
    def get_input_dim(self) -> Tuple[int, ...]:
        """
        Get the input dimension for the network.
        
        Returns:
            Tuple[int, ...]: Input dimension (n_assets, ohlcv_dim, window_size)
        """
        return (self.n_assets, self.ohlcv_dim, self.window_size) 