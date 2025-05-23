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
        asset_list: List[str] = []
    ):
        """
        Initialize the OHLCV processor.
        
        Args:
            ohlcv_cols: List of OHLCV column names
            window_size: Size of the lookback window
            asset_list: List of asset tickers to process
        """
        self.ohlcv_cols = ohlcv_cols
        self.window_size = window_size
        self.asset_list = asset_list
        self.n_assets = len(self.asset_list)
        self.ohlcv_dim = len(self.ohlcv_cols)
    
    def process(self, data: pd.DataFrame, current_step: int) -> np.ndarray:
        """
        Process the OHLCV data into a matrix.
        
        Args:
            data: DataFrame containing the OHLCV data with day as index
            current_step: Current step in the environment
            
        Returns:
            np.ndarray: OHLCV data matrix of shape (n_assets, ohlcv_dim, window_size)
        """
        # Initialize the OHLCV data matrix
        ohlcv_data = np.zeros((self.n_assets, self.ohlcv_dim, self.window_size))
        
        # Get the window of data
        window = list(range(current_step - self.window_size, current_step))
        if window[0] < 0:
            raise ValueError(f"Window is negative. Current step: {current_step}, Window size: {self.window_size}")
        
        # For each asset, extract its data
        for i, asset in enumerate(self.asset_list):
            # Get data for this asset
            asset_data = data[data['ticker'] == asset].iloc[window]
            
            if len(asset_data) != self.window_size:
                raise ValueError(f"Not enough data points for asset {asset}. Expected {self.window_size}, got {len(asset_data)}")
            
            # Process OHLCV data
            for j, col in enumerate(self.ohlcv_cols):
                ohlcv_data[i, j, :] = asset_data[col].values
            
        return ohlcv_data
    
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