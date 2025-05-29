from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from .base_processor import BaseProcessor

class VIXProcessor(BaseProcessor):
    """
    A processor that handles VIX index data.
    Since VIX is a market-wide index, it returns the same value for all assets.
    Returns a 1D array of VIX values of length window_size.
    """
    
    def __init__(
        self,
        vix_col: str = "vix",
        window_size: int = 10,
        raw_data_feature_indices: Dict[str, int] = None,
        processed_data_feature_indices: Dict[str, int] = None,
        tech_col_indices: Dict[str, int] = None
    ):
        """
        Initialize the VIX processor.
        
        Args:
            vix_col: Name of the VIX column in the data
            window_size: Size of the lookback window
        """
        super().__init__(raw_data_feature_indices, processed_data_feature_indices, tech_col_indices)
        self.vix_col = vix_col
        self.window_size = window_size

    def process(self, processed_data: np.ndarray, current_step: int) -> np.ndarray:
        """
        Process the VIX data into a 1D array.
        
        Args:
            processed_data (np.ndarray): Numpy array containing the processed data with shape (n_steps, n_assets, n_features)
            current_step (int): Current step in the environment
            
        Returns:
            np.ndarray: VIX data array of shape (window_size,)
        """
        # Get first asset
        # asset_data = data[data['ticker'] == data['ticker'].unique()[0]]
# 
        # # Get the window of data
        # window = list(range(current_step - self.window_size, current_step))
        # if window[0] < 0:
        #     raise ValueError(f"Window is negative. Current step: {current_step}, Window size: {self.window_size}")
        # 
        # # Get VIX values for the window
        # vix_values = asset_data.iloc[window][self.vix_col].values
        # 
        # # Convert to numpy array
        # return np.array(vix_values, dtype=np.float32)
        return processed_data[current_step:current_step+self.window_size, 0, self.processed_data_feature_indices[self.vix_col]]
    
    def get_observation_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the observation space for this processor.
        
        Returns:
            Dict containing the observation space specification
        """
        return {
            'vix': {
                'low': 0,  # VIX is always non-negative
                'high': np.inf,
                'shape': (self.window_size,),
                'dtype': np.float32
            }
        }
    
    def get_input_dim(self) -> Tuple[int, ...]:
        """
        Get the input dimension for the network.
        
        Returns:
            Tuple[int, ...]: Input dimension (window_size,)
        """
        return (self.window_size,)