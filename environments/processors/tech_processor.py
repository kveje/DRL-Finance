from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from .base_processor import BaseProcessor

class TechProcessor(BaseProcessor):
    """
    A processor that handles technical indicators at the current time point.
    Returns a 2D array of shape (n_assets, n_technical_indicators).
    """
    
    def __init__(
        self,
        tech_cols: List[str],
        asset_list: List[str] = [],
        raw_data_feature_indices: Dict[str, int] = None,
        processed_data_feature_indices: Dict[str, int] = None,
        tech_col_indices: List[int] = None,
    ):
        """
        Initialize the technical indicators processor.
        
        Args:
            tech_cols: List of technical indicator column names
            asset_list: List of asset tickers to process
        """
        super().__init__(raw_data_feature_indices, processed_data_feature_indices, tech_col_indices)
        self.tech_cols = tech_cols
        self.asset_list = asset_list
        self.n_assets = len(self.asset_list)
        self.tech_dim = len(self.tech_cols)

    def process(self, processed_data: np.ndarray, current_step: int) -> np.ndarray:
        """
        Process the technical indicators into a 2D array.
        Only takes the current time point's technical indicators.
        
        Args:
            processed_data (np.ndarray): Numpy array containing the processed data with shape (n_steps, n_assets, n_features)
            current_step (int): Current step in the environment
            
        Returns:
            np.ndarray: Technical indicators matrix of shape (n_assets, tech_dim)
        """
        return processed_data[current_step, :, self.tech_col_indices]
    
    def get_observation_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the observation space for this processor.
        
        Returns:
            Dict containing the observation space specification
        """
        return {
            'tech': {
                'low': -np.inf,
                'high': np.inf,
                'shape': (self.n_assets, self.tech_dim),
                'dtype': np.float32
            }
        }
    
    def get_input_dim(self) -> Tuple[int, ...]:
        """
        Get the input dimension for the network.
        
        Returns:
            Tuple[int, ...]: Input dimension (n_assets, tech_dim)
        """
        return (self.n_assets, self.tech_dim) 