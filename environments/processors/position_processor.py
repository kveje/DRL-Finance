from typing import Dict, Any, List, Tuple
import numpy as np
from .base_processor import BaseProcessor

class PositionProcessor(BaseProcessor):
    """
    A processor that normalizes positions relative to their limits.
    Returns a value between 0 and 1, where:
    - 0 means the position is at its minimum limit
    - 1 means the position is at its maximum limit
    - 0.5 means the position is halfway between min and max
    """
    
    def __init__(
        self,
        position_limits: Dict[str, float],  # {'min': float, 'max': float}
        asset_list: List[str] = [],
        raw_data_feature_indices: Dict[str, int] = None,
        processed_data_feature_indices: Dict[str, int] = None,
        tech_col_indices: Dict[str, int] = None
    ):
        """
        Initialize the position processor.
        
        Args:
            position_limits: Dictionary containing min and max position limits (in number of shares)
            asset_list: List of asset tickers to process
        """
        super().__init__(raw_data_feature_indices, processed_data_feature_indices, tech_col_indices)
        self.position_limits = position_limits
        self.asset_list = asset_list
        self.n_assets = len(self.asset_list)
    
    def process(self, positions: np.ndarray) -> np.ndarray:
        """
        Process the positions into a normalized vector.
        
        Args:
            positions: Current positions array (number of shares)
            
        Returns:
            np.ndarray: Normalized position vector of shape (n_assets,)
            Values are between 0 and 1, indicating position relative to limits
        """
        # Normalize positions to [0, 1] range using the same formula for all positions
        # (position - min_limit) / (max_limit - min_limit)
        normalized_positions = (
            (positions - self.position_limits['min']) / 
            (self.position_limits['max'] - self.position_limits['min'])
        )
        
        # Clip to ensure values are between 0 and 1
        normalized_positions = np.clip(normalized_positions, 0, 1)
        
        return normalized_positions
    
    def get_observation_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the observation space for this processor.
        
        Returns:
            Dict containing the observation space specification
        """
        return {
            'position': {
                'low': 0,
                'high': 1,
                'shape': (self.n_assets,),
                'dtype': np.float32
            }
        }
    
    def get_input_dim(self) -> Tuple[int, ...]:
        """
        Get the input dimension for the network.
        
        Returns:
            Tuple[int, ...]: Input dimension (n_assets,)
        """
        return (self.n_assets,) 