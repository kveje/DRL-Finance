import numpy as np
from typing import Dict, Any, Tuple
from .base_processor import BaseProcessor

class CashProcessor(BaseProcessor):
    """
    Processor for cash balance information.
    Provides normalized cash balance and remaining available cash as features.
    """
    
    def __init__(self, cash_limit: float, 
                 raw_data_feature_indices: Dict[str, int], 
                 processed_data_feature_indices: Dict[str, int],
                 tech_col_indices: Dict[str, int]):
        """
        Initialize the cash processor.
        
        Args:
            cash_limit (float): Maximum allowed cash balance
        """
        super().__init__(raw_data_feature_indices, processed_data_feature_indices, tech_col_indices)
        self.cash_limit = cash_limit
        
    def process(self, cash_balance: float) -> np.ndarray:
        """
        Process cash balance information into normalized features.
        
        Args:
            cash_balance (float): Current cash balance
            
        Returns:
            np.ndarray: Array of shape (2,) containing:
                - Normalized current cash balance (0 to 1)
                - Normalized remaining available cash (0 to 1)
        """
        # Normalize current cash balance
        normalized_balance = min(cash_balance / self.cash_limit, 1.0)
        
        # Calculate and normalize remaining available cash
        remaining_cash = max(0, self.cash_limit - cash_balance)
        normalized_remaining = remaining_cash / self.cash_limit
        
        return np.array([normalized_balance, normalized_remaining], dtype=np.float32)
    
    def get_observation_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the observation space for this processor.
        
        Returns:
            Dict containing the observation space specification
        """
        return {
            'cash': {
                'low': 0.0,
                'high': 1.0,
                'shape': (2,),
                'dtype': np.float32
            }
        }
    
    def get_input_dim(self) -> Tuple[int, ...]:
        """
        Get the input dimension for the network.
        
        Returns:
            tuple[int, ...]: Input dimensions (2,)
        """
        return (2,) 