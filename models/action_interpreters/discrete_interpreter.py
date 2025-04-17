import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

from environments.trading_env import TradingEnv

class DiscreteInterpreter:
    """
    Interprets continuous Q-values as discrete actions.
    
    Maps normalized values (-1 to 1) to discrete actions:
    - Values between -1 and -0.2: Sell (with intensity proportional to value)
    - Values between -0.2 and 0.2: Hold
    - Values between 0.2 and 1: Buy (with intensity proportional to value)
    """
    
    def __init__(
        self,
        env: TradingEnv,
        sell_threshold: float = -0.2,
        buy_threshold: float = 0.2,
    ):
        """
        Initialize the discrete action interpreter.
        
        Args:
            env: Trading environment instance
            sell_threshold: Value below which action is interpreted as sell
            buy_threshold: Value above which action is interpreted as buy
            min_position: Minimum position limit (if None, use environment default)
            max_position: Maximum position limit (if None, use environment default)
        """
        self.env = env
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold
        
        # Get position limits from environment if not specified
        position_limits = env.get_position_limits()
        self.min_position = position_limits["min"]
        self.max_position = position_limits["max"]
        
        # Calculate position range for scaling
        self.position_range = self.max_position - self.min_position
        
        # Log initialization
        print(f"Initialized DiscreteInterpreter with thresholds: sell={sell_threshold}, buy={buy_threshold}")
        print(f"Position limits: min={self.min_position}, max={self.max_position}")
    
    def interpret(
        self,
        q_values: np.ndarray,
        current_position: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Interpret Q-values as discrete actions (number of shares to buy/sell).
        
        Args:
            q_values: a numpy array of normalized values (-1 to 1)
            current_position: Current position of the agent
            deterministic: Whether to use deterministic interpretation
            
        Returns:
            Action array representing number of shares to buy (positive) or sell (negative) for each asset
        """        
        # Create masks (without using intent)
        sell_mask = q_values < self.sell_threshold
        buy_mask = q_values > self.buy_threshold

        # Step 1: Normalize q_values to [0, 1]
        normalized_q_values = np.zeros_like(q_values)
        normalized_q_values[buy_mask] = (q_values[buy_mask] - self.buy_threshold) / (1 - self.buy_threshold)
        normalized_q_values[sell_mask] = (q_values[sell_mask] - self.sell_threshold) / (1 + self.sell_threshold)

        # Step 2: Scale to difference between current position and position limits
        actions = np.zeros_like(current_position)
        actions[buy_mask] = normalized_q_values[buy_mask] * (self.max_position - current_position[buy_mask])
        actions[sell_mask] = normalized_q_values[sell_mask] * (current_position[sell_mask] - self.min_position)
        
        # Round to integers
        actions = np.round(actions).astype(int)

        return actions