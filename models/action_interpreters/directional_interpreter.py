import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from utils.logger import Logger
logger = Logger.get_logger()

from environments.trading_env import TradingEnv

class DirectionalActionInterpreter:
    """
    Interprets direction logits from a direction head to discrete actions.
    
    The direction head outputs logits for buy/hold/sell actions.
    - Uses softmax to normalize outputs to probabilities
    - Uses argmax to find the most likely action (0=sell, 1=hold, 2=buy)
    - Uses signal strength (probability) for buy/sell to scale the action size
    """
    
    def __init__(
        self,
        env: TradingEnv,
        min_action_scale: float = 0.1,  # Minimum scale to apply even with low confidence
    ):
        """
        Initialize the directional action interpreter.
        
        Args:
            env: Trading environment instance
            min_action_scale: Minimum scale to apply to actions (prevents tiny actions)
        """
        self.env = env
        self.min_action_scale = min_action_scale
        
        # Get position limits from environment
        position_limits = env.get_position_limits()
        self.min_position = position_limits["min"]
        self.max_position = position_limits["max"]
        
        # Calculate position range for scaling
        self.position_range = self.max_position - self.min_position
        
        # Log initialization
        logger.info(f"Initialized DirectionalActionInterpreter with min_action_scale={min_action_scale}")
        logger.info(f"Position limits: min={self.min_position}, max={self.max_position}")
    
    def interpret(
        self,
        direction_logits: np.ndarray,
        current_position: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Interpret direction logits as discrete actions (number of shares to buy/sell).
        
        Args:
            direction_logits: Direction logits from network (shape: [n_assets, 3] for [sell, hold, buy])
            current_position: Current position of the agent
            deterministic: Whether to use deterministic interpretation (argmax vs sampling)
            
        Returns:
            Action array representing number of shares to buy (positive) or sell (negative) for each asset
        """
        # Convert tensor to numpy if it's a tensor
        if isinstance(direction_logits, torch.Tensor):
            # Handle CUDA tensors by moving to CPU first
            if direction_logits.is_cuda:
                direction_logits = direction_logits.cpu()
            direction_logits = direction_logits.detach().numpy()
            
        # If batch dimension exists, assume first non-batch prediction
        if len(direction_logits.shape) == 3:  # [batch, n_assets, 3]
            direction_logits = direction_logits[0]
            
        # SAFETY CHECK: If all logits are NaN, return no action (all zeros)
        if np.isnan(direction_logits).all():
            print("WARNING: All logits from model are NaN. Returning no action.")
            return np.zeros_like(current_position)
            
        # SAFETY CHECK: If some logits are NaN, replace with zeros
        if np.isnan(direction_logits).any():
            print(f"WARNING: Some NaN values in logits ({np.isnan(direction_logits).sum()} NaNs). Replacing with zeros.")
            direction_logits = np.nan_to_num(direction_logits, nan=0.0)
        
        # Apply softmax to get probabilities
        probs = self._softmax(direction_logits)  # Shape: [n_assets, 3]
        
        # Either select max probability action (deterministic) or sample from distribution
        if deterministic:
            selected_actions = np.argmax(probs, axis=1)  # Shape: [n_assets]
        else:
            # Sample from probability distribution for each asset
            selected_actions = np.array([
                np.random.choice(3, p=probs[i]) for i in range(probs.shape[0])
            ])

        # DEBUG: Check for NaN in selected_actions and probs
        if np.isnan(selected_actions).any():
            print("WARNING: NaN found in selected_actions")
            print(f"selected_actions: {selected_actions}")
        if np.isnan(probs).any():
            print("WARNING: NaN found in probs")
            print(f"probs: {probs}")
        if np.isnan(direction_logits).any():
            print("WARNING: NaN found in direction_logits")
            print(f"direction_logits: {direction_logits}")

        # Create masks for different actions
        sell_mask = selected_actions == 0
        hold_mask = selected_actions == 1
        buy_mask = selected_actions == 2
        
        # Initialize actions array
        actions = np.zeros_like(current_position)
        
        # For sell actions, use probability to scale how much to sell
        if np.any(sell_mask):
            # Get confidence (probability) for sell actions
            sell_confidence = probs[sell_mask, 0]
            # Scale confidence and ensure minimum action scale
            scaled_confidence = np.maximum(sell_confidence, self.min_action_scale)
            # Calculate position difference for selling (how much can be sold)
            position_diff = current_position[sell_mask] - self.min_position
            
            # DEBUG: Check for NaN in sell calculations
            if np.isnan(sell_confidence).any():
                print("WARNING: NaN found in sell_confidence")
            if np.isnan(scaled_confidence).any():
                print("WARNING: NaN found in scaled_confidence for sell")
            if np.isnan(position_diff).any():
                print("WARNING: NaN found in position_diff for sell")
            
            # Only sell positions that are above minimum (avoid negative or invalid values)
            valid_sells = position_diff > 0
            if np.any(valid_sells):
                # Calculate the values to use for selling
                sell_values = -np.round(scaled_confidence[valid_sells] * position_diff[valid_sells])
                
                # DEBUG: Check for NaN in sell_values
                if np.isnan(sell_values).any():
                    print("WARNING: NaN found in sell_values before casting")
                    print(f"scaled_confidence: {scaled_confidence[valid_sells]}")
                    print(f"position_diff: {position_diff[valid_sells]}")
                # Convert to int safely (NaN values handled above)
                sell_actions = sell_values.astype(np.int32)
                # Get indices of assets with sell actions
                sell_indices = np.where(sell_mask)[0][valid_sells]
                # Apply sell actions
                for i, idx in enumerate(sell_indices):
                    actions[idx] = sell_actions[i]
        
        # For buy actions, use probability to scale how much to buy
        if np.any(buy_mask):
            # Get confidence (probability) for buy actions
            buy_confidence = probs[buy_mask, 2]
            # Scale confidence and ensure minimum action scale
            scaled_confidence = np.maximum(buy_confidence, self.min_action_scale)
            # Calculate position difference for buying (how much can be bought)
            position_diff = self.max_position - current_position[buy_mask]
            
            # DEBUG: Check for NaN in buy calculations
            if np.isnan(buy_confidence).any():
                print("WARNING: NaN found in buy_confidence")
            if np.isnan(scaled_confidence).any():
                print("WARNING: NaN found in scaled_confidence for buy")
            if np.isnan(position_diff).any():
                print("WARNING: NaN found in position_diff for buy")
            
            # Only buy positions that have room (avoid negative or invalid values)
            valid_buys = position_diff > 0
            if np.any(valid_buys):
                # Calculate the values to use for buying
                buy_values = np.round(scaled_confidence[valid_buys] * position_diff[valid_buys])
                
                # DEBUG: Check for NaN in buy_values
                if np.isnan(buy_values).any():
                    print("WARNING: NaN found in buy_values before casting")
                    print(f"scaled_confidence: {scaled_confidence[valid_buys]}")
                    print(f"position_diff: {position_diff[valid_buys]}")
                # Convert to int safely (NaN values handled above)
                buy_actions = buy_values.astype(np.int32)
                # Get indices of assets with buy actions
                buy_indices = np.where(buy_mask)[0][valid_buys]
                # Apply buy actions
                for i, idx in enumerate(buy_indices):
                    actions[idx] = buy_actions[i]
        
        # Hold actions remain as 0
        return actions
    
    def interpret_with_log_prob(
        self,
        direction_logits: Union[np.ndarray, Dict[str, np.ndarray]],
        current_position: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpret direction logits and also return log probabilities for the selected actions.
        Used for training with policy gradient methods.
        
        Args:
            direction_logits: Direction logits from network (shape: [n_assets, 3])
            current_position: Current position of the agent
            
        Returns:
            Tuple of (actions, log_probs)
        """
        if isinstance(direction_logits, dict):
            direction_logits = direction_logits["direction"]
            
        if isinstance(direction_logits, torch.Tensor):
            # Handle CUDA tensors by moving to CPU first
            if direction_logits.is_cuda:
                direction_logits = direction_logits.cpu()
            direction_logits = direction_logits.detach().numpy()
            
        # If batch dimension exists, assume first non-batch prediction
        if len(direction_logits.shape) == 3:  # [batch, n_assets, 3]
            direction_logits = direction_logits[0]
            
        # SAFETY CHECK: If all logits are NaN, return no action (all zeros)
        if np.isnan(direction_logits).all():
            print("WARNING: All logits from model are NaN. Returning no action.")
            return np.zeros_like(current_position), np.zeros(current_position.shape[0])
            
        # SAFETY CHECK: If some logits are NaN, replace with zeros
        if np.isnan(direction_logits).any():
            print(f"WARNING: Some NaN values in logits ({np.isnan(direction_logits).sum()} NaNs). Replacing with zeros.")
            direction_logits = np.nan_to_num(direction_logits, nan=0.0)
        
        # Apply softmax to get probabilities
        probs = self._softmax(direction_logits)  # Shape: [n_assets, 3]
        
        # Sample from probability distribution for each asset
        selected_actions = np.array([
            np.random.choice(3, p=probs[i]) for i in range(probs.shape[0])
        ])
        
        # Get log probabilities of selected actions
        log_probs = np.log(np.array([
            probs[i, selected_actions[i]] for i in range(len(selected_actions))
        ]))
        
        # Create masks for different actions
        sell_mask = selected_actions == 0
        hold_mask = selected_actions == 1
        buy_mask = selected_actions == 2
        
        # Initialize actions array
        actions = np.zeros_like(current_position)
        
        # For sell actions, use probability to scale how much to sell
        if np.any(sell_mask):
            # Get confidence (probability) for sell actions
            sell_confidence = probs[sell_mask, 0]
            # Scale confidence and ensure minimum action scale
            scaled_confidence = np.maximum(sell_confidence, self.min_action_scale)
            # Calculate position difference for selling (how much can be sold)
            position_diff = current_position[sell_mask] - self.min_position
            
            # Only sell positions that are above minimum (avoid negative or invalid values)
            valid_sells = position_diff > 0
            if np.any(valid_sells):
                # Calculate the values to use for selling
                sell_values = -np.round(scaled_confidence[valid_sells] * position_diff[valid_sells])
                # Convert to int safely (NaN values handled above)
                sell_actions = sell_values.astype(np.int32)
                # Get indices of assets with sell actions
                sell_indices = np.where(sell_mask)[0][valid_sells]
                # Apply sell actions
                for i, idx in enumerate(sell_indices):
                    actions[idx] = sell_actions[i]
        
        # For buy actions, use probability to scale how much to buy
        if np.any(buy_mask):
            # Get confidence (probability) for buy actions
            buy_confidence = probs[buy_mask, 2]
            # Scale confidence and ensure minimum action scale
            scaled_confidence = np.maximum(buy_confidence, self.min_action_scale)
            # Calculate position difference for buying (how much can be bought)
            position_diff = self.max_position - current_position[buy_mask]

            
            # Only buy positions that have room (avoid negative or invalid values)
            valid_buys = position_diff > 0
            if np.any(valid_buys):
                # Calculate the values to use for buying
                buy_values = np.round(scaled_confidence[valid_buys] * position_diff[valid_buys])
                # Convert to int safely (NaN values handled above)
                buy_actions = buy_values.astype(np.int32)
                # Get indices of assets with buy actions
                buy_indices = np.where(buy_mask)[0][valid_buys]
                # Apply buy actions
                for i, idx in enumerate(buy_indices):
                    actions[idx] = buy_actions[i]
        
        # Hold actions remain as 0
        return actions, log_probs
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax function to the last dimension of the input array.
        Includes improved numerical stability and NaN handling.
        
        Args:
            x: Input array
            
        Returns:
            Softmax of the input array along the last dimension
        """
        # First, check for and handle any NaN/inf values in input
        if np.isnan(x).any() or np.isinf(x).any():
            # Replace NaN/inf with zeros before softmax to avoid propagation
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Improved numerical stability - use a smaller shift value
        # to avoid extreme values that could lead to NaN after exp()
        shiftx = x - np.max(x, axis=-1, keepdims=True)
        
        # Clip to avoid overflow/underflow
        shiftx = np.clip(shiftx, -88.0, 88.0)  # exp(±88) is within float32 range
        
        # Compute softmax with the stabilized values
        exps = np.exp(shiftx)
        sums = np.sum(exps, axis=-1, keepdims=True)
        
        # Avoid division by zero
        sums = np.maximum(sums, 1e-10)
        
        # Calculate softmax and ensure no NaN values
        result = exps / sums
        
        # Final NaN check (defensive programming)
        if np.isnan(result).any():
            # If we still have NaNs, use a uniform distribution as fallback
            dim = x.shape[-1]
            mask = np.any(np.isnan(result), axis=-1, keepdims=True)
            uniform = np.ones_like(result) / dim
            result = np.where(mask, uniform, result)
            
        return result 