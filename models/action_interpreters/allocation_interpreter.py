"""Allocation interpreters for trading agents."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from models.action_interpreters.base_action_interpreter import BaseActionInterpreter
from environments.trading_env import TradingEnv

class AllocationInterpreter(BaseActionInterpreter):
    """
    Translates raw network outputs and current state into constrained,
    discrete actions using a risk-aware allocation strategy.
    """
    ALLOCATION_METHODS = ["greedy", "equal", "risk_parity"]
    def __init__(self, env: TradingEnv, w1: float = 1.0, w2: float = 1.0, allocation_method: str = "greedy"):
        """
        Initialize the allocation interpreter.

        Args:
            env: The trading environment.
            w1: The weight of the first risk factor.
            w2: The weight of the second risk factor.
            allocation_method: The method to use for allocation.
        """
        super().__init__(env)
        self.w1 = w1
        self.w2 = w2
        self.allocation_method = allocation_method
        self.position_limits = env.get_position_limits() # Dict of min and max position limits for each asset


    def _get_beta_distribution_params(self, raw_alpha: np.ndarray, raw_beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the parameters of the beta distribution.
        """
        # Softplus in numpy
        alpha = np.log1p(np.exp(raw_alpha)) + 1e-6
        beta = np.log1p(np.exp(raw_beta)) + 1e-6
        return alpha, beta
    
    def _calculate_risk_aware_objective_params(self, alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates mean and variance of the beta distribution scaled by the position limits.

        Args:
            alpha: The alpha parameter of the beta distribution. # Shape: (batch, assets)
            beta: The beta parameter of the beta distribution. # Shape: (batch, assets)

        Returns:
            target_mean: The target mean of each asset. # Shape: (batch, assets)
            target_variance: The target variance of each asset. # Shape: (batch, assets)
        """
        # Calculate the mean and variance of the beta distribution
        mean = alpha / (alpha + beta)
        variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

        # Scale the mean and variance by the position limits
        target_mean = mean * (self.position_limits["max"] - self.position_limits["min"]) + self.position_limits["min"]
        target_variance = variance * (self.position_limits["max"] - self.position_limits["min"]) ** 2

        return target_mean, target_variance
    
    def _determine_intents(self, direction_logits: np.ndarray, target_mean: np.ndarray, current_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determines the intents of the agent.

        Args:
            direction_logits: The logits of the direction of the agent. # Shape:  (assets, 3)
            target_mean: The target mean of the agent. # Shape:(assets)
            current_position: The current position of the agent. # Shape: (assets)

        Returns:
            intended_trade: The intended trade of the agent. # Shape: (assets)
        """
        # Get the direction probabilities using softmax
        exp_logits = np.exp(direction_logits - np.max(direction_logits, axis=-1, keepdims=True))
        direction_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        directions = np.argmax(direction_probs, axis=-1)  # (assets) (0 = B, 1 = S, 2 = H) or 

        # Calculate desired position changes
        desired_change = target_mean - current_position  # (assets)

        # Determine the mask
        # hold_mask = directions == 2
        buy_mask = directions == 0
        sell_mask = directions == 1

        # Calculate the intent (Default to hold)
        intent = np.ones_like(directions) * 2  #  (assets)

        # Intents
        intent[buy_mask & (desired_change > 0)] = 0
        intent[sell_mask & (desired_change < 0)] = 1

        # Max trade size
        intended_trade = np.zeros_like(intent)  # (assets)

        # Apply logic for intent
        intended_trade[intent == 0] = desired_change[intent == 0]
        intended_trade[intent == 1] = -desired_change[intent == 1]

        # Round the inteded trade to the nearest integer
        return np.round(intended_trade)
    
    def interpret(self, model_output: Dict[str,np.ndarray], current_position: np.ndarray) -> np.ndarray:
        """
        Interpret the model output into an action.

        Args:
            model_output: The output of the model. {"direction": np.ndarray, "alpha": np.ndarray, "beta": np.ndarray}
            current_position: The current position of the agent.

        Returns:
            The inteded trade of the agent. (possibly infeasible)
        """
        # Get the model output
        direction_logits = model_output["direction"]
        alpha_logits = model_output["alpha"]
        beta_logits = model_output["beta"]

        # Get the parameters of the beta distribution
        alpha, beta = self._get_beta_distribution_params(alpha_logits, beta_logits)

        # Calculate the target mean and variance (only mean is used)
        target_mean, _ = self._calculate_risk_aware_objective_params(alpha, beta)

        # Determine the intents
        intended_trade = self._determine_intents(direction_logits, target_mean, current_position)

        return intended_trade