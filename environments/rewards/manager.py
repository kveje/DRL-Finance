from .returns import ReturnsReward
from .sharpe import SharpeReward
from .constraint_violation import ConstraintViolationReward
from .log_returns import LogReturnsReward
from .zero_action import ZeroActionReward
from .projected_returns import ProjectedReturnsReward
from .projected_sharpe import ProjectedSharpeReward
from .projected_log_returns import ProjectedLogReturnsReward
from .projected_max_drawdown import ProjectedMaxDrawdownReward
from .max_drawdown import MaxDrawdownReward
import pandas as pd
from typing import Tuple, Dict, Any, List
import numpy as np
from collections import defaultdict

class RewardManager:
    """Manager class that handles multiple reward components."""

    REWARD_TYPES = {
        "log_returns": LogReturnsReward,
        "returns": ReturnsReward,
        "sharpe_based": SharpeReward,
        "constraint_violation": ConstraintViolationReward,
        "zero_action": ZeroActionReward,
        "max_drawdown": MaxDrawdownReward,
        "projected_returns": ProjectedReturnsReward,
        "projected_sharpe": ProjectedSharpeReward,
        "projected_log_returns": ProjectedLogReturnsReward,
        "projected_max_drawdown": ProjectedMaxDrawdownReward,
    }

    def __init__(self, reward_params: Dict[str, Any], raw_data_feature_indices: Dict[str, int]):
        """
        Initialize the reward manager.

        Args:
            reward_params: Dictionary of reward parameters

        Example:
        {
            "log_returns": {"scale": 1.0},
            "returns": {"scale": 1.0},
            "sharpe_based": {"scale": 1.0},
            "constraint_violation": {"scale": 1.0},
            "zero_action": {"scale": 0.001, "window_size": 5}
        }
        """
        self.reward_params = reward_params
        self.rewards = {}
        self.reward_history = defaultdict(list)  # Track individual reward components
        self.raw_data_feature_indices = raw_data_feature_indices
        self._initialize_rewards()

    def _initialize_rewards(self):
        """Initialize the reward components based on configuration."""
        # Initialize the main reward component
        for reward_type, reward_params in self.reward_params.items():
            if reward_type == "returns":
                self.rewards[reward_type] = ReturnsReward(reward_params)
            elif reward_type == "sharpe_based":
                self.rewards[reward_type] = SharpeReward(reward_params)
            elif reward_type == "log_returns":
                self.rewards[reward_type] = LogReturnsReward(reward_params)
            elif reward_type == "constraint_violation":
                self.rewards[reward_type] = ConstraintViolationReward(reward_params)
            elif reward_type == "zero_action":
                self.rewards[reward_type] = ZeroActionReward(reward_params)
            elif reward_type == "projected_returns":
                self.rewards[reward_type] = ProjectedReturnsReward(reward_params, self.raw_data_feature_indices)
            elif reward_type == "projected_sharpe":
                self.rewards[reward_type] = ProjectedSharpeReward(reward_params, self.raw_data_feature_indices)
            elif reward_type == "projected_log_returns":
                self.rewards[reward_type] = ProjectedLogReturnsReward(reward_params, self.raw_data_feature_indices)
            elif reward_type == "projected_max_drawdown":
                self.rewards[reward_type] = ProjectedMaxDrawdownReward(reward_params, self.raw_data_feature_indices)
            elif reward_type == "max_drawdown":
                self.rewards[reward_type] = MaxDrawdownReward(reward_params, self.raw_data_feature_indices)
            else:
                raise ValueError(f"Unknown reward type: {reward_type}")

    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        intended_action: np.ndarray = None,
        feasible_action: np.ndarray = None,
        raw_data: np.ndarray = None,
        current_day: int = None,
        current_position: np.ndarray = None,
        cash_balance: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the total reward and individual reward components.

        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Previous portfolio value
            intended_action: Original action proposed by the agent
            feasible_action: Action after applying constraints

        Returns:
            Tuple of (total_reward, reward_components)
            - total_reward (float): Total reward value
            - reward_components (Dict[str, float]): Dictionary mapping reward types to their values
        """
        total_reward = 0.0
        step_rewards = {}  # Track rewards for this step

        for reward_type, reward in self.rewards.items():
            if isinstance(reward, ReturnsReward):
                reward_value = reward.calculate(portfolio_value, previous_portfolio_value)
            elif isinstance(reward, SharpeReward):
                reward_value = reward.calculate(portfolio_value, previous_portfolio_value)
            elif isinstance(reward, LogReturnsReward):
                reward_value = reward.calculate(portfolio_value, previous_portfolio_value)
            elif isinstance(reward, ConstraintViolationReward):
                if intended_action is not None and feasible_action is not None:
                    reward_value = reward.calculate(intended_action, feasible_action)
                else:
                    reward_value = 0.0
            elif isinstance(reward, ZeroActionReward):
                if feasible_action is not None and intended_action is not None:
                    reward_value = reward.calculate(intended_action, feasible_action)
                else:
                    reward_value = 0.0
            elif isinstance(reward, ProjectedReturnsReward):
                if raw_data is not None and current_day is not None and current_position is not None:
                    reward_value = reward.calculate(raw_data, current_day, current_position, previous_portfolio_value, cash_balance)
                else:
                    reward_value = 0.0
            elif isinstance(reward, ProjectedSharpeReward):
                if raw_data is not None and current_day is not None and current_position is not None:
                    reward_value = reward.calculate(raw_data, current_day, current_position, cash_balance)
                else:
                    reward_value = 0.0
            elif isinstance(reward, ProjectedLogReturnsReward):
                if raw_data is not None and current_day is not None and current_position is not None:
                    reward_value = reward.calculate(raw_data, current_day, current_position, previous_portfolio_value, cash_balance)
                else:
                    reward_value = 0.0
            elif isinstance(reward, MaxDrawdownReward):
                if raw_data is not None and current_day is not None and current_position is not None:
                    reward_value = reward.calculate(raw_data, current_day, current_position)
                else:
                    reward_value = 0.0
            elif isinstance(reward, ProjectedMaxDrawdownReward):
                if raw_data is not None and current_day is not None and current_position is not None:
                    reward_value = reward.calculate(raw_data, current_day, current_position, cash_balance)
                else:
                    reward_value = 0.0
            else:
                raise ValueError(f"Unknown reward type: {type(reward)}")
            
            total_reward += reward_value
            step_rewards[reward_type] = reward_value

        # Store individual rewards for this step
        for reward_type, value in step_rewards.items():
            self.reward_history[reward_type].append(value)

        return total_reward, step_rewards

    def reset(self):
        """Reset the reward components and history."""
        for reward in self.rewards.values():
            if hasattr(reward, 'reset'):
                reward.reset()
        self.reward_history.clear()

    def get_reward_history(self) -> Dict[str, List[float]]:
        """
        Get the history of individual reward components.

        Returns:
            Dict[str, List[float]]: Dictionary mapping reward types to lists of their values
        """
        return dict(self.reward_history)

    def get_reward_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each reward component.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping reward types to their statistics
            Statistics include: mean, std, min, max, total
        """
        stats = {}
        for reward_type, values in self.reward_history.items():
            if values:  # Only calculate stats if we have values
                stats[reward_type] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "total": np.sum(values)
                }
        return stats

    def __str__(self) -> str:
        """Return a string representation of the reward manager."""
        return f"RewardManager(rewards={self.rewards})"
    
    def __repr__(self) -> str:
        """Return a string representation of the reward manager."""
        return self.__str__()

    def get_parameters(self) -> Dict[str, Any]:
        """Get the reward parameters."""
        params = {}
        for reward in self.rewards.values():
            params.update(reward.get_parameters())
        return params

