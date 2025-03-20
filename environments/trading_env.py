from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from gym import spaces
from .base_env import BaseTradingEnv
from .rewards import returns_based, sharpe_based, risk_adjusted
from .market_friction import slippage, commission, market_impact
from .constraints import position_limits, risk_limits, regulatory_limits


class TradingEnv(BaseTradingEnv):
    """
    A trading environment that simulates trading with realistic market conditions.
    Supports multiple assets, market frictions, and various constraints.
    Uses pre-calculated technical indicators from the data manager.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        raw_data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_fee_percent: float = 0.001,
        slippage_model: str = "constant",
        reward_type: str = "returns",
        position_limits: Optional[Dict[str, float]] = None,
        risk_limits: Optional[Dict[str, float]] = None,
        window_size: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Initialize the trading environment.

        Args:
            data: DataFrame with normalized OHLCV data and technical indicators
            raw_data: DataFrame with raw OHLCV data for actual trading calculations
            initial_balance: Starting capital
            transaction_fee_percent: Transaction fee as percentage
            slippage_model: Type of slippage model to use
            reward_type: Type of reward function to use
            position_limits: Dictionary of position limits per asset
            risk_limits: Dictionary of risk limits (e.g., max drawdown)
            window_size: Number of time steps to include in observation
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Data setup
        self.data = data
        self.raw_data = raw_data
        self.n_assets = len(data.columns) // 5  # Assuming OHLCV format
        self.window_size = window_size

        # Identify feature columns (excluding OHLCV and metadata)
        self.feature_columns = [
            col
            for col in data.columns
            if col not in ["Open", "High", "Low", "Close", "Volume", "date", "ticker"]
        ]

        # Separate price and technical indicator columns
        self.price_columns = ["Open", "High", "Low", "Close", "Volume"]
        self.technical_columns = self.feature_columns

        # Trading parameters
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = np.zeros(self.n_assets)
        self.transaction_fee_percent = transaction_fee_percent

        # Define action space (percentage of portfolio to allocate to each asset)
        self.action_space = spaces.Box(
            low=-1,  # Short positions
            high=1,  # Long positions
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # Define observation space
        # [Normalized OHLCV data for window_size steps] + [technical indicators] + [current positions] + [current balance]
        obs_shape = (
            self.window_size * 5 * self.n_assets  # Normalized OHLCV data
            + len(self.technical_columns) * self.n_assets  # Technical indicators
            + self.n_assets  # Current positions
            + 1  # Current balance
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

        # Initialize components
        self._init_reward_function(reward_type)
        self._init_market_friction(slippage_model)
        self._init_constraints(position_limits, risk_limits)

        # Set random seed
        self.seed(seed)

        # Initialize state
        self.current_step = self.window_size
        self.done = False
        self.info = {}

    def _init_reward_function(self, reward_type: str) -> None:
        """Initialize the reward function based on type."""
        reward_functions = {
            "returns": returns_based.ReturnsBasedReward(),
            "sharpe": sharpe_based.SharpeBasedReward(),
            "risk_adjusted": risk_adjusted.RiskAdjustedReward(),
        }
        self.reward_function = reward_functions.get(reward_type)
        if not self.reward_function:
            raise ValueError(f"Unknown reward type: {reward_type}")

    def _init_market_friction(self, slippage_model: str) -> None:
        """Initialize market friction components."""
        self.slippage = slippage.SlippageModel(model_type=slippage_model)
        self.commission = commission.CommissionModel(
            fee_percent=self.transaction_fee_percent
        )
        self.market_impact = market_impact.MarketImpactModel()

    def _init_constraints(
        self,
        position_limits: Optional[Dict[str, float]],
        risk_limits: Optional[Dict[str, float]],
    ) -> None:
        """Initialize trading constraints."""
        self.position_constraints = position_limits.PositionLimits(
            limits=position_limits or {}
        )
        self.risk_constraints = risk_limits.RiskLimits(limits=risk_limits or {})
        self.regulatory_constraints = regulatory_limits.RegulatoryLimits()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = self.window_size
        self.current_balance = self.initial_balance
        self.positions = np.zeros(self.n_assets)
        self.done = False
        self.info = {}
        return self._get_observation()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Array of portfolio weights for each asset

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current market data
        current_data = self._get_current_data()

        # Apply market frictions
        action = self._apply_market_frictions(action, current_data[0])

        # Check constraints
        if not self._check_constraints(action):
            self.done = True
            return self._get_observation(), -1000.0, True, self.info

        # Execute trades
        self._execute_trades(action, current_data)

        # Update state
        self.current_step += 1
        self.done = self._is_done()

        # Calculate reward
        reward = self._calculate_reward(action)

        # Update info
        self.info = self._update_info()

        return self._get_observation(), reward, self.done, self.info

    def _get_current_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get the current window of market data (both normalized and raw)."""
        normalized_data = self.data.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        raw_data = self.raw_data.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        return normalized_data, raw_data

    def _apply_market_frictions(
        self, action: np.ndarray, current_data: pd.DataFrame
    ) -> np.ndarray:
        """Apply market frictions to the action."""
        # Apply slippage
        action = self.slippage.apply(action, current_data)

        # Apply commission
        action = self.commission.apply(action, self.current_balance)

        # Apply market impact
        action = self.market_impact.apply(action, current_data)

        return action

    def _check_constraints(self, action: np.ndarray) -> bool:
        """Check if the action satisfies all constraints."""
        # Check position limits
        if not self.position_constraints.check(action):
            return False

        # Check risk limits
        if not self.risk_constraints.check(action, self.positions):
            return False

        # Check regulatory constraints
        if not self.regulatory_constraints.check(action):
            return False

        return True

    def _execute_trades(
        self, action: np.ndarray, current_data: Tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Execute trades based on the action."""
        _, raw_data = current_data

        # Calculate target positions
        target_positions = action * self.current_balance

        # Update positions
        self.positions = target_positions

        # Update balance using raw price data
        price_changes = raw_data[self.price_columns].iloc[-1].pct_change()
        self.current_balance *= 1 + np.sum(action * price_changes)

    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        # Get current window of market data
        normalized_data, _ = self._get_current_data()

        # Extract normalized OHLCV data for observations
        ohlcv_data = normalized_data[self.price_columns].values.flatten()

        # Extract technical indicators (already normalized)
        technical_indicators = (
            normalized_data[self.technical_columns].iloc[-1].values.flatten()
        )

        # Combine all components
        observation = np.concatenate(
            [
                ohlcv_data,  # Normalized price data for observations
                technical_indicators,  # Normalized technical indicators
                self.positions,
                np.array([self.current_balance]),
            ]
        )

        return observation

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate the reward for the action."""
        return self.reward_function.calculate(
            action=action,
            positions=self.positions,
            current_balance=self.current_balance,
            market_data=self._get_current_data(),
        )

    def _is_done(self) -> bool:
        """Check if the episode is done."""
        return self.current_step >= len(self.data) - 1

    def _update_info(self) -> Dict[str, Any]:
        """Update the info dictionary with current state."""
        return {
            "balance": self.current_balance,
            "positions": self.positions,
            "step": self.current_step,
            "returns": (self.current_balance - self.initial_balance)
            / self.initial_balance,
        }

    def render(self, mode: str = "human") -> None:
        """Render the environment."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.current_balance:.2f}")
            print(f"Positions: {self.positions}")
            print(f"Returns: {self.info['returns']:.2%}")

    def close(self) -> None:
        """Clean up resources."""
        pass

    def seed(self, seed: Optional[int] = None) -> list:
        """Set the random seed."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]
