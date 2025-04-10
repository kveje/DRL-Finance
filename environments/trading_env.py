from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from gym import spaces
import random

# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()

# Import internal dependencies
from .base_env import BaseTradingEnv
from .rewards import RewardManager
from .market_friction import MarketFrictionManager
from .constraints import ConstraintManager

class TradingEnv(BaseTradingEnv):
    """
    A trading environment that simulates trading with realistic market conditions.
    Supports multiple assets, market frictions, and various constraints.
    Uses pre-calculated technical indicators from the data manager.
    """

    def __init__(
        self,
        processed_data: pd.DataFrame,
        raw_data: pd.DataFrame,
        columns: Dict[str, Union[str, List[str]]], # {ticker: "ticker", price: "close", day: "day", ohlcv: ["open", "high", "low", "close", "volume"], tech_cols: ["RSI", "MACD", "Bollinger Bands"]}
        env_params: Dict[str, Any] = {}, # {initial_balance: 100000.0, window_size: 10}
        friction_params: Dict[str, Dict[str, float]] = {}, # {slippage: {slippage_mean: 0.0, slippage_std: 0.001}, commission: {commission_rate: 0.001}}
        reward_params: Tuple[str, Dict[str, Any]] = ("returns_based", {"scale": 1.0}), # (reward_type, reward_params) e.g. ("returns_based", {"scale": 1.0})
        constraint_params: Dict[str, Dict[str, float]] = {}, # {position_limits: {min: -1000, max: 1000}}
        seed: Optional[int] = None,
    ):
        """
        Initialize the trading environment.

        Args:
            processed_data: DataFrame with normalized OHLCV data and technical indicators 
            raw_data: DataFrame with raw OHLCV data for actual trading calculations
            env_params: Dictionary of environment parameters (initial_balance, window_size, etc.)
            friction_params: Dictionary of friction parameters (slippage, commission, etc.)
            reward_params: Tuple of (reward_type, reward_params) e.g. ("returns_based", {"scale": 1.0})
            constraint_params: Dictionary of constraint parameters (position_limits, etc.)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        logger.info(f"Initializing TradingEnv")

        # Data setup
        self.processed_data: pd.DataFrame = processed_data
        self.raw_data: pd.DataFrame = raw_data

        # Data columns setup
        self.tic_col: str = columns.get("ticker", "ticker")
        self.price_col: str = columns.get("price", "close")
        self.day_col: str = columns.get("day", "day")
        self.ohlcv_cols: List[str] = columns.get("ohlcv", [])
        self.tech_cols: List[str] = columns.get("tech_cols", [])
        self.market_cols: List[str] = self.ohlcv_cols + self.tech_cols

        # Asset setup
        self.asset_list: List[str] = list(processed_data[self.tic_col].unique()) # List of asset tickers
        self.n_assets = len(self.asset_list)

        # Environment parameters
        self.env_params = env_params
        self.initial_balance: float = env_params.get("initial_balance", 100000.0) # Defaults to 100,000
        self.window_size: int = env_params.get("window_size", 10) # Defaults to 10
        
        # Portfolio tracking
        self.current_cash = self.initial_balance
        self.positions = np.zeros(self.n_assets)  # Number of shares for each asset
        self.asset_values = np.zeros(self.n_assets)  # Value of each asset position
        self.portfolio_value_history = [self.initial_balance]

        # Initialize managers for market frictions, constraints, and rewards
        self.market_frictions = MarketFrictionManager(friction_params)
        self.constraint_manager = ConstraintManager(constraint_params)
        self.reward_manager = RewardManager(reward_params)

        # Define action space (Action is the number of shares to buy or sell for each asset)
        # Actions represent the change in the number of shares to hold
        self.action_space = spaces.Box(
            low=-np.inf,  # Selling (negative values)
            high=np.inf,  # Buying (positive values)
            shape=(self.n_assets,),
            dtype=np.int32,
        )

        # Define observation space
        # [Normalized OHLCV data for window_size steps] + [technical indicators] + [current positions] + [current cash] + [portfolio value]
        self.ohlcv_dim = 5
        self.tech_dim = len(self.tech_cols)
        self.cash_dim = 1
        self.portfolio_value_dim = 1
        
        # Observation shape
        self.observation_space = spaces.Dict({
            'data': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.n_assets, self.ohlcv_dim + self.tech_dim, self.window_size), 
                dtype=np.float32
            ),
            'positions': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.n_assets,), 
                dtype=np.float32
            ),
            'portfolio_info': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(2,),
                dtype=np.float32
            )
        })

        # Initialize state
        self.current_step = self.window_size
        self.done = False
        self.info = {}

        # Log information about the environment
        self._log_env_start()

        # Set random seed
        self.seed = self._set_seed(seed)

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = self.window_size
        self.current_cash = self.initial_balance
        self.positions = np.zeros(self.n_assets)
        self.asset_values = np.zeros(self.n_assets)
        self.portfolio_value_history = [self.initial_balance]
        self.done = False
        self.info = {}
        self.reward_manager.reset()
        
        # Get observation
        observation = self._get_observation()
        
        return observation

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Array of trading actions (positive for buy, negative for sell, in number of shares)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current prices
        current_prices = self._get_current_prices()
        
        # Store previous portfolio state
        # Might just get this from portfolio_value_history
        previous_portfolio_value = self._calculate_portfolio_value(self.positions, current_prices)
        previous_portfolio_value = self.portfolio_value_history[-1]
        
        # Apply market frictions (slippage, etc.)
        adjusted_prices = self.market_frictions.apply_frictions(action, current_prices)
        
        # Check constraints
        if not self.constraint_manager.check_constraints(action, self.positions, self.current_cash, adjusted_prices):
            self.done = True
            self.info = self._update_info(previous_portfolio_value, previous_portfolio_value, "constraint_violation")
            return self._get_observation(), -1000.0, True, self.info
        
        # Execute trades
        self._execute_trades(action, adjusted_prices)
        
        # Move to next step
        self.current_step += 1
        
        # Update portfolio value history
        new_portfolio_value = self._calculate_portfolio_value(self.positions, current_prices)
        self.portfolio_value_history.append(new_portfolio_value)
        
        # Calculate reward
        reward = self.reward_manager.calculate(
            portfolio_value=new_portfolio_value,
            previous_portfolio_value=previous_portfolio_value,
        )
        
        # Update state for next step
        self.previous_prices = current_prices
        
        # Check if episode is done
        self.done = self._is_done()
        
        # Update info dictionary
        self.info = self._update_info(previous_portfolio_value, new_portfolio_value)
        
        return self._get_observation(), reward, self.done, self.info

    def _get_current_prices(self) -> np.ndarray:
        """Get current close prices for all assets."""
        prices = np.array([
            self.raw_data[self.price_col].iloc[self.current_step]
            for asset in self.asset_list
        ], dtype=np.float32)
        return prices
    
    def _get_market_data(self, asset: Optional[str] = None) -> np.ndarray:
        """Get the window of market data for one or all assets.
        Market data includes OHLCV data and technical indicators.
        
        Returns:
            np.ndarray: Market data for the window of data
            Asset Shape: (window_size, 5 + tech_dim)
            All Assets Shape: (n_assets, 5 + tech_dim, window_size)
        """ 
        if asset and asset in self.asset_list:
            market_data = self.processed_data[self.processed_data[self.tic_col] == asset].iloc[self.current_step - self.window_size:self.current_step][self.market_cols].values
            return np.array(market_data, dtype=np.float32).transpose()
        elif asset is None:
            market_data = np.zeros((self.n_assets, self.ohlcv_dim + self.tech_dim, self.window_size))
            for i, asset in enumerate(self.asset_list):
                temp = self.processed_data[self.processed_data[self.tic_col] == asset].iloc[self.current_step - self.window_size:self.current_step][self.market_cols].values
                market_data[i, :, :] = np.array(temp, dtype=np.float32).transpose()
            return market_data
        else:
            raise ValueError(f"Asset {asset} not found in asset list")

    def _execute_trades(self, action: np.ndarray, prices: np.ndarray) -> None:
        """
        Execute trades based on the action.
        
        Args:
            action: Array of changes in positions (positive for buy, negative for sell)
            prices: Current asset prices
        """
        # Update positions
        self.positions += action
        # Update cash
        self.current_cash -= np.sum(action * prices)

        # Update asset values
        self.asset_values = self.positions * prices

        # Update portfolio value
        self.portfolio_value = self.current_cash + np.sum(self.positions * prices)

        # Update portfolio value history
        self.portfolio_value_history.append(self.portfolio_value)

    def _calculate_portfolio_value(self, positions: np.ndarray, prices: np.ndarray) -> float:
        """Calculate the total portfolio value."""
        return self.current_cash + np.sum(positions * prices)

    def _get_observation(self) -> np.ndarray:
        """Get the current observation.
        
        Returns:
            np.ndarray: Observation for all assets
            Shape: (n_assets, ohlcv_dim + tech_dim + cash_dim + portfolio_value_dim, window_size)
            dtype: np.float32
        """
        
        # Get current prices and portfolio value
        current_prices = self._get_current_prices()
        portfolio_value = self._calculate_portfolio_value(self.positions, current_prices)

        # Get Market Data
        market_data = self._get_market_data()

        for i, asset in enumerate(self.asset_list):
            asset_market_data = self._get_market_data(asset)
            

        return {
            'data': market_data,
            'positions': self.positions,
            'portfolio_info': np.array([self.current_cash, portfolio_value])
        }

    def _is_done(self) -> bool:
        """Check if the episode is done."""
        # Episode ends when we reach the end of the data
        if self.current_step >= len(self.processed_data) - 1:
            return True
            
        # Episode can also end if portfolio value drops below a threshold
        # current_prices = self._get_current_prices()
        # portfolio_value = self._calculate_portfolio_value(self.positions, current_prices)
        
        # bankruptcy_threshold = np.inf # TODO: Add bankruptcy threshold ?
        # if portfolio_value < self.initial_balance * bankruptcy_threshold:
        #     logger.info(f"Bankruptcy: Portfolio value {portfolio_value} below threshold {self.initial_balance * bankruptcy_threshold}")
        #     return True
            
        return False

    def _update_info(self, previous_value: float, current_value: float, termination_reason: Optional[str] = "") -> Dict[str, Any]:
        """Update the info dictionary with current state."""
        # Calculate period return
        period_return = (current_value - previous_value) / previous_value if previous_value > 0 else 0
        
        # Calculate cumulative return
        cumulative_return = (current_value - self.initial_balance) / self.initial_balance
        
        return {
            "step": self.current_step,
            "portfolio_value": current_value,
            "cash": self.current_cash,
            "positions": self.positions,
            "asset_values": self.asset_values,
            "period_return": period_return,
            "cumulative_return": cumulative_return,
            "termination_reason": termination_reason,
        }

    def render(self, mode: str = "human") -> None:
        """Render the environment."""
        if mode == "human":
            current_prices = self._get_current_prices()
            portfolio_value = self._calculate_portfolio_value(self.positions, current_prices)
            
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Cash: ${self.current_cash:.2f}")
            print(f"Assets:")
            for i, asset in enumerate(self.asset_list):
                print(f"  {asset}: {self.positions[i]:.4f} shares @ ${current_prices[i]:.2f} = ${self.asset_values[i]:.2f}")
            print(f"Returns: {self.info.get('cumulative_return', 0):.2%}")
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented")

    def close(self) -> None:
        """Clean up resources."""
        self.reset()

    def _set_seed(self, seed: Optional[int] = None) -> list:
        """Set the random seed."""
        if seed is not None:
            logger.info(f"Setting random seed to {seed}")
            np.random.seed(seed)
            random.seed(seed)
            return seed
        else:
            random_seed = np.random.randint(0, 1000000)
            np.random.seed(random_seed)
            random.seed(random_seed)
            logger.info(f"Random seed set to {random_seed}")
            return random_seed
        
    def _log_env_start(self) -> None:
        """Log information about the environment at initialization."""
        logger.info(f"Environment initialized with {self.n_assets} assets")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        logger.info(f"Window size: {self.window_size}")
        logger.info(f"Data length: {len(self.processed_data)}")
        logger.info(f"Number of stocks: {len(self.asset_list)}")
        logger.info(f"Action space: {self.action_space}")
