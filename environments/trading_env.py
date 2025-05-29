from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from gym import spaces
import random
import time
# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()

# Import internal dependencies
from .base_env import BaseTradingEnv
from .rewards import RewardManager
from .market_friction import MarketFrictionManager
from .constraints import ConstraintManager
from .constraints.action_validator import ActionValidator
from visualization.trading_visualizer import TradingVisualizer
from .processors.composite_processor import CompositeProcessor

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
        columns: Dict[str, Union[str, List[str]]],
        env_params: Dict[str, Any] = {},
        friction_params: Dict[str, Dict[str, float]] = {},
        reward_params: Dict[str, Any] = {},
        constraint_params: Dict[str, Dict[str, float]] = {},
        processor_configs: List[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
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
            processor_configs: List of processor configurations for the observation space
            seed: Random seed for reproducibility
            render_mode: Mode for rendering, options are "human" for visualization or None
        """
        super().__init__()
        
        logger.info(f"Initializing TradingEnv")

        # Data columns setup
        self.tic_col: str = columns.get("ticker", "ticker")
        self.price_col: str = columns.get("price", "close")
        self.day_col: str = columns.get("day", "day")
        self.ohlcv_cols: List[str] = columns.get("ohlcv", [])
        self.tech_cols: List[str] = columns.get("tech_cols", [])
        self.market_cols: List[str] = self.ohlcv_cols + self.tech_cols


        self.processed_data: pd.DataFrame = processed_data
        self.raw_data: pd.DataFrame = raw_data

        # Save columns
        self.columns = columns

        # Get max step
        self.max_step = self.processed_data[self.day_col].max()

        # Asset setup
        self.asset_list: List[str] = list(processed_data[self.tic_col].unique()) # List of asset tickers
        self.n_assets = len(self.asset_list)

        # Precompute data arrays
        self._precompute_data_arrays()
        logger.info(f"Precomputed data arrays to vectorized operations")

        # Environment parameters
        self.env_params = env_params
        self.initial_balance: float = env_params.get("initial_balance", 100000.0) # Defaults to 100,000
        self.window_size: int = env_params.get("window_size", 10) # Defaults to 10
        
        # Portfolio tracking
        self.current_cash = self.initial_balance
        self.positions = np.zeros(self.n_assets)  # Number of shares for each asset
        self.asset_values = np.zeros(self.n_assets)  # Value of each asset position
        self.portfolio_value_history = [self.initial_balance]
        self.portfolio_value = self.initial_balance

        # Initialize managers for market frictions, constraints, and rewards
        self.market_frictions = MarketFrictionManager(friction_params)
        self.constraint_manager = ConstraintManager(constraint_params)
        self.action_validator = ActionValidator(self.constraint_manager)
        self.reward_manager = RewardManager(reward_params, self.raw_data_feature_indices)

        # Save the parameters
        self.env_params = env_params
        self.friction_params = friction_params
        self.constraint_params = constraint_params
        self.reward_params = reward_params
        self.processor_configs = processor_configs

        # Get limits from constraints (for scaling)
        self.limits = {constraint: self.constraint_manager.get_parameters(constraint) for constraint in self.constraint_manager.constraints}

        # Define action space (Action is the number of shares to buy or sell for each asset)
        # Actions represent the change in the number of shares to hold
        self.action_space = spaces.Box(
            low=-np.inf,  # Selling (negative values)
            high=np.inf,  # Buying (positive values)
            shape=(self.n_assets,),
            dtype=np.int32,
        )

        # Initialize processor
        self.processor = CompositeProcessor(processor_configs, self.raw_data_feature_indices, self.processed_data_feature_indices, self.tech_col_indices)
        
        # Update observation space based on processor
        self.observation_space = spaces.Dict(self.processor.get_observation_space())

        # Initialize state
        self.current_step = self.window_size
        self.done = False
        self.info = {}

        # Set rendering mode
        self.render_mode = render_mode
        self.visualizer = None
        if self.render_mode == "human":
            # Initialize the visualizer with a reasonable window size
            vis_window_size = min(100, int(self.max_step * 0.1))  # 10% of max steps or 100, whichever is smaller
            self.visualizer = TradingVisualizer(
                asset_names=self.asset_list,
                window_size=vis_window_size,
                update_interval=1  # Update every step when rendered
            )

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
        self.portfolio_value = self.initial_balance
        self.done = False
        self.info = {}
        self.reward_manager.reset()
        
        # Get observation
        observation = self.processor.process(raw_data = self.raw_matrix, 
                                    processed_data = self.processed_matrix, 
                                    current_step = self.current_step, 
                                    position = self.positions, 
                                    cash = self.current_cash)
        
        return observation

    def step(
        self, intended_action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            intended_action: Array of desired trading actions (change in shares) from the agent.

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # --- 1. Get current prices and info---
        current_prices = self._get_current_prices()
        previous_portfolio_value = self.portfolio_value # Use the value calculated at the end of the previous step
        current_positions = self.positions.copy() # Keep a copy before potential changes
        current_cash = self.current_cash

        # --- 2. Apply Market Frictions ---
        # Prices might change due to slippage based on the intended trade size
        # This includes commission and slippage
        adjusted_prices = self.market_frictions.apply_frictions(intended_action, current_prices)

        # --- 3. Apply Constraints to get Feasible Action ---
        feasible_action, violation_info = self.action_validator.validate_and_adjust_action(
            intended_action=intended_action,
            current_positions=current_positions,
            current_cash=current_cash,
            adjusted_prices=adjusted_prices
        )

        # --- 4. Execute Trades ---
        # Use the feasible_action and adjusted_prices for execution
        self._execute_trades(feasible_action, adjusted_prices)

        # --- 5. Calculate Reward ---
        # Reward is based on the change in portfolio value and constraint violations
        reward, reward_components = self.reward_manager.calculate(
            portfolio_value=self.portfolio_value,
            previous_portfolio_value=previous_portfolio_value,
            intended_action=intended_action,
            feasible_action=feasible_action,
            raw_data=self.raw_matrix,
            current_day=self.current_step,
            current_position=current_positions,
            cash_balance=current_cash
        )

        # --- 6. Post-computation and State Update ---
        # Portfolio value is updated within _execute_trades based on feasible action
        self.current_step += 1

        # --- 7. Check if episode is done ---
        self.done = self._is_done()

        # --- 8. Update Info Dictionary ---
        # Include info about the trade execution if desired
        self.info = self._update_info(
            previous_value=previous_portfolio_value,
            current_value=self.portfolio_value,
            intended_action=intended_action,
            feasible_action=feasible_action,
            reward=reward,
            reward_components=reward_components
        )

        # --- 9. Get Next Observation ---
        observation = self.processor.process(raw_data = self.raw_matrix, 
                                    processed_data = self.processed_matrix, 
                                    current_step = self.current_step, 
                                    position = self.positions, 
                                    cash = self.current_cash)

        return observation, reward, self.done, self.info

    def _get_current_prices(self) -> np.ndarray:
        """
        Get current prices for all assets.
        
        Returns:
            np.ndarray: Array of current prices for all assets
        """
        # Extract prices for each asset in the correct order
        prices = self.raw_matrix[self.current_step, :, self.raw_data_feature_indices[self.price_col]]
        
        return prices

    def _execute_trades(self, feasible_action: np.ndarray, prices: np.ndarray) -> None:
        """
        Execute trades based on the *feasible* action.
        
        Args:
            feasible_action: Array of feasible actions (change in shares)
            prices: Asset prices used for execution (already adjusted for slippage)
        """
        # Calculate value change based on *feasible* trades
        value_change = feasible_action * prices
        
        # Update cash: subtract cost of buys, add proceeds from sells, subtract total commission
        self.current_cash -= np.sum(value_change)
        
        # Update positions
        self.positions += feasible_action
        
        # Update asset values (using the prices at which trades occurred)
        self.asset_values = self.positions * prices
        
        # Update total portfolio value
        self.portfolio_value = self.current_cash + np.sum(self.asset_values)
        
        # Append to history *after* all updates for the step are done
        self.portfolio_value_history.append(self.portfolio_value)

    def _calculate_portfolio_value(self, positions: np.ndarray, prices: np.ndarray) -> float:
        """Calculate the total portfolio value."""
        return self.current_cash + np.sum(positions * prices)
        

    def _is_done(self) -> bool:
        """Check if the episode is done."""
        # Episode ends when we reach the maximum day
        if self.current_step >= self.max_step:
            return True
                
        return False

    def _update_info(self, previous_value: float, current_value: float, intended_action: np.ndarray, feasible_action: np.ndarray, reward: float, reward_components: Dict[str, float]) -> Dict[str, Any]:
        """Update the info dictionary with current state and execution details."""
        # Calculate period return
        period_return = (current_value - previous_value) / previous_value if previous_value > 1e-9 else 0
        
        # Calculate cumulative return
        cumulative_return = (current_value - self.initial_balance) / self.initial_balance
        
        # Calculate clipping
        clipping_details = intended_action - feasible_action # Difference shows how much was clipped
        
        # Get current prices
        current_prices = self._get_current_prices()
        
        # Get reward statistics
        reward_stats = self.reward_manager.get_reward_statistics()
        
        return {
            "step": self.current_step,
            "portfolio_value": current_value,
            "cash": self.current_cash,
            "positions": self.positions,
            "asset_values": self.asset_values,
            "prices": current_prices,
            "period_return": period_return,
            "cumulative_return": cumulative_return,
            "intended_action": intended_action,
            "feasible_action": feasible_action,
            "action_clipping": clipping_details,
            "reward": reward,
            "reward_components": reward_components,  # Add individual reward components
            "reward_statistics": reward_stats,  # Add reward component statistics
            "termination_reason": "" # Update if termination conditions added
        }

    def render(self, mode: str = "human") -> None:
        """Render the environment."""
        if mode == "human":
            # If visualizer is enabled, update it with the current state
            if self.visualizer is not None:
                # Get the latest info for visualization
                current_prices = self._get_current_prices()
                
                # Create visualization data
                viz_info = {
                    'portfolio_value': self.portfolio_value,
                    'positions': self.positions,
                    'action': self.info.get('feasible_action', np.zeros(self.n_assets)),
                    'returns': self.info.get('period_return', 0),
                    'reward': self.info.get('reward', 0),
                    'cash': self.current_cash,
                    'loss': self.info.get('loss', 0),  # Get from info dict, set by update_agent_info
                    'epsilon': self.info.get('epsilon', 0)  # Get from info dict, set by update_agent_info
                }
                
                # Update the visualizer
                self.visualizer.update(self.current_step, viz_info)
            else:
                # Fallback to text-based rendering if visualizer isn't initialized
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
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None
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
        if self.render_mode == "human":
            logger.info(f"Visualization enabled in human mode")

    def get_current_position(self) -> np.ndarray:
        """Get the current position of the agent."""
        return self.positions
    
    def get_current_cash(self) -> float:
        """Get the current cash of the agent."""
        return self.current_cash
    
    def get_position_limits(self) -> Dict[str, float]:
        """Get the position limits of the environment."""
        position_limits = self.constraint_manager.get_parameters("position_limits") # Expects {'min': N, 'max': M}
        return position_limits
        
    def get_action_dim(self) -> int:
        """Get the dimension of the action space."""
        return self.action_space.shape[0]
    
    def update_agent_info(self, agent_info: Dict[str, Any]) -> None:
        """
        Update the environment with agent information for visualization.
        
        Args:
            agent_info: Dictionary containing agent information (e.g., loss, epsilon)
        """
        # Update the info dictionary with agent information
        if self.visualizer is not None:
            # Store the latest agent information in the info dictionary
            # This will be used by the render method
            for key, value in agent_info.items():
                self.info[key] = value
    
    def _precompute_data_arrays(self) -> None:
        """Precompute data arrays for faster processing."""
        # Remove date column from original data if present (original data is pd.DataFrame)
        if "date" in self.raw_data.columns:
            self.raw_data = self.raw_data.drop(columns=["date"])
        if "date" in self.processed_data.columns:
            self.processed_data = self.processed_data.drop(columns=["date"])

        # Reset index, but keep the day column
        self.raw_data = self.raw_data.reset_index(drop=True)
        self.processed_data = self.processed_data.reset_index(drop=True)

        # Make data double indexed by day and ticker
        self.raw_data = self.raw_data.set_index([self.day_col, self.tic_col])
        self.processed_data = self.processed_data.set_index([self.day_col, self.tic_col])

        # Create feature indice mapping
        self.raw_data_feature_indices = {column_name: i for i, column_name in enumerate(self.raw_data.columns)}
        self.processed_data_feature_indices = {column_name: i for i, column_name in enumerate(self.processed_data.columns)}
        self._asset_to_idx = {asset: i for i, asset in enumerate(self.asset_list)}


        # Save list of columns
        self.raw_data_cols = list(self.raw_data_feature_indices.keys())
        self.processed_data_cols = list(self.processed_data_feature_indices.keys())

        # Initialize matrices (day, asset, feature)
        self.raw_matrix = np.zeros((self.max_step + 1, self.n_assets, len(self.raw_data_cols)), dtype=np.float32)
        self.processed_matrix = np.zeros((self.max_step + 1, self.n_assets, len(self.processed_data_cols)), dtype=np.float32)

        # Save list of technical and OHLCV column indices
        self.tech_col_indices = [self.processed_data_feature_indices[col] for col in self.tech_cols]
        self.ohlcv_col_indices = [self.raw_data_feature_indices[col] for col in self.ohlcv_cols]

        # Precompute the raw matrix
        for (day, ticker), group in self.raw_data.groupby([self.day_col, self.tic_col]):
            if day <= self.max_step and ticker in self.asset_list:
                asset_index = self._asset_to_idx[ticker]
                self.raw_matrix[day, asset_index, :] = group[self.raw_data_cols].values.squeeze()

        # Precompute the processed matrix
        for (day, ticker), group in self.processed_data.groupby([self.day_col, self.tic_col]):
            if day <= self.max_step and ticker in self.asset_list:
                asset_index = self._asset_to_idx[ticker]
                self.processed_matrix[day, asset_index, :] = group[self.processed_data_cols].values.squeeze()