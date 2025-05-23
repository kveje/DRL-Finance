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
        columns: Dict[str, Union[str, List[str]]], # {ticker: "ticker", price: "close", day: "day", ohlcv: ["open", "high", "low", "close", "volume"], tech_cols: ["RSI", "MACD", "Bollinger Bands"]}
        env_params: Dict[str, Any] = {}, # {initial_balance: 100000.0, window_size: 10}
        friction_params: Dict[str, Dict[str, float]] = {}, # {slippage: {slippage_mean: 0.0, slippage_std: 0.001}, commission: {commission_rate: 0.001}}
        reward_params: Tuple[str, Dict[str, Any]] = ("returns_based", {"scale": 1.0}), # (reward_type, reward_params) e.g. ("returns_based", {"scale": 1.0})
        constraint_params: Dict[str, Dict[str, float]] = {}, # {position_limits: {min: -1000, max: 1000}}
        processor_configs: List[Dict[str, Any]] = None, # List of processor configurations
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

        # Data setup - ensure day is the index
        if processed_data.index.name != self.day_col:
            logger.info(f"Setting {self.day_col} as index for processed_data")
            self.processed_data: pd.DataFrame = processed_data.set_index(self.day_col)
        else:
            self.processed_data: pd.DataFrame = processed_data

        if raw_data.index.name != self.day_col:
            logger.info(f"Setting {self.day_col} as index for raw_data")
            self.raw_data: pd.DataFrame = raw_data.set_index(self.day_col)
        else:
            self.raw_data: pd.DataFrame = raw_data

        # Save columns
        self.columns = columns

        # Get max step
        self.max_step = self.processed_data.index.max()

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
        self.portfolio_value = self.initial_balance

        # Initialize managers for market frictions, constraints, and rewards
        self.market_frictions = MarketFrictionManager(friction_params)
        self.constraint_manager = ConstraintManager(constraint_params)
        self.reward_manager = RewardManager(reward_params)

        # Save the parameters
        self.env_params = env_params
        self.friction_params = friction_params
        self.constraint_params = constraint_params
        self.reward_params = reward_params

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

        # Initialize the composite processor
        if processor_configs is None:
            # Default processor configuration if none provided
            processor_configs = [
                {
                    'type': 'price',
                    'data_name': 'market_data',
                    'kwargs': {
                        'window_size': self.window_size,
                        'asset_list': self.asset_list
                    }
                },
                {
                    'type': 'cash',
                    'data_name': 'cash_data',
                    'kwargs': {
                        'cash_limit': self.constraint_manager.get_parameters("cash_limit")["max"]
                    }
                },
                {
                    'type': 'position',
                    'data_name': 'position_data',
                    'kwargs': {
                        'position_limits': self.constraint_manager.get_parameters("position_limits"),
                        'asset_list': self.asset_list
                    }
                }
            ]
        
        self.processor = CompositeProcessor(processor_configs)
        
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
        observation = self._get_observation()
        
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
        feasible_action = self._apply_constraints_and_get_feasible_action(
            intended_action=intended_action,
            current_positions=current_positions,
            current_cash=current_cash,
            adjusted_prices=adjusted_prices
        )

        # --- 4. Execute Trades ---
        # Use the feasible_action and adjusted_prices for execution
        self._execute_trades(feasible_action, adjusted_prices)

        # --- 5. Post-computation and State Update ---
        # Portfolio value is updated within _execute_trades based on feasible action
        self.current_step += 1

        # --- 6. Calculate Reward ---
        # Reward is based on the change in portfolio value resulting from the *feasible_action*
        reward = self.reward_manager.calculate(
            portfolio_value=self.portfolio_value,
            previous_portfolio_value=previous_portfolio_value,
        )

        # Add penalty for constraint violations
        penalty = np.sum(np.abs(feasible_action - intended_action)) * 0.001
        reward -= penalty

        # --- 7. Check if episode is done ---
        self.done = self._is_done()

        # --- 8. Update Info Dictionary ---
        # Include info about the trade execution if desired
        self.info = self._update_info(
            previous_value=previous_portfolio_value,
            current_value=self.portfolio_value,
            intended_action=intended_action,
            feasible_action=feasible_action,
            reward=reward
        )

        # --- 9. Get Next Observation ---
        observation = self._get_observation()

        return observation, reward, self.done, self.info

    def _apply_constraints_and_get_feasible_action(
        self,
        intended_action: np.ndarray,
        current_positions: np.ndarray,
        current_cash: float,
        adjusted_prices: np.ndarray
    ) -> np.ndarray:
        """
        Adjusts the intended action to comply with position and cash constraints.

        Args:
            intended_action: Desired change in shares for each asset.
            current_positions: Current shares held for each asset.
            current_cash: Current cash balance.
            adjusted_prices: Prices after accounting for slippage.

        Returns:
            The feasible action (change in shares) that respects constraints.
        """
        feasible_action = intended_action.copy()

        # --- 1. Position Limit Constraints ---
        pos_limits = self.constraint_manager.get_parameters("position_limits") # Expects {'min': N, 'max': M}
        if pos_limits:
            feasible_action = np.clip(feasible_action, pos_limits["min"] - current_positions, pos_limits["max"] - current_positions)

        # --- 2. Cash Limit Constraints (Overall Portfolio) ---
        # Net cash required for the trade
        net_cash_flow_required = np.sum(feasible_action * adjusted_prices)

        # Check against minimum cash limit
        cash_limits = self.constraint_manager.get_parameters("cash_limit") # Expects {'min': X, 'max': Y}
        available_cash_for_trade = current_cash - cash_limits['min']

        if net_cash_flow_required > available_cash_for_trade:
            # Not enough cash. Need to scale down buys. Sells help cash flow.
            scaling_factor = available_cash_for_trade / net_cash_flow_required

            # Apply scaling factor to buys
            buy_mask = feasible_action > 0
            feasible_action[buy_mask] = np.floor(feasible_action[buy_mask] * scaling_factor)

        return feasible_action.astype(int) # Ensure same dtype

    def _get_current_prices(self) -> np.ndarray:
        """
        Get current prices for all assets.
        
        Returns:
            np.ndarray: Array of current prices for all assets
        """
        # Get current day's data for all assets
        current_data = self.raw_data.loc[self.current_step]
        
        # Extract prices for each asset in the correct order
        prices = np.array([
            current_data[current_data[self.tic_col] == asset][self.price_col].iloc[0]
            for asset in self.asset_list
        ], dtype=np.float32)
        
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

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation for the agent using the composite processor.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with observation data
        """
        # Prepare data for processors
        data = {
            'market_data': self.processed_data,
            'cash_data': self.current_cash,
            'position_data': self.positions
        }
        
        # Process data using the composite processor
        observation = self.processor.process(data, self.current_step)
        
        return observation

    def _is_done(self) -> bool:
        """Check if the episode is done."""
        # Episode ends when we reach the maximum day
        if self.current_step >= self.max_step:
            return True
            
        # TODO: Might add other termination conditions here
        
        return False

    def _update_info(self, previous_value: float, current_value: float, intended_action: np.ndarray, feasible_action: np.ndarray, reward: float) -> Dict[str, Any]:
        """Update the info dictionary with current state and execution details."""
        # Calculate period return
        period_return = (current_value - previous_value) / previous_value if previous_value > 1e-9 else 0
        
        # Calculate cumulative return
        cumulative_return = (current_value - self.initial_balance) / self.initial_balance
        
        # Calculate clipping
        clipping_details = intended_action - feasible_action # Difference shows how much was clipped
        
        # Get current prices
        current_prices = self._get_current_prices()
        
        return {
            "step": self.current_step,
            "portfolio_value": current_value,
            "cash": self.current_cash,
            "positions": self.positions.copy(), # Return copy
            "asset_values": self.asset_values.copy(), # Return copy
            "prices": current_prices.copy(), # Include current prices for backtesting
            "period_return": period_return,
            "cumulative_return": cumulative_return,
            "intended_action": intended_action.copy(),
            "feasible_action": feasible_action.copy(),
            "action_clipping": clipping_details.copy(),
            "reward": reward,
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