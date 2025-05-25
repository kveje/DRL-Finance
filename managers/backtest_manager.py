"""Enhanced backtesting manager for training and validation data"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict

from models.agents.base_agent import BaseAgent
from environments.trading_env import TradingEnv
from utils.logger import Logger
from managers.visualization_manager import VisualizationManager

class BacktestManager:
    """
    Backtesting manager that supports:
    - Training data backtesting
    - Validation data backtesting
    - Performance metrics calculation
    - Detailed visualization
    """
    
    def __init__(
        self,
        train_env: TradingEnv,
        val_env: TradingEnv,
        agent: BaseAgent,
        save_dir: str = "backtest_results",
        save_visualizations: bool = True,
        asset_names: Optional[List[str]] = None,
        risk_free_rate: float = 0.02,
        transaction_cost: float = 0.001,
        initial_balance: float = 100000.0,
    ):
        """
        Initialize the backtesting manager.
        
        Args:
            train_env: Training environment instance
            val_env: Validation environment instance
            agent: Agent instance to use for backtesting
            save_dir: Directory to save backtest results
            save_visualizations: Whether to save backtest visualizations
            asset_names: List of asset names for visualization
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            transaction_cost: Transaction cost as a fraction
            initial_balance: Initial portfolio balance
        """
        self.train_env = train_env
        self.val_env = val_env
        self.agent = agent
        self.save_dir = Path(save_dir)
        self.save_visualizations = save_visualizations
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        
        # Set up asset names
        self.asset_names = asset_names
        if self.asset_names is None:
            # Try to get asset names from environments
            if hasattr(train_env, 'asset_names'):
                self.asset_names = train_env.asset_names
            elif hasattr(train_env, 'symbols'):
                self.asset_names = train_env.symbols
            elif hasattr(train_env, 'tickers'):
                self.asset_names = train_env.tickers
            else:
                n_assets = train_env.n_assets if hasattr(train_env, 'n_assets') else 1
                self.asset_names = [f"Asset {i+1}" for i in range(n_assets)]
        
        # Create save directory
        if self.save_visualizations:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize logger
        self.logger = Logger.get_logger("backtest_manager")
        
        # Initialize visualizer
        self.visualizer = VisualizationManager()
    
    def run_full_backtest(
        self,
        model_state_dict: Dict[str, Any],
        deterministic: bool = True,
        episode_id: Optional[int] = None,
        include_train: bool = True,
        include_val: bool = True,
        save_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete backtest on both training and validation data.
        
        Args:
            model_state_dict: Model state dictionary to load
            deterministic: Whether to use deterministic actions
            episode_id: Optional episode ID for naming
            include_train: Whether to include training data backtest
            include_val: Whether to include validation data backtest
            save_visualization: Whether to save visualizations
            
        Returns:
            Dictionary containing backtest results
        """
        results = {}
        
        # Load model state
        self.agent.load_state_dict(model_state_dict)
        
        # Run validation backtest
        if include_val:
            val_results = self._run_single_backtest(
                env=self.val_env,
                data_type="validation",
                deterministic=deterministic,
                episode_id=episode_id,
                save_visualization=save_visualization
            )
            results["validation"] = val_results
        
        # Run training backtest
        if include_train:
            train_results = self._run_single_backtest(
                env=self.train_env,
                data_type="training",
                deterministic=deterministic,
                episode_id=episode_id,
                save_visualization=save_visualization
            )
            results["training"] = train_results
        
        # Save comprehensive results
        self._save_backtest_results(results, episode_id)
        
        return results
    
    def _run_single_backtest(
        self,
        env: TradingEnv,
        data_type: str,
        deterministic: bool = True,
        episode_id: Optional[int] = None,
        save_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        Run a backtest on a single environment.
        
        Args:
            env: Environment to backtest on
            data_type: Type of data ("training" or "validation")
            deterministic: Whether to use deterministic actions
            episode_id: Optional episode ID for naming
            save_visualization: Whether to save visualization
            
        Returns:
            Dictionary containing backtest results
        """
        # Reset environment
        obs = env.reset()
        done = False
        
        # Initialize tracking
        portfolio_values = [self.initial_balance]
        actions = []
        positions = [env.get_current_position().copy()]
        rewards = []
        reward_components = defaultdict(list)  # Track individual reward components
        cash_values = [self.initial_balance]
        asset_prices = []
        trade_history = []
        
        # Get initial prices
        current_prices = self._get_env_prices(env)
        if current_prices is not None:
            asset_prices.append(current_prices)
        
        # Run backtest
        step = 0
        while not done:
            # Get action from agent
            scaled_action, action_choice = self.agent.get_intended_action(obs, positions[-1], deterministic=deterministic)
            
            # Take step
            next_obs, reward, done, info = env.step(scaled_action)
            
            # Track current position
            current_position = env.get_current_position().copy()
            
            # Record data
            portfolio_values.append(info['portfolio_value'])
            actions.append(scaled_action)
            positions.append(current_position)
            rewards.append(reward)
            
            # Record reward components if available
            if 'reward_components' in info:
                for component_name, value in info['reward_components'].items():
                    reward_components[component_name].append(value)
            
            # Record cash if available
            if 'cash' in info:
                cash_values.append(info['cash'])
            
            # Record asset prices if available
            if 'prices' in info:
                asset_prices.append(info['prices'].copy())
            
            # Record trade if position changed
            if len(positions) > 1 and not np.array_equal(positions[-1], positions[-2]):
                trade = {
                    'step': step,
                    'action': scaled_action,
                    'position_before': positions[-2],
                    'position_after': positions[-1],
                    'portfolio_value': info['portfolio_value'],
                    'cash': info.get('cash', None),
                    'prices': info.get('prices', None)
                }
                trade_history.append(trade)
            
            # Update observation
            obs = next_obs
            step += 1
        
        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        win_rate = self._calculate_win_rate(trade_history, portfolio_values) if trade_history else 0
        avg_trade_return = np.mean([t['portfolio_value'] / portfolio_values[t['step']] - 1 for t in trade_history]) if trade_history else 0
        
        # Prepare results
        results = {
            'portfolio_values': np.array(portfolio_values),
            'actions': np.array(actions),
            'positions': np.array(positions),
            'rewards': np.array(rewards),
            'reward_components': dict(reward_components),  # Add reward components to results
            'returns': np.array(returns),
            'cash': np.array(cash_values),
            'prices': np.array(asset_prices) if asset_prices else None,
            'trade_history': trade_history,
            'metrics': {
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'max_drawdown': float(max_drawdown),
                'volatility': float(volatility),
                'avg_trade_return': float(avg_trade_return),
                'num_trades': int(len(trade_history)),
                'win_rate': float(win_rate)
            },
            'final_portfolio_value': float(portfolio_values[-1]),
            'initial_portfolio_value': float(self.initial_balance),
            'num_steps': int(len(portfolio_values) - 1),
            'asset_names': self.asset_names
        }
        
        # Save visualization if requested
        if save_visualization:
            self._save_backtest_visualization(
                results=results,
                data_type=data_type,
                episode_id=episode_id
            )
        
        return results
    
    def _save_backtest_visualization(
        self,
        results: Dict[str, Any],
        data_type: str,
        episode_id: Optional[int] = None
    ) -> str:
        """
        Save visualization of backtest results.
        
        Args:
            results: Dictionary with backtest results
            data_type: Type of data ("training" or "validation")
            episode_id: Optional episode ID for naming
            
        Returns:
            Path to saved visualization
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_str = f"_ep{episode_id}" if episode_id is not None else ""
        filename = f"backtest_{data_type}{episode_str}_{timestamp}.png"
        
        # Ensure save_dir is a Path object
        save_dir = Path(self.save_dir)
        
        # Full path for visualization
        filepath = save_dir / filename
        
        # Create visualization
        self.visualizer.create_and_save_backtest_visualization(
            asset_names=self.asset_names,
            backtest_data=results,
            filename=str(filepath),
            title=f"Backtest Results - {data_type.capitalize()}"
        )
        
        return str(filepath)
    
    def _save_backtest_results(
        self,
        results: Dict[str, Any],
        episode_id: Optional[int] = None
    ) -> None:
        """
        Save backtest results to disk.
        
        Args:
            results: Dictionary with backtest results
            episode_id: Optional episode ID for naming
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_str = f"_ep{episode_id}" if episode_id is not None else ""
        filename = f"backtest_results{episode_str}_{timestamp}.json"
        
        # Ensure save_dir is a Path object
        save_dir = Path(self.save_dir)
        
        # Save to file
        with open(save_dir / filename, 'w') as f:
            json.dump(self._convert_to_serializable(results), f, indent=4)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate the Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate/252  # Daily risk-free rate
        if len(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)  # Annualized
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate the Sortino ratio."""
        excess_returns = returns - self.risk_free_rate/252  # Daily risk-free rate
        if len(excess_returns) == 0:
            return 0.0
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        return np.mean(excess_returns) / (np.std(downside_returns) + 1e-8) * np.sqrt(252)  # Annualized
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate the Calmar ratio."""
        if max_drawdown == 0:
            return float('inf')
        annual_return = np.mean(returns) * 252
        return annual_return / max_drawdown
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate the maximum drawdown."""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
    
    def _calculate_win_rate(self, trade_history: List[Dict[str, Any]], portfolio_values: List[float]) -> float:
        """
        Calculate the win rate from trade history.
        
        Args:
            trade_history: List of trade dictionaries
            portfolio_values: List of portfolio values
            
        Returns:
            Win rate as a float between 0 and 1
        """
        if not trade_history:
            return 0.0
            
        winning_trades = 0
        for trade in trade_history:
            # Get portfolio value at trade step
            trade_value = portfolio_values[trade['step']]
            # Get next portfolio value
            next_value = portfolio_values[trade['step'] + 1] if trade['step'] + 1 < len(portfolio_values) else trade['portfolio_value']
            # Count as win if next value is higher
            if next_value > trade_value:
                winning_trades += 1
                
        return winning_trades / len(trade_history)
    
    def _get_env_prices(self, env: TradingEnv) -> Optional[np.ndarray]:
        """Helper method to get current prices from environment."""
        if hasattr(env, '_get_current_prices'):
            return env._get_current_prices()
        return None
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_serializable(i) for i in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                return str(obj)
            except:
                return None 