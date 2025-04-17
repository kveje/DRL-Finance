import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from environments.trading_env import TradingEnv
from models.agents.base_agent import BaseAgent
from visualization.trading_visualizer import TradingVisualizer

class Backtester:
    """
    Handles running backtests on validation data using saved model checkpoints.
    """
    
    def __init__(
        self,
        env: TradingEnv,
        agent: BaseAgent,
        asset_names: Optional[List[str]] = None,
        visualizer: Optional[TradingVisualizer] = None,
        save_visualizations: bool = True,
        visualization_dir: str = "backtest_visualizations"
    ):
        """
        Initialize the backtester.
        
        Args:
            env: Trading environment instance
            agent: Agent instance to use for backtesting
            asset_names: List of asset names for visualization (if None, will try to get from env)
            visualizer: Optional visualizer for real-time backtesting visualization
            save_visualizations: Whether to save backtest visualizations
            visualization_dir: Directory to save backtest visualizations
        """
        self.env = env
        self.agent = agent
        self.visualizer = visualizer
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir
        
        # Try to get asset names from env or use provided names
        self.asset_names = asset_names
        if self.asset_names is None:
            # Try different possible attribute names
            if hasattr(env, 'asset_names'):
                self.asset_names = env.asset_names
            elif hasattr(env, 'symbols'):
                self.asset_names = env.symbols
            elif hasattr(env, 'tickers'):
                self.asset_names = env.tickers
            else:
                # Default to generic names based on number of assets
                n_assets = env.n_assets if hasattr(env, 'n_assets') else 1
                self.asset_names = [f"Asset {i+1}" for i in range(n_assets)]
        
        # Create visualization directory if needed
        if self.save_visualizations:
            Path(self.visualization_dir).mkdir(parents=True, exist_ok=True)
    
    def run_backtest(
        self,
        model_state_dict: Dict[str, Any],
        deterministic: bool = True,
        episode_id: Optional[int] = None,
        save_visualization: Optional[bool] = None,
        visualization_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest using the provided model state.
        
        Args:
            model_state_dict: Model state dictionary to load
            deterministic: Whether to use deterministic actions
            episode_id: Optional episode ID for naming the visualization
            save_visualization: Whether to save visualization (overrides instance setting)
            visualization_filename: Optional custom filename for the visualization
            
        Returns:
            Dictionary containing backtest results
        """
        # Load model state
        self.agent.load_state_dict(model_state_dict)
        
        # Reset environment
        obs = self.env.reset()
        done = False
        
        # Get initial state info
        current_position = self.env.get_current_position().copy()  # Start with initial position (zeros)
        
        # Initialize tracking
        portfolio_values = [self.env.initial_balance]
        actions = []
        positions = [current_position]  # Store initial position
        rewards = []
        cash_values = [self.env.initial_balance]  # Start with all cash
        asset_prices = []  # Track asset prices for reference
        
        # Track current prices
        current_prices = self._get_env_prices()
        if current_prices is not None:
            asset_prices.append(current_prices)
        
        # Run backtest
        step = 0
        while not done:
            # Get action from agent
            action = self.agent.get_intended_action(obs, current_position, deterministic=deterministic)
            
            # Take step
            next_obs, reward, done, info = self.env.step(action)
            
            # Track current position after action - make a copy to ensure independent storage
            current_position = self.env.get_current_position().copy()
            
            # Record data
            portfolio_values.append(info['portfolio_value'])
            actions.append(action)
            positions.append(current_position)  # Use the actual position from env (as a copy)
            rewards.append(reward)
            
            # Record cash if available
            if 'cash' in info:
                cash_values.append(info['cash'])
            
            # Record asset prices if available
            if 'prices' in info:
                asset_prices.append(info['prices'].copy())
                
            # Update observation for next iteration
            obs = next_obs
            step += 1
        
        # Make sure positions is a proper array for visualization
        positions_array = np.array(positions)
        
        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - self.env.initial_balance) / self.env.initial_balance
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Prepare results
        results = {
            'portfolio_values': portfolio_values,
            'actions': actions,
            'positions': positions_array,  # Store as numpy array
            'rewards': rewards,
            'returns': returns,
            'cash': cash_values,
            'prices': asset_prices if asset_prices else None,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': portfolio_values[-1],
            'initial_portfolio_value': self.env.initial_balance,
            'num_steps': len(portfolio_values) - 1,
            'asset_names': self.asset_names
        }
        
        # Calculate asset-specific returns if prices are available
        if asset_prices and len(asset_prices) > 1:
            # Convert to numpy array for easier calculations
            prices_array = np.array(asset_prices)
            
            # Calculate asset returns (day-to-day percentage changes)
            asset_returns = np.diff(prices_array, axis=0) / prices_array[:-1]
            
            # Add to results
            results['asset_returns'] = asset_returns
            results['asset_prices'] = prices_array
        
        # Save visualization if requested
        should_save = self.save_visualizations if save_visualization is None else save_visualization
        if should_save:
            visualization_path = self._save_backtest_visualization(results, episode_id, visualization_filename)
            # Add visualization path to results
            results['visualization_path'] = visualization_path
        
        return results
    
    def _save_backtest_visualization(
        self, 
        results: Dict[str, Any],
        episode_id: Optional[int] = None,
        custom_filename: Optional[str] = None
    ):
        """
        Save a visualization of the backtest results.
        
        Args:
            results: Dictionary with backtest results
            episode_id: Optional episode ID for the filename
            custom_filename: Optional custom filename
        """
        # Create filename
        if custom_filename:
            filename = custom_filename
        else:
            timestamp = results.get('timestamp', '')
            episode_str = f"_ep{episode_id}" if episode_id is not None else ""
            filename = f"backtest{episode_str}_{timestamp}.png" if timestamp else f"backtest{episode_str}.png"
        
        # Ensure directory exists
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Full path for visualization
        filepath = os.path.join(self.visualization_dir, filename)
        
        # Generate title
        title = f"Backtest Results"
        if episode_id is not None:
            title += f" - Episode {episode_id}"
        
        # Use the static method to create and save visualization
        TradingVisualizer.create_and_save_backtest_visualization(
            asset_names=self.asset_names,
            backtest_data=results,
            filename=filepath,
            title=title
        )
        
        return filepath
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate the Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        if len(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate the maximum drawdown."""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
    
    def _get_env_prices(self) -> Optional[np.ndarray]:
        """Helper method to get current prices from environment if available"""
        if hasattr(self.env, '_get_current_prices'):
            return self.env._get_current_prices()
        elif hasattr(self.env, 'latest_prices'):
            return self.env.latest_prices
        return None 