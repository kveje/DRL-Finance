import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime

class TradingVisualizer:
    """
    Visualizer for trading environments that provides real-time visualization
    of the agent's performance during training.
    """
    
    def __init__(
        self,
        asset_names: List[str],
        window_size: int = 100,  # Number of steps to show in the visualization
        update_interval: int = 100  # Update visualization every N steps
    ):
        """
        Initialize the trading visualizer.
        
        Args:
            asset_names: List of asset names
            window_size: Number of steps to show in the visualization
            update_interval: Update visualization every N steps
        """
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.window_size = window_size
        self.update_interval = update_interval
        self.update_counter = 0  # Counter to track when to update
        
        # Initialize data storage with fixed-size deques
        self.portfolio_values = []
        self.positions = []
        self.actions = []
        self.returns = []
        self.dates = []
        self.losses = []
        self.cash = []
        self.epsilon = []
        self.rewards = []
        
        # Initialize figure and subplots
        self.fig = plt.figure(figsize=(18, 12))
        self.gs = GridSpec(4, 2, figure=self.fig)
        
        # Create subplots
        self.portfolio_ax = self.fig.add_subplot(self.gs[0, :])
        self.positions_ax = self.fig.add_subplot(self.gs[1, 0])
        self.cash_ax = self.fig.add_subplot(self.gs[1, 1])
        self.actions_ax = self.fig.add_subplot(self.gs[2, 0])
        self.returns_ax = self.fig.add_subplot(self.gs[2, 1])
        self.loss_ax = self.fig.add_subplot(self.gs[3, 0])
        self.metrics_ax = self.fig.add_subplot(self.gs[3, 1])
        
        # Initialize lines
        self.portfolio_line, = self.portfolio_ax.plot([], [], 'b-', label='Portfolio Value')
        self.position_lines = [self.positions_ax.plot([], [], label=asset)[0] for asset in asset_names]
        self.cash_line, = self.cash_ax.plot([], [], 'g-', label='Cash')
        self.action_lines = [self.actions_ax.plot([], [], label=asset)[0] for asset in asset_names]
        self.returns_line, = self.returns_ax.plot([], [], 'r-', label='Returns %')
        self.loss_line, = self.loss_ax.plot([], [], 'c-', label='Loss')
        self.epsilon_line, = self.metrics_ax.plot([], [], 'm-', label='Epsilon')
        self.reward_line, = self.metrics_ax.plot([], [], 'y-', label='Reward')
        
        # Set up axes
        self._setup_axes()
        
        # Calculate metrics
        self.last_portfolio_value = None
        
        # Show the plot
        plt.ion()
        plt.tight_layout()
        plt.show(block=False)
    
    def _setup_axes(self):
        """Set up the axes with proper labels and formatting."""
        # Portfolio value plot
        self.portfolio_ax.set_title('Portfolio Value Over Time')
        self.portfolio_ax.set_xlabel('Time')
        self.portfolio_ax.set_ylabel('Value ($)')
        self.portfolio_ax.grid(True)
        
        # Positions plot
        self.positions_ax.set_title('Asset Positions')
        self.positions_ax.set_xlabel('Time')
        self.positions_ax.set_ylabel('Position Size')
        self.positions_ax.grid(True)
        
        # Cash plot
        self.cash_ax.set_title('Cash Balance')
        self.cash_ax.set_xlabel('Time')
        self.cash_ax.set_ylabel('Cash ($)')
        self.cash_ax.grid(True)
        
        # Actions plot
        self.actions_ax.set_title('Trading Actions')
        self.actions_ax.set_xlabel('Time')
        self.actions_ax.set_ylabel('Action Value')
        self.actions_ax.grid(True)
        
        # Rewards plot
        self.returns_ax.set_title('Returns Over Time')
        self.returns_ax.set_xlabel('Time')
        self.returns_ax.set_ylabel('Returns %')
        self.returns_ax.grid(True)
        
        # Loss plot
        self.loss_ax.set_title('Training Loss')
        self.loss_ax.set_xlabel('Time')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True)
        
        # Metrics plot
        self.metrics_ax.set_title('Training Metrics')
        self.metrics_ax.set_xlabel('Time')
        self.metrics_ax.set_ylabel('Value')
        self.metrics_ax.grid(True)
        self.metrics_ax.legend()
    
    def update(self, step: int, info: Dict):
        """
        Update the visualization with new data.
        
        Args:
            step: Current step number
            info: Dictionary containing environment information
        """
        # Increment counter and only update visualization at specified intervals
        self.update_counter += 1
        if self.update_counter % self.update_interval != 0:
            return
            
        # Store new data
        portfolio_value = info['portfolio_value']
        self.portfolio_values.append(portfolio_value)
        self.positions.append(info['positions'])
        self.actions.append(info.get('action', np.zeros(self.n_assets)))
        self.returns.append(info.get('returns', 0))
        self.dates.append(step)
        
        # Store loss information
        self.losses.append(info.get('loss', 0))
        
        # Store cash information
        self.cash.append(info.get('cash', 0))
        
        # Store epsilon (exploration rate)
        self.epsilon.append(info.get('epsilon', 0))
        self.rewards.append(info.get('reward', 0))

        self.last_portfolio_value = portfolio_value
        
        # Keep only the last window_size steps
        if len(self.portfolio_values) > self.window_size:
            self.portfolio_values = self.portfolio_values[-self.window_size:]
            self.positions = self.positions[-self.window_size:]
            self.actions = self.actions[-self.window_size:]
            self.returns = self.returns[-self.window_size:]
            self.dates = self.dates[-self.window_size:]
            self.losses = self.losses[-self.window_size:]
            self.cash = self.cash[-self.window_size:]
            self.epsilon = self.epsilon[-self.window_size:]
            self.rewards = self.rewards[-self.window_size:]
        
        # Update plots
        self._update_plots()
    
    def _update_plots(self):
        """Update all plots with current data."""
        if not self.portfolio_values:
            return  # Skip update if no data available
            
        # Update portfolio value plot
        self.portfolio_line.set_data(self.dates, self.portfolio_values)
        self.portfolio_ax.relim()
        self.portfolio_ax.autoscale_view()
        
        # Update positions plot
        positions_array = np.array(self.positions)
        for i, line in enumerate(self.position_lines):
            if i < positions_array.shape[1]:  # Ensure we don't exceed array bounds
                line.set_data(self.dates, positions_array[:, i])
        self.positions_ax.relim()
        self.positions_ax.autoscale_view()
        
        # Update cash plot
        self.cash_line.set_data(self.dates, self.cash)
        self.cash_ax.relim()
        self.cash_ax.autoscale_view()
        
        # Update actions plot
        actions_array = np.array(self.actions)
        for i, line in enumerate(self.action_lines):
            if i < actions_array.shape[1]:  # Ensure we don't exceed array bounds
                line.set_data(self.dates, actions_array[:, i])
        self.actions_ax.relim()
        self.actions_ax.autoscale_view()
        
        # Update rewards plot
        self.returns_line.set_data(self.dates, self.returns)
        self.returns_ax.relim()
        self.returns_ax.autoscale_view()
        
        # Update loss plot
        self.loss_line.set_data(self.dates, self.losses)
        self.loss_ax.relim()
        self.loss_ax.autoscale_view()
        
        # Update metrics plot
        self.epsilon_line.set_data(self.dates, self.epsilon)
        self.reward_line.set_data(self.dates, self.rewards)
        self.metrics_ax.relim()
        self.metrics_ax.autoscale_view()
        
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save_figure(self, filename: str):
        """
        Save the current figure to a file.
        
        Args:
            filename: Path to save the figure
        """
        self.fig.savefig(filename)
    
    def close(self):
        """Close the visualization."""
        plt.close(self.fig)
        plt.ioff()
        
    @staticmethod
    def create_and_save_backtest_visualization(
        asset_names: List[str],
        backtest_data: Dict[str, Any],
        filename: str,
        title: Optional[str] = None
    ):
        """
        Create and save a visualization of backtest results without rendering.
        
        Args:
            asset_names: List of asset names
            backtest_data: Dictionary containing backtest results
            filename: Path to save the visualization
            title: Optional title for the visualization
        """
        # Extract data from backtest results
        portfolio_values = backtest_data.get('portfolio_values', [])
        positions = backtest_data.get('positions', [])
        actions = backtest_data.get('actions', [])
        returns = backtest_data.get('returns', [])
        rewards = backtest_data.get('rewards', [])
        asset_returns = backtest_data.get('asset_returns', None)
        asset_prices = backtest_data.get('asset_prices', None)
        dates = list(range(len(portfolio_values)))
        
        # Use asset names from backtest data if available
        if 'asset_names' in backtest_data:
            asset_names = backtest_data['asset_names']
            
        # Ensure positions data is properly shaped
        positions_array = None
        if isinstance(positions, np.ndarray):
            positions_array = positions
        elif positions and len(positions) > 0:
            try:
                positions_array = np.array(positions)
            except:
                positions_array = None
        
        # Ensure actions data is properly shaped
        actions_array = None
        if isinstance(actions, np.ndarray):
            actions_array = actions
        elif actions and len(actions) > 0:
            try:
                actions_array = np.array(actions)
            except:
                actions_array = None
        
        # Check dimensions and adjust asset_names if needed
        n_positions = 0
        if positions_array is not None:
            if len(positions_array.shape) > 1:
                n_positions = positions_array.shape[1]
            else:
                # Single asset case
                n_positions = 1
        
        if n_positions > 0 and len(asset_names) != n_positions:
            # If there's a mismatch, create generic names
            asset_names = [f"Asset {i+1}" for i in range(n_positions)]
        
        # Create figure and subplots (non-interactive)
        plt.ioff()
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(4, 2, figure=fig)
        
        # Create subplots
        portfolio_ax = fig.add_subplot(gs[0, :])
        positions_ax = fig.add_subplot(gs[1, 0])
        cash_ax = fig.add_subplot(gs[1, 1])
        actions_ax = fig.add_subplot(gs[2, 0])
        returns_ax = fig.add_subplot(gs[2, 1])
        metrics_ax = fig.add_subplot(gs[3, :])
        
        # Set main title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        
        # Plot portfolio values
        portfolio_ax.plot(dates, portfolio_values, 'b-', label='Portfolio Value')
        portfolio_ax.set_title('Portfolio Value Over Time')
        portfolio_ax.set_xlabel('Time Step')
        portfolio_ax.set_ylabel('Value ($)')
        portfolio_ax.grid(True)
        
        # Plot positions - be extra careful here to show all data correctly
        if positions_array is not None and positions_array.size > 0:
            # Check if positions is a 2D array (multiple assets)
            if len(positions_array.shape) > 1 and positions_array.shape[1] > 0:
                for i, asset in enumerate(asset_names):
                    if i < positions_array.shape[1]:  # Ensure we don't exceed array bounds
                        # Get position data for this asset
                        asset_pos = positions_array[:, i]
                        
                        # Plot with offset to ensure visibility
                        positions_ax.plot(dates, asset_pos, '-o', markersize=3, label=asset)
            else:
                # Single asset case
                if len(positions_array.shape) > 1:
                    # 2D array with one column
                    positions_ax.plot(dates, positions_array[:, 0], '-o', markersize=3, label=asset_names[0])
                else:
                    # 1D array
                    positions_ax.plot(dates, positions_array, '-o', markersize=3, label=asset_names[0])
        
        positions_ax.set_title('Asset Positions')
        positions_ax.set_xlabel('Time Step')
        positions_ax.set_ylabel('Position Size')
        positions_ax.grid(True)
        
        # Plot cash (if available)
        cash = []
        if 'cash' in backtest_data and backtest_data['cash']:
            cash = backtest_data['cash']
        elif portfolio_values and positions_array is not None and positions_array.size > 0:
            # Estimate cash if not provided directly
            if asset_prices is not None:
                # Calculate position values using actual prices
                prices = np.array(asset_prices)
                if len(positions_array.shape) > 1 and len(prices.shape) > 1:
                    # Multiple assets case
                    position_values = np.sum(positions_array * prices, axis=1)
                else:
                    # Single asset case
                    position_values = positions_array * prices
                
                # Cash is portfolio value minus position values
                cash = [portfolio_values[0]] + [portfolio_values[i+1] - position_values[i] for i in range(len(position_values))]
            else:
                # Sum position values (this is approximate)
                if len(positions_array.shape) > 1:
                    position_values = np.sum(positions_array, axis=1)
                else:
                    position_values = positions_array
                    
                # Cash is portfolio value minus position values
                cash = [portfolio_values[0]] + [portfolio_values[i+1] - position_values[i] for i in range(len(position_values))]
        
        if cash:
            cash_ax.plot(dates[:len(cash)], cash, 'g-', label='Cash')
            cash_ax.set_title('Cash Balance')
            cash_ax.set_xlabel('Time Step')
            cash_ax.set_ylabel('Cash ($)')
            cash_ax.grid(True)
        
        # Plot actions
        if actions_array is not None and actions_array.size > 0:
            # Check if actions is a 2D array (multiple assets)
            if len(actions_array.shape) > 1 and actions_array.shape[1] > 1:
                for i, asset in enumerate(asset_names):
                    if i < actions_array.shape[1]:  # Ensure we don't exceed array bounds
                        actions_ax.plot(dates[1:], actions_array[:, i], label=asset)
            else:
                # Single asset case
                if len(actions_array.shape) > 1:
                    # 2D array with one column
                    actions_ax.plot(dates[1:], actions_array[:, 0], label=asset_names[0])
                else:
                    # 1D array
                    actions_ax.plot(dates[1:], actions_array, label=asset_names[0])
                    
        actions_ax.set_title('Trading Actions')
        actions_ax.set_xlabel('Time Step')
        actions_ax.set_ylabel('Action Value')
        actions_ax.grid(True)
        
        # Plot portfolio returns and asset-specific returns if available
        if isinstance(returns, np.ndarray) and returns.size > 0:
            # Handle numpy array
            cumulative_returns = np.cumprod(1 + returns) - 1
            returns_ax.plot(dates[1:len(returns)+1], cumulative_returns * 100, 'r-', label='Portfolio Returns (%)')
            returns_ax.plot(dates[1:len(returns)+1], returns * 100, 'g-', alpha=0.3, label='Step Returns (%)')
        elif isinstance(returns, list) and len(returns) > 0:
            # Handle list
            returns_array = np.array(returns)
            cumulative_returns = np.cumprod(1 + returns_array) - 1
            returns_ax.plot(dates[1:len(returns)+1], cumulative_returns * 100, 'r-', label='Portfolio Returns (%)')
            returns_ax.plot(dates[1:len(returns)+1], returns_array * 100, 'g-', alpha=0.3, label='Step Returns (%)')
            
        # Plot asset-specific returns if available
        if asset_returns is not None and isinstance(asset_returns, np.ndarray) and asset_returns.size > 0:
            # Plot cumulative returns for each asset
            if len(asset_returns.shape) > 1 and asset_returns.shape[1] > 1:
                # Multiple assets
                cum_asset_returns = np.cumprod(1 + asset_returns, axis=0) - 1
                for i, asset in enumerate(asset_names):
                    if i < asset_returns.shape[1]:
                        returns_ax.plot(dates[1:len(asset_returns)+1], cum_asset_returns[:, i] * 100, 
                                     '--', alpha=0.7, label=f'{asset} Returns (%)')
            else:
                # Single asset
                if len(asset_returns.shape) > 1:
                    cum_asset_returns = np.cumprod(1 + asset_returns[:, 0]) - 1
                    returns_ax.plot(dates[1:len(asset_returns)+1], cum_asset_returns * 100, 
                                 '--', alpha=0.7, label=f'{asset_names[0]} Returns (%)')
                else:
                    cum_asset_returns = np.cumprod(1 + asset_returns) - 1
                    returns_ax.plot(dates[1:len(asset_returns)+1], cum_asset_returns * 100, 
                                 '--', alpha=0.7, label=f'{asset_names[0]} Returns (%)')
                
        returns_ax.set_title('Returns Over Time')
        returns_ax.set_xlabel('Time Step')
        returns_ax.set_ylabel('Returns %')
        returns_ax.grid(True)
        
        # Plot metrics - e.g., rewards and any other metrics
        if isinstance(rewards, np.ndarray) and rewards.size > 0:
            # Handle numpy array
            metrics_ax.plot(dates[1:len(rewards)+1], rewards, 'y-', label='Rewards')
            metrics_ax.plot(dates[1:len(rewards)+1], np.cumsum(rewards), 'm-', label='Cumulative Reward')
        elif isinstance(rewards, list) and len(rewards) > 0:
            # Handle list
            rewards_array = np.array(rewards)
            metrics_ax.plot(dates[1:len(rewards)+1], rewards_array, 'y-', label='Rewards')
            metrics_ax.plot(dates[1:len(rewards)+1], np.cumsum(rewards_array), 'm-', label='Cumulative Reward')
        metrics_ax.set_title('Performance Metrics')
        metrics_ax.set_xlabel('Time Step')
        metrics_ax.set_ylabel('Value')
        metrics_ax.grid(True)
        metrics_ax.legend()
        
        # Add key statistics as text
        start_value = portfolio_values[0] if portfolio_values else 0
        end_value = portfolio_values[-1] if portfolio_values else 0
        total_return = ((end_value / start_value) - 1) * 100 if start_value > 0 else 0
        
        # Calculate other metrics if available
        sharpe = backtest_data.get('sharpe_ratio', 'N/A')
        max_drawdown = backtest_data.get('max_drawdown', 'N/A')
        if isinstance(max_drawdown, float):
            max_drawdown = f"{max_drawdown:.2%}"
        
        stats_text = (
            f"Initial Portfolio: ${start_value:.2f}\n"
            f"Final Portfolio: ${end_value:.2f}\n"
            f"Total Return: {total_return:.2f}%\n"
            f"Sharpe Ratio: {sharpe}\n"
            f"Max Drawdown: {max_drawdown}"
        )
        
        # Add stats text to the figure
        fig.text(0.01, 0.01, stats_text, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for title
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plt.ion()  # Re-enable interactive mode if it was enabled before 