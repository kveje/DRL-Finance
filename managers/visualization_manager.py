"""Enhanced visualization manager for trading results"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

class VisualizationManager:
    """
    Enhanced visualization manager for trading results that provides:
    - Portfolio value and performance metrics
    - Asset positions and allocations
    - Trading actions and cash balance
    - Returns analysis
    - Risk metrics
    - Trade statistics
    """
    
    def __init__(self):
        """Initialize the visualizer with default style settings."""
        # Set style
        plt.style.use('seaborn-v0_8')  # Use a specific seaborn style version
        sns.set_theme()  # Set seaborn theme
        
        # Set default figure size and DPI
        self.figsize = (18, 12)
        self.dpi = 150
        
        # Set default font sizes
        self.title_fontsize = 16
        self.label_fontsize = 12
        self.legend_fontsize = 10
        self.stats_fontsize = 10
    
    @staticmethod
    def create_and_save_backtest_visualization(
        asset_names: List[str],
        backtest_data: Dict[str, Any],
        filename: str,
        title: Optional[str] = None
    ):
        """
        Create and save a comprehensive visualization of backtest results.
        
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
        reward_components = backtest_data.get('reward_components', {})
        cash = backtest_data.get('cash', [])
        metrics = backtest_data.get('metrics', {})
        trade_history = backtest_data.get('trade_history', [])
        
        # Create figure with subplots
        plt.ioff()  # Turn off interactive mode
        fig = plt.figure(figsize=(18, 18))  # Increased height for new subplot
        gs = GridSpec(6, 2, figure=fig, height_ratios=[1.2, 1, 1, 1, 1, 0.8], hspace=0.4, wspace=0.3)
        
        # Create subplots
        portfolio_ax = fig.add_subplot(gs[0, :])
        positions_ax = fig.add_subplot(gs[1, 0])
        cash_ax = fig.add_subplot(gs[1, 1])
        actions_ax = fig.add_subplot(gs[2, 0])
        returns_ax = fig.add_subplot(gs[2, 1])
        metrics_ax = fig.add_subplot(gs[3, 0])
        trade_ax = fig.add_subplot(gs[3, 1])
        reward_components_ax = fig.add_subplot(gs[4, :])  # New subplot for reward components
        stats_ax = fig.add_subplot(gs[5, :])
        
        # Set main title
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        
        # 1. Portfolio Value Plot
        dates = list(range(len(portfolio_values)))
        portfolio_ax.plot(dates, portfolio_values, 'b-', label='Portfolio Value', linewidth=2)
        portfolio_ax.set_title('Portfolio Value Over Time', fontsize=14)
        portfolio_ax.set_xlabel('Time Step', fontsize=12)
        portfolio_ax.set_ylabel('Value ($)', fontsize=12)
        portfolio_ax.grid(True, alpha=0.3)
        
        # Add drawdown overlay
        if len(portfolio_values) > 0:
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            drawdown_ax = portfolio_ax.twinx()
            drawdown_ax.fill_between(dates, drawdown * 100, alpha=0.3, color='red', label='Drawdown')
            drawdown_ax.set_ylabel('Drawdown (%)', fontsize=12)
            drawdown_ax.set_ylim(0, max(drawdown) * 100 * 1.1)
        
        # 2. Positions Plot
        if isinstance(positions, np.ndarray) and positions.size > 0:
            if len(positions.shape) > 1:
                for i, asset in enumerate(asset_names):
                    if i < positions.shape[1]:
                        positions_ax.plot(dates, positions[:, i], label=asset, alpha=0.7)
            else:
                positions_ax.plot(dates, positions, label=asset_names[0])
        positions_ax.set_title('Asset Positions', fontsize=14)
        positions_ax.set_xlabel('Time Step', fontsize=12)
        positions_ax.set_ylabel('Position Size', fontsize=12)
        positions_ax.grid(True, alpha=0.3)
        
        # 3. Cash Balance Plot
        if len(cash) > 0:
            cash_ax.plot(dates[:len(cash)], cash, 'g-', label='Cash Balance', linewidth=2)
            cash_ax.set_title('Cash Balance Over Time', fontsize=14)
            cash_ax.set_xlabel('Time Step', fontsize=12)
            cash_ax.set_ylabel('Cash ($)', fontsize=12)
            cash_ax.grid(True, alpha=0.3)
        
        # 4. Actions Plot
        if isinstance(actions, np.ndarray) and actions.size > 0:
            if len(actions.shape) > 1:
                for i, asset in enumerate(asset_names):
                    if i < actions.shape[1]:
                        actions_ax.plot(dates[:len(actions[:, i])], actions[:, i], label=asset, alpha=0.7)
            else:
                actions_ax.plot(dates, actions, label=asset_names[0])
        actions_ax.set_title('Trading Actions', fontsize=14)
        actions_ax.set_xlabel('Time Step', fontsize=12)
        actions_ax.set_ylabel('Action Value', fontsize=12)
        actions_ax.grid(True, alpha=0.3)
        
        # 5. Returns Plot
        if isinstance(returns, np.ndarray) and returns.size > 0:
            cumulative_returns = np.cumprod(1 + returns) - 1
            # Ensure dates array matches returns length
            returns_dates = dates[1:len(returns)+1]  # Skip first date since returns start from second point
            returns_ax.plot(returns_dates, cumulative_returns * 100, 'r-', 
                          label='Cumulative Returns (%)', linewidth=2)
            returns_ax.plot(returns_dates, returns * 100, 'g-', alpha=0.3, 
                          label='Step Returns (%)')
        returns_ax.set_title('Returns Over Time', fontsize=14)
        returns_ax.set_xlabel('Time Step', fontsize=12)
        returns_ax.set_ylabel('Returns (%)', fontsize=12)
        returns_ax.grid(True, alpha=0.3)
        returns_ax.legend(fontsize=10)
        
        # 6. Performance Metrics Plot
        if len(rewards) > 0:
            rewards_array = np.array(rewards)
            # Ensure dates array matches rewards length
            rewards_dates = dates[1:len(rewards)+1]  # Skip first date since rewards start from second point
            metrics_ax.plot(rewards_dates, rewards_array, 'y-', 
                          label='Total Reward', alpha=0.7)
            metrics_ax.plot(rewards_dates, np.cumsum(rewards_array), 'm-', 
                          label='Cumulative Reward', linewidth=2)
        metrics_ax.set_title('Performance Metrics', fontsize=14)
        metrics_ax.set_xlabel('Time Step', fontsize=12)
        metrics_ax.set_ylabel('Value', fontsize=12)
        metrics_ax.grid(True, alpha=0.3)
        metrics_ax.legend(fontsize=10)
        
        # 7. Reward Components Plot
        if reward_components:
            # Plot total rewards as baseline
            if len(rewards) > 0:
                rewards_dates = dates[1:len(rewards)+1]
                reward_components_ax.plot(rewards_dates, rewards_array, 
                                       'k-', label='Total Reward', 
                                       linewidth=2.5, alpha=0.8)
            
            # Plot each reward component
            for component_name, values in reward_components.items():
                if len(values) > 0:
                    # Ensure dates array matches values length
                    component_dates = dates[1:len(values)+1]
                    reward_components_ax.plot(component_dates, values, 
                                           label=component_name, alpha=0.7)
            
            reward_components_ax.set_title('Reward Components', fontsize=14)
            reward_components_ax.set_xlabel('Time Step', fontsize=12)
            reward_components_ax.set_ylabel('Value', fontsize=12)
            reward_components_ax.grid(True, alpha=0.3)
            reward_components_ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # 8. Trade Analysis Plot
        if len(trade_history) > 0:
            trade_returns = [t['portfolio_value'] / portfolio_values[t['step']] - 1 
                           for t in trade_history]
            trade_ax.hist(trade_returns, bins=20, alpha=0.7, color='blue')
            trade_ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            trade_ax.set_title('Trade Return Distribution', fontsize=14)
            trade_ax.set_xlabel('Trade Return', fontsize=12)
            trade_ax.set_ylabel('Frequency', fontsize=12)
            trade_ax.grid(True, alpha=0.3)
        
        # 9. Statistics Summary
        stats_text = []
        
        # Portfolio Statistics
        if len(portfolio_values) > 0:
            start_value = portfolio_values[0]
            end_value = portfolio_values[-1]
            total_return = ((end_value / start_value) - 1) * 100
            stats_text.extend([
                "Portfolio Statistics:",
                f"Initial Value: ${start_value:,.2f}",
                f"Final Value: ${end_value:,.2f}",
                f"Total Return: {total_return:.2f}%"
            ])
        
        # Risk Metrics
        if metrics:
            stats_text.extend([
                "\nRisk Metrics:",
                f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.2f}",
                f"Sortino Ratio: {metrics.get('sortino_ratio', 'N/A'):.2f}",
                f"Calmar Ratio: {metrics.get('calmar_ratio', 'N/A'):.2f}",
                f"Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2%}",
                f"Volatility: {metrics.get('volatility', 'N/A'):.2%}"
            ])
        
        # Add statistics text to the figure
        stats_ax.axis('off')
        stats_ax.text(0.02, 0.98, '\n'.join(stats_text),
                     fontsize=10, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Adjust layout and save
        plt.subplots_adjust(
            top=0.95,      # Top margin
            bottom=0.05,   # Bottom margin
            left=0.1,      # Left margin
            right=0.9,     # Right margin
            hspace=0.4,    # Height space between subplots
            wspace=0.3     # Width space between subplots
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plt.ion()  # Re-enable interactive mode 