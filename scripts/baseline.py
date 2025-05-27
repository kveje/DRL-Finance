import sys
import os
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
from utils.setup_logging import setup_logging
logger = setup_logging(
    log_filename="baseline_evaluation.log",
    console_level=logging.INFO
)

# Import baseline strategies
from models.baseline.buy_and_hold import BuyAndHoldStrategy
from models.baseline.momentum import MomentumStrategy
from models.baseline.mean_reversion import MeanReversionStrategy
from models.baseline.random_strategy import RandomStrategy
from models.baseline.equal_weight import EqualWeightStrategy

# Import data management
from data.data_manager import DataManager

# Import configuration
from config.data import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
from config.tickers import DOW_30_TICKER, NASDAQ_100_TICKER, SP_500_TICKER

# Strategy label mapping for better visualization
STRATEGY_LABELS = {
    "buy_and_hold": "Buy & Hold",
    "momentum_2": "Momentum (Top 2)",
    "momentum_3": "Momentum (Top 3)", 
    "momentum_5": "Momentum (Top 5)",
    "mean_reversion_10": "Mean Reversion (10d)",
    "mean_reversion_20": "Mean Reversion (20d)",
    "mean_reversion_50": "Mean Reversion (50d)",
    "random": "Random Strategy",
    "equal_weight_10": "Equal Weight (10d)",
    "equal_weight_20": "Equal Weight (20d)",
    "equal_weight_50": "Equal Weight (50d)"
}

def calculate_enhanced_metrics(portfolio_values):
    """
    Calculate comprehensive performance metrics matching the experiment manager.
    
    Args:
        portfolio_values: List or array of portfolio values over time
        
    Returns:
        Dictionary of performance metrics
    """
    if len(portfolio_values) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    
    # Convert to numpy array for calculations
    portfolio_values = np.array(portfolio_values, dtype=float)
    
    # Calculate returns as percentage changes in portfolio value
    returns = np.zeros_like(portfolio_values[1:], dtype=float)
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i-1] != 0:
            returns[i-1] = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        else:
            returns[i-1] = 0.0
    
    # Basic metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] if portfolio_values[0] != 0 else 0.0
    
    # Annualized return (assuming daily data)
    n_periods = len(portfolio_values)
    years = n_periods / 252.0  # Assuming 252 trading days per year
    annualized_return = ((portfolio_values[-1] / portfolio_values[0]) ** (1/years) - 1) if years > 0 and portfolio_values[0] != 0 else 0.0
    
    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
    
    # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = np.mean(returns) * np.sqrt(252) / volatility if volatility > 0 else 0.0
    
    # Sortino Ratio (using negative returns only)
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0.0
    sortino_ratio = np.mean(returns) * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0.0
    
    # Maximum Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Calmar Ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
    }

def run_strategy(strategy_obj, data_df: pd.DataFrame, start_date: str, end_date: str, assets: list):
    """Run a trading strategy on the provided data."""
    strategy_obj.reset()
    
    filtered_df = data_df[(data_df.index >= start_date) & (data_df.index <= end_date)]
    dates = filtered_df.index.unique()
    
    for step, date in enumerate(dates):
        day_data = filtered_df[filtered_df.index == date]
        prices = {}
        for asset in assets:
            asset_data = day_data[day_data['ticker'] == asset]
            if not asset_data.empty:
                prices[asset] = asset_data['close'].values[0]
        strategy_obj.step(step, prices)
    
    # Calculate enhanced metrics
    enhanced_metrics = calculate_enhanced_metrics(strategy_obj.portfolio_value_history)
    
    return {
        "metrics": enhanced_metrics,
        "portfolio_value_history": strategy_obj.portfolio_value_history,
        "position_history": strategy_obj.position_history
    }

def save_results(results, output_dir, period):
    """Save strategy results for a specific period."""
    period_dir = os.path.join(output_dir, period)
    os.makedirs(period_dir, exist_ok=True)
    
    # Save summary metrics
    summary_data = []
    for strategy_name, result in results.items():
        metrics = result['metrics']
        strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
        summary_data.append({
            'Strategy': strategy_label,
            'Total Return': f"{metrics['total_return'] * 100:.2f}%",
            'Annualized Return': f"{metrics['annualized_return'] * 100:.2f}%",
            'Volatility': f"{metrics['volatility'] * 100:.2f}%",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.4f}",
            'Sortino Ratio': f"{metrics['sortino_ratio']:.4f}",
            'Calmar Ratio': f"{metrics['calmar_ratio']:.4f}",
            'Max Drawdown': f"{metrics['max_drawdown'] * 100:.2f}%",

        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(period_dir, "strategy_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Plot 1: Portfolio Values
    plt.figure(figsize=(15, 8))
    for strategy_name, result in results.items():
        values = result['portfolio_value_history']
        strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
        plt.plot(range(len(values)), values, label=strategy_label, linewidth=2)
    
    plt.title(f'Portfolio Value Evolution - {period.title()} Period', fontsize=16)
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(period_dir, "portfolio_values.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Risk-Return Scatter
    plt.figure(figsize=(12, 8))
    strategies = []
    returns = []
    volatilities = []
    sharpe_ratios = []
    
    for strategy_name, result in results.items():
        strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
        strategies.append(strategy_label)
        returns.append(result['metrics']['annualized_return'] * 100)
        volatilities.append(result['metrics']['volatility'] * 100)
        sharpe_ratios.append(result['metrics']['sharpe_ratio'])
    
    # Create scatter plot with color-coded Sharpe ratios
    scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, s=100, alpha=0.7, cmap='viridis')
    
    # Add strategy labels
    for i, strategy in enumerate(strategies):
        plt.annotate(strategy, (volatilities[i], returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.title(f'Risk-Return Profile - {period.title()} Period', fontsize=16)
    plt.xlabel('Volatility (%)', fontsize=12)
    plt.ylabel('Annualized Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(period_dir, "risk_return_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Metrics Comparison Bar Chart (replacing radar chart)
    metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    metric_labels = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        strategy_names = []
        metric_values = []
        
        for strategy_name, result in results.items():
            strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
            strategy_names.append(strategy_label)
            metric_values.append(result['metrics'][metric])
        
        # Create bar plot
        bars = axes[i].bar(range(len(strategy_names)), metric_values, alpha=0.7)
        axes[i].set_title(f'{label} Comparison', fontsize=14)
        axes[i].set_ylabel(label, fontsize=12)
        axes[i].set_xticks(range(len(strategy_names)))
        axes[i].set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Color bars based on values
        max_val = max(metric_values) if metric_values else 1
        min_val = min(metric_values) if metric_values else 0
        for j, bar in enumerate(bars):
            if max_val != min_val:
                normalized_val = (metric_values[j] - min_val) / (max_val - min_val)
            else:
                normalized_val = 0.5
            bar.set_color(plt.cm.viridis(normalized_val))
    
    plt.suptitle(f'Performance Metrics Comparison - {period.title()} Period', fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(period_dir, "metrics_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Improved Drawdown Analysis
    n_strategies = len(results)
    fig, axes = plt.subplots(n_strategies, 1, figsize=(15, 3 * n_strategies))
    
    # Handle single strategy case
    if n_strategies == 1:
        axes = [axes]
    
    for i, (strategy_name, result) in enumerate(results.items()):
        values = np.array(result['portfolio_value_history'])
        strategy_label = STRATEGY_LABELS.get(strategy_name, strategy_name)
        
        # Calculate drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak * 100  # Convert to percentage
        
        # Plot drawdown
        axes[i].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[i].plot(range(len(drawdown)), drawdown, color='red', linewidth=1)
        
        # Set title and labels with proper spacing
        max_dd = result["metrics"]["max_drawdown"] * 100
        axes[i].set_title(f'{strategy_label} (Max DD: {max_dd:.2f}%)', 
                         fontsize=12, pad=10)
        axes[i].set_ylabel('Drawdown (%)', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(bottom=0)  # Ensure drawdown starts at 0
        
        # Only add x-label to the bottom plot
        if i == n_strategies - 1:
            axes[i].set_xlabel('Trading Days', fontsize=10)
    
    plt.suptitle(f'Drawdown Analysis - {period.title()} Period', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    
    plot_path = os.path.join(period_dir, "drawdown_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed portfolio history
    for strategy_name, result in results.items():
        portfolio_df = pd.DataFrame({
            'Day': range(len(result['portfolio_value_history'])),
            'Portfolio_Value': result['portfolio_value_history']
        })
        portfolio_path = os.path.join(period_dir, f"{strategy_name}_portfolio.csv")
        portfolio_df.to_csv(portfolio_path, index=False)

def main():
    # Fixed parameters
    assets = DOW_30_TICKER
    initial_capital = 100000.0
    transaction_cost = 0.001
    output_dir = "baseline_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data manager
    data_manager = DataManager(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        normalized_data_dir="data/normalized",
        save_raw_data=True,
        save_processed_data=True,
        save_normalized_data=True,
        use_saved_data=True,
    )

    # Download data for the entire period
    raw_data = data_manager.download_data(
        tickers=assets,
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        source="yahoo",
        time_interval="1d",
        add_day_index_bool=False,
        force_download=False,
    )

    # Make sure date column is index
    raw_data.set_index('date', inplace=True)
    
    # Initialize all strategies
    strategy_objects = {
        "buy_and_hold": BuyAndHoldStrategy(
            assets=assets,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost
        ),
        "momentum_2": MomentumStrategy(
            assets=assets,
            initial_capital=initial_capital,
            lookback_period=20,
            top_k=2,
            transaction_cost=transaction_cost
        ),
        "momentum_3": MomentumStrategy(
            assets=assets,
            initial_capital=initial_capital,
            lookback_period=20,
            top_k=3,
            transaction_cost=transaction_cost
        ),
        "momentum_5": MomentumStrategy(
            assets=assets,
            initial_capital=initial_capital,
            lookback_period=20,
            top_k=5,
            transaction_cost=transaction_cost
        ),
        "mean_reversion_10": MeanReversionStrategy(
            assets=assets,
            initial_capital=initial_capital,
            window_size=10,
            theta_buy=0.5,
            theta_sell=-0.5,
            alpha=0.2,
            z_max=2.0,
            transaction_cost=transaction_cost
        ),
        "mean_reversion_20": MeanReversionStrategy(
            assets=assets,
            initial_capital=initial_capital,
            window_size=20,
            theta_buy=0.5,
            theta_sell=-0.5,
            alpha=0.2,
            z_max=2.0,
            transaction_cost=transaction_cost
        ),
        "mean_reversion_50": MeanReversionStrategy(
            assets=assets,
            initial_capital=initial_capital,
            window_size=50,
            theta_buy=0.5,
            theta_sell=-0.5,
            alpha=0.2,
            z_max=2.0,
            transaction_cost=transaction_cost
        ),
        "random": RandomStrategy(
            assets=assets,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            seed=42
        ),
        "equal_weight_10": EqualWeightStrategy(
            assets=assets,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            rebalance_frequency=10
        ),
        "equal_weight_20": EqualWeightStrategy(
            assets=assets,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            rebalance_frequency=20
        ),
        "equal_weight_50": EqualWeightStrategy(
            assets=assets,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            rebalance_frequency=50
        )
    }
    
    # Evaluate on each period
    periods = {
        "train": (TRAIN_START_DATE, TRAIN_END_DATE),
        "validation": (TEST_START_DATE, TEST_END_DATE),
        "trade": (TRADE_START_DATE, TRADE_END_DATE)
    }
    
    for period, (start_date, end_date) in periods.items():
        logger.info(f"\nEvaluating strategies on {period} period ({start_date} to {end_date})")
        
        period_results = {}
        for name, strategy in strategy_objects.items():
            logger.info(f"Running {name} strategy...")
            results = run_strategy(
                strategy_obj=strategy,
                data_df=raw_data,
                start_date=start_date,
                end_date=end_date,
                assets=assets
            )
            period_results[name] = results
            
            # Log key metrics
            metrics = results['metrics']
            logger.info(f"{name} - Total Return: {metrics['total_return']*100:.2f}%, "
                       f"Annualized Return: {metrics['annualized_return']*100:.2f}%, "
                       f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}, "
                       f"Sortino Ratio: {metrics['sortino_ratio']:.4f}, "
                       f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        # Save results for this period
        save_results(period_results, output_dir, period)
    
    logger.info(f"\nAll results have been saved to {output_dir}")

if __name__ == "__main__":
    main() 