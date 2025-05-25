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

def run_strategy(strategy_obj, data_df, start_date, end_date, assets):
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
    
    return {
        "metrics": strategy_obj.get_performance_metrics(),
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
        summary_data.append({
            'Strategy': strategy_name,
            'Total Return': f"{metrics['total_return'] * 100:.2f}%",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.4f}",
            'Max Drawdown': f"{metrics['max_drawdown'] * 100:.2f}%",
            'Annualized Return': f"{metrics.get('annualized_return', 0) * 100:.2f}%",
            'Volatility': f"{metrics.get('volatility', 0) * 100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(period_dir, "strategy_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Plot 1: Portfolio Values
    plt.figure(figsize=(12, 6))
    for strategy_name, result in results.items():
        values = result['portfolio_value_history']
        plt.plot(range(len(values)), values, label=strategy_name)
    
    plt.title('Portfolio Value Evolution')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(period_dir, "portfolio_values.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Sharpe Ratios
    plt.figure(figsize=(12, 6))
    strategies = []
    sharpe_ratios = []
    for strategy_name, result in results.items():
        strategies.append(strategy_name)
        sharpe_ratios.append(result['metrics']['sharpe_ratio'])
    
    plt.bar(strategies, sharpe_ratios)
    plt.title('Sharpe Ratio Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(period_dir, "sharpe_ratios.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Maximum Drawdowns
    plt.figure(figsize=(12, 6))
    max_drawdowns = []
    for strategy_name, result in results.items():
        max_drawdowns.append(result['metrics']['max_drawdown'] * 100)  # Convert to percentage
    
    plt.bar(strategies, max_drawdowns)
    plt.title('Maximum Drawdown Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Maximum Drawdown (%)')
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(period_dir, "max_drawdowns.png")
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
            top_n=2,
            transaction_cost=transaction_cost
        ),
        "momentum_3": MomentumStrategy(
            assets=assets,
            initial_capital=initial_capital,
            lookback_period=20,
            top_n=3,
            transaction_cost=transaction_cost
        ),
        "momentum_5": MomentumStrategy(
            assets=assets,
            initial_capital=initial_capital,
            lookback_period=20,
            top_n=5,
            transaction_cost=transaction_cost
        ),
        "mean_reversion_10": MeanReversionStrategy(
            assets=assets,
            initial_capital=initial_capital,
            window_size=10,
            z_threshold=1.0,
            transaction_cost=transaction_cost
        ),
        "mean_reversion_20": MeanReversionStrategy(
            assets=assets,
            initial_capital=initial_capital,
            window_size=20,
            z_threshold=1.0,
            transaction_cost=transaction_cost
        ),
        "mean_reversion_50": MeanReversionStrategy(
            assets=assets,
            initial_capital=initial_capital,
            window_size=50,
            z_threshold=1.0,
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
                       f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}, "
                       f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        # Save results for this period
        save_results(period_results, output_dir, period)
    
    logger.info(f"\nAll results have been saved to {output_dir}")

if __name__ == "__main__":
    main() 