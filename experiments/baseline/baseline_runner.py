import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path to import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import baseline strategies
from experiments.baseline.buy_and_hold import BuyAndHoldStrategy
from experiments.baseline.momentum import MomentumStrategy
from experiments.baseline.mean_reversion import MeanReversionStrategy
from experiments.baseline.random_strategy import RandomStrategy
from experiments.baseline.equal_weight import EqualWeightStrategy

# Internal imports
from data.data_manager import DataManager

# Import configuration
from config.data import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    PROCESSOR_PARAMS,
    NORMALIZATION_PARAMS,
)

# Import logger
from utils.logger import Logger
logger = Logger.get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Run and compare baseline trading strategies")
    parser.add_argument("--assets", type=str, nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
    parser.add_argument("--start-date", type=str, default=TEST_START_DATE)
    parser.add_argument("--end-date", type=str, default=TEST_END_DATE)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--momentum-lookback", type=int, default=20)
    parser.add_argument("--momentum-top-n", type=int, default=2)
    parser.add_argument("--mean-reversion-window", type=int, default=20)
    parser.add_argument("--mean-reversion-z-threshold", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/baseline")
    parser.add_argument("--strategies", type=str, nargs="+", 
                        choices=["buy_and_hold", "momentum", "mean_reversion", "random", "equal_weight", "all"],
                        default=["all"])
    
    return parser.parse_args()


def run_strategy(strategy_obj, data_df, start_date, end_date, assets):
    """
    Run a trading strategy on the provided data.
    
    Args:
        strategy_obj: Strategy object to use
        data_df: DataFrame containing price data
        start_date: Start date for testing
        end_date: End date for testing
        assets: List of assets to trade
        
    Returns:
        Dictionary containing performance metrics and portfolio history
    """
    # Reset strategy
    strategy_obj.reset()
    
    # Filter data for date range
    filtered_df = data_df[(data_df.index >= start_date) & (data_df.index <= end_date)]
    dates = filtered_df.index.unique()
    
    # Run strategy for each day
    for step, date in enumerate(dates):
        day_data = filtered_df[filtered_df.index == date]
        
        # Create price dict for the day
        prices = {}
        for asset in assets:
            asset_data = day_data[day_data['ticker'] == asset]
            if not asset_data.empty:
                prices[asset] = asset_data['close'].values[0]
        
        # Execute strategy step
        step_result = strategy_obj.step(step, prices)
    
    # Get performance metrics
    metrics = strategy_obj.get_performance_metrics()
    
    return {
        "metrics": metrics,
        "portfolio_value_history": strategy_obj.portfolio_value_history,
        "position_history": strategy_obj.position_history
    }


def compare_strategies(strategy_results, output_dir):
    """
    Compare the performance of different strategies.
    
    Args:
        strategy_results: Dictionary mapping strategy names to their results
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare summary dataframe
    summary_data = []
    
    for strategy_name, results in strategy_results.items():
        metrics = results['metrics']
        summary_data.append({
            'Strategy': strategy_name,
            'Total Return': f"{metrics['total_return'] * 100:.2f}%",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.4f}",
            'Max Drawdown': f"{metrics['max_drawdown'] * 100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "strategy_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Strategy comparison summary saved to {summary_path}")
    logger.info("\nStrategy Performance Summary:")
    logger.info(summary_df.to_string(index=False))
    
    # Plot portfolio values
    plt.figure(figsize=(12, 6))
    
    for strategy_name, results in strategy_results.items():
        values = results['portfolio_value_history']
        plt.plot(range(len(values)), values, label=strategy_name)
    
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "portfolio_values.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Portfolio value plot saved to {plot_path}")
    
    # Save detailed results to CSV files
    for strategy_name, results in strategy_results.items():
        portfolio_df = pd.DataFrame({
            'Day': range(len(results['portfolio_value_history'])),
            'Portfolio_Value': results['portfolio_value_history']
        })
        
        portfolio_path = os.path.join(output_dir, f"{strategy_name}_portfolio.csv")
        portfolio_df.to_csv(portfolio_path, index=False)
        logger.info(f"{strategy_name} portfolio history saved to {portfolio_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    logger.info(f"Running baseline strategy comparison with assets: {', '.join(args.assets)}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Data preparation
    logger.info("Preparing data...")
    data_manager = DataManager(
        data_dir="data/raw",
        cache_dir="data/processed",
        save_raw_data=True,
        use_cache=True,
    )
    
    # Download and process data
    raw_data = data_manager.download_data(
        tickers=args.assets,
        start_date=args.start_date,
        end_date=args.end_date,
        source="yahoo",
        time_interval="1d",
        add_day_index_bool=True,
    )
    
    # Get the strategies to run
    strategies_to_run = args.strategies
    if "all" in strategies_to_run:
        strategies_to_run = ["buy_and_hold", "momentum", "mean_reversion", "random", "equal_weight"]
    
    # Initialize strategies
    strategy_objects = {}
    
    if "buy_and_hold" in strategies_to_run:
        strategy_objects["buy_and_hold"] = BuyAndHoldStrategy(
            assets=args.assets,
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost
        )
    
    if "momentum" in strategies_to_run:
        strategy_objects["momentum"] = MomentumStrategy(
            assets=args.assets,
            initial_capital=args.initial_capital,
            lookback_period=args.momentum_lookback,
            top_n=args.momentum_top_n,
            transaction_cost=args.transaction_cost
        )
    
    if "mean_reversion" in strategies_to_run:
        strategy_objects["mean_reversion"] = MeanReversionStrategy(
            assets=args.assets,
            initial_capital=args.initial_capital,
            window_size=args.mean_reversion_window,
            z_threshold=args.mean_reversion_z_threshold,
            transaction_cost=args.transaction_cost
        )
    
    if "random" in strategies_to_run:
        strategy_objects["random"] = RandomStrategy(
            assets=args.assets,
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost,
            seed=args.random_seed
        )
    
    if "equal_weight" in strategies_to_run:
        strategy_objects["equal_weight"] = EqualWeightStrategy(
            assets=args.assets,
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost
        )
    
    # Run each strategy
    strategy_results = {}
    
    for name, strategy in strategy_objects.items():
        logger.info(f"Running {name} strategy...")
        results = run_strategy(
            strategy_obj=strategy,
            data_df=raw_data,
            start_date=args.start_date,
            end_date=args.end_date,
            assets=args.assets
        )
        strategy_results[name] = results
        
        logger.info(f"{name} strategy completed.")
        logger.info(f"Total Return: {results['metrics']['total_return'] * 100:.2f}%")
        logger.info(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.4f}")
        logger.info(f"Max Drawdown: {results['metrics']['max_drawdown'] * 100:.2f}%")
        logger.info("-" * 50)
    
    # Compare strategies
    logger.info("Comparing strategy performance...")
    compare_strategies(strategy_results, args.output_dir)
    
    logger.info("Baseline comparison completed!")


if __name__ == "__main__":
    main() 