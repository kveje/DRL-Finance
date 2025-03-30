#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize the logger
from utils.logger import Logger
# Initialize the logger once at the start of your application
logger = Logger(
    name="REDUNANDANT",
    level="INFO",
    log_dir="logs",
    log_to_file=True,
    log_to_console=False,
    experiment_name="test_trading_env",
    # Add other configuration as needed
)

import pandas as pd
import numpy as np
from datetime import datetime

from data.data_manager import DataManager
from environments.trading_env import TradingEnv
from config.data import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    PROCESSOR_PARAMS,
    NORMALIZATION_PARAMS,
)
from config.env import ENV_PARAMS, MARKET_FRIC_PARAMS, CONSTRAINT_PARAMS, REWARD_PARAMS


def main():
    print("Starting trading environment test...")

    # Initialize data manager
    print("\nInitializing data manager...")
    data_manager = DataManager(
        data_dir="data/raw",
        cache_dir="data/processed",
        save_raw_data=True,
        use_cache=True,
    )

    # Define test assets
    test_assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Download and process data
    print(f"\nDownloading data for {len(test_assets)} assets...")
    raw_data = data_manager.download_data(
        tickers=test_assets,
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        source="yahoo",
        time_interval="1d",
    )

    # Process data with technical indicators
    print("\nProcessing data with technical indicators...")
    processed_data = data_manager.process_data(
        data=raw_data,
        processor_params=PROCESSOR_PARAMS,
    )

    # Normalize data
    print("\nNormalizing data...")
    normalized_data = data_manager.normalize_data(
        data=processed_data, method=NORMALIZATION_PARAMS["method"]
    )

    # Print data shapes
    print("\nData shapes:")
    print(f"Raw data: {raw_data.shape}")
    print(f"Processed data: {processed_data.shape}")
    print(f"Normalized data: {normalized_data.shape}")

    # Create environment
    print("\nCreating trading environment...")
    env = TradingEnv(
        processed_data=normalized_data,
        raw_data=raw_data,
        env_params=ENV_PARAMS,
        friction_params=MARKET_FRIC_PARAMS,
        constraint_params=CONSTRAINT_PARAMS,
        reward_params=REWARD_PARAMS,
        seed=42,
    )

    # Print environment info
    print("\nEnvironment information:")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of assets: {env.n_assets}")
    print(f"Number of technical indicators: {len(env.technical_columns)}")

    # Run test episode
    print("\nRunning test episode...")
    obs = env.reset()
    done = False
    total_reward = 0
    portfolio_values = [env.initial_balance]
    step_count = 0

    while not done:
        # Random action (for testing)
        action = env.action_space.sample()

        # Take step
        obs, reward, done, info = env.step(action)

        # Record portfolio value
        portfolio_values.append(info["balance"])

        # Accumulate reward
        total_reward += reward
        step_count += 1

        if step_count % 100 == 0:
            print(f"Step {step_count}: Portfolio value = ${info['balance']:.2f}")

    # Calculate statistics
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    final_value = portfolio_values[-1]
    total_return = (final_value - env.initial_balance) / env.initial_balance

    # Print results
    print("\nTest Episode Results:")
    print(f"Total steps: {step_count}")
    print(f"Initial portfolio value: ${env.initial_balance:.2f}")
    print(f"Final portfolio value: ${final_value:.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average daily return: {np.mean(returns):.2%}")
    print(f"Return volatility: {np.std(returns):.2%}")

    print("\nFinal positions:")
    for i, pos in enumerate(env.positions):
        print(f"{test_assets[i]}: {pos:.2f}")


if __name__ == "__main__":
    main()
