import sys
import os
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging first, before any other imports
from utils.setup_logging import setup_logging
logger = setup_logging(
    log_filename="train_long_term.log",
    console_level=logging.INFO
)

# Now import the rest of the modules
from utils.logger import Logger
from environments.trading_env import TradingEnv
from models.agents.dqn_agent import DQNAgent
from models.agents.directional_dqn_agent import DirectionalDQNAgent
from models.processors.trading_processor import TradingObservationProcessor
from data.data_manager import DataManager
from models.model_manager import ModelManager
from models.backtesting import Backtester
from models.experiment_manager import ExperimentManager

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
from config.env import ENV_PARAMS, MARKET_FRIC_PARAMS, CONSTRAINT_PARAMS, REWARD_PARAMS
from config.tickers import DOW_30_TICKER, NASDAQ_100_TICKER, SP_500_TICKER


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DRL trading agent for an extended period")
    parser.add_argument("--experiment-name", type=str, default=f"directional_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--assets", type=str, nargs="+", default=DOW_30_TICKER)
    parser.add_argument("--n-episodes", type=int, default=1000000)
    parser.add_argument("--max-train-time", type=int, default=86400, help="Maximum training time in seconds (24h default)")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--backtest-interval", type=int, default=50)
    parser.add_argument("--render-train", action="store_true", help="Render training environment")
    parser.add_argument("--render-eval", action="store_true", help="Render evaluation environment")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    logger.info(f"Starting long-term training experiment: {args.experiment_name}")
    logger.info(f"Training with assets: {args.assets}")
    logger.info(f"Maximum training time: {timedelta(seconds=args.max_train_time)}")
    
    # Data preparation
    logger.info("Preparing data...")
    data_manager = DataManager(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        normalized_data_dir="data/normalized",
        save_raw_data=True,
        save_processed_data=True,
        save_normalized_data=True,
        use_saved_data=True,
    )
    
    # Download and process data
    raw_data = data_manager.download_data(
        tickers=args.assets,
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        source="yahoo",
        time_interval="1d",
        add_day_index_bool=True,
        force_download=True,
    )
    
    # Process with technical indicators
    processed_data = data_manager.process_data(
        data=raw_data,
        processor_params=PROCESSOR_PARAMS,
        save_data=True,
    )
    
    # Normalize data
    normalized_data = data_manager.normalize_data(
        data=processed_data, 
        method=NORMALIZATION_PARAMS["method"],
        save_data=True,
        handle_outliers=True,
        fill_value=0
    )

    raw_train_data, raw_val_data = data_manager.split_data(
        data=raw_data,
        train_start_date=TRAIN_START_DATE,
        train_end_date=TRAIN_END_DATE,
        test_start_date=TEST_START_DATE,
        test_end_date=TEST_END_DATE,
        strict_chronological=False
    )
    
    
    # Split the raw data exactly the same way for consistent behavior
    normalized_train_data, normalized_val_data = data_manager.split_data(
        data=normalized_data,
        train_start_date=TRAIN_START_DATE,
        train_end_date=TRAIN_END_DATE,
        test_start_date=TEST_START_DATE,
        test_end_date=TEST_END_DATE,
        strict_chronological=False
    )

    # Define column mapping
    columns = {
        "ticker": "ticker",
        "price": "close",
        "day": "day",
        "ohlcv": ["open", "high", "low", "close", "volume"],
        "tech_cols": [col for col in processed_data.columns if col not in ["ticker", "day", "open", "high", "low", "close", "volume", "date", "timestamp"]]
    }
    
    # Create environments
    logger.info("Creating environments...")
    train_env = TradingEnv(
        processed_data=normalized_train_data,
        raw_data=raw_train_data,
        columns=columns,
        env_params=ENV_PARAMS,
        friction_params=MARKET_FRIC_PARAMS,
        constraint_params=CONSTRAINT_PARAMS,
        reward_params=REWARD_PARAMS["returns_based"],
        render_mode="human" if args.render_train else None,
    )
    
    val_env = TradingEnv(
        processed_data=normalized_val_data,
        raw_data=raw_val_data,
        columns=columns,
        env_params=ENV_PARAMS,
        friction_params=MARKET_FRIC_PARAMS,
        constraint_params=CONSTRAINT_PARAMS,
        reward_params=REWARD_PARAMS["returns_based"],
        render_mode="human" if args.render_eval else None,
    )
    
    # Create agent
    logger.info("Initializing agent...")
    agent = DirectionalDQNAgent(
        env=train_env,
        observation_processor_class=TradingObservationProcessor,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999,
        target_update=10,
        memory_size=10000,
        batch_size=256,
    )
    
    # Initialize model manager
    model_manager = ModelManager(
        base_dir=f"models/saved/{args.experiment_name}",
        save_interval=args.save_interval,
        backtest_interval=args.backtest_interval,
        max_saves=10,
        save_visualizations=True
    )
    
    # Initialize backtester
    backtester = Backtester(
        env=val_env,
        agent=agent,
        asset_names=args.assets,
        visualizer=None,
        save_visualizations=True,
        visualization_dir=f"experiments/{args.experiment_name}/backtest_visualizations"
    )
    
    # Initialize experiment manager
    experiment_manager = ExperimentManager(
        experiment_name=args.experiment_name,
        train_env=train_env,
        val_env=val_env,
        agent=agent,
        model_manager=model_manager,
        backtester=backtester,
        max_train_time=args.max_train_time,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        n_eval_episodes=5,
        early_stopping_patience=50 if not args.no_early_stopping else float('inf'),
        early_stopping_threshold=0.01,
        render_train=args.render_train,
        render_eval=args.render_eval,
        base_dir="experiments"
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        metrics = experiment_manager.train(n_episodes=args.n_episodes)
        
        # Print summary
        logger.info("Training completed!")
        logger.info(f"Total episodes: {len(metrics['episode'])}")
        if metrics['episode_reward']:
            logger.info(f"Final training reward: {metrics['episode_reward'][-1]:.4f}")
        if metrics['eval_return']:
            logger.info(f"Final evaluation return: {metrics['eval_return'][-1]:.4f}")
        if 'sharpe_ratio' in metrics and metrics['sharpe_ratio']:
            logger.info(f"Final Sharpe ratio: {metrics['sharpe_ratio'][-1]:.4f}")
        if 'total_return' in metrics and metrics['total_return']:
            logger.info(f"Final total return: {metrics['total_return'][-1]:.2%}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Close environments
        train_env.close()
        val_env.close()
        logger.info("Environments closed.")

if __name__ == "__main__":
    main() 