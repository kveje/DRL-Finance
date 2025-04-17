import sys
import os
import argparse
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging first, before any other imports
from utils.setup_logging import setup_logging
logger = setup_logging(
    log_filename="setup_experiment.log",
    console_level=logging.INFO
)

# Import other modules
from data.data_manager import DataManager
from models.agents.directional_dqn_agent import DirectionalDQNAgent
from models.agents.dqn_agent import DQNAgent
from models.agents.ppo_agent import PPOAgent
from models.agents.a2c_agent import A2CAgent
from models.processors.trading_processor import TradingObservationProcessor
from models.backtesting import Backtester
from models.experiment_manager import ExperimentManager
from environments.trading_env import TradingEnv

# Config imports
from config.data import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    PROCESSOR_PARAMS,
    NORMALIZATION_PARAMS,
)
from config.env import ENV_PARAMS, MARKET_FRIC_PARAMS, CONSTRAINT_PARAMS, REWARD_PARAMS
from config.tickers import DOW_30_TICKER, NASDAQ_100_TICKER, SP_500_TICKER


def parse_args():
    parser = argparse.ArgumentParser(description="Setup an experiment with data visualization and exploration")
    parser.add_argument("--experiment-name", type=str, default=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--assets", type=str, nargs="+", default=DOW_30_TICKER)
    parser.add_argument("--train-start-date", type=str, default=TRAIN_START_DATE)
    parser.add_argument("--train-end-date", type=str, default=TRAIN_END_DATE)
    parser.add_argument("--val-start-date", type=str, default=TEST_START_DATE)
    parser.add_argument("--val-end-date", type=str, default=TEST_END_DATE)
    parser.add_argument("--processors", type=str, nargs="+", default=["technical_indicator", "vix"])
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--agent-type", type=str, choices=["DirectionalDQNAgent", "DQNAgent", "PPOAgent", "A2CAgent"], 
                        default="DirectionalDQNAgent", help="Type of agent to use for the experiment")
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create experiment directory
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config directory
    config_dir = experiment_dir / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Set up logging to save in the experiment directory
    logger = setup_logging(
        log_filename=f"setup_{args.experiment_name}.log",
        console_level=logging.INFO,
        experiment_dir=str(experiment_dir),
        experiment_name=args.experiment_name
    )
    
    logger.info(f"Setting up experiment: {args.experiment_name}")
    logger.info(f"Using assets: {args.assets}")
    logger.info(f"Training period: {args.train_start_date} to {args.train_end_date}")
    logger.info(f"Validation period: {args.val_start_date} to {args.val_end_date}")
    
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
    
    # Data preparation
    logger.info("Downloading data...")
    raw_data = data_manager.download_data(
        tickers=args.assets,
        start_date=args.train_start_date,
        end_date=args.val_end_date,  # End with validation end date to cover both periods
        source="yahoo",
        time_interval="1d",
        add_day_index_bool=True,
        force_download=False,
    )
    
    # Process with technical indicators
    logger.info("Processing data with technical indicators...")
    processed_data = data_manager.process_data(
        data=raw_data,
        processors=args.processors,
        processor_params=PROCESSOR_PARAMS,
        save_data=True,
    )
    
    # Normalize data
    logger.info("Normalizing data...")
    normalized_data = data_manager.normalize_data(
        data=processed_data, 
        method=NORMALIZATION_PARAMS["method"],
        save_data=True,
        handle_outliers=True,
        fill_value=0
    )

    # Split the data into training and validation sets
    logger.info("Splitting data into training and validation sets...")
    train_data, val_data = data_manager.split_data(
        data=normalized_data,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        test_start_date=args.val_start_date,
        test_end_date=args.val_end_date,
        strict_chronological=False
    )
    
    # Split the raw data exactly the same way for consistent behavior
    raw_train_data, raw_val_data = data_manager.split_data(
        data=raw_data,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        test_start_date=args.val_start_date,
        test_end_date=args.val_end_date,
        strict_chronological=False
    )
    
    # Define column mapping for the environment
    columns = {
        "ticker": "ticker",
        "price": "close",
        "day": "day",
        "ohlcv": ["open", "high", "low", "close", "volume"],
        "tech_cols": [col for col in train_data.columns if col not in ["ticker", "day", "open", "high", "low", "close", "volume", "date", "timestamp"]]
    }
    
    # Generate data visualizations
    visualizations_dir = experiment_dir / "visualizations"
    visualizations_dir.mkdir(exist_ok=True)
    
    # logger.info("Generating data visualizations...")
    # data_manager.visualize_data(
    #     raw_data=raw_train_data,
    #     processed_data=train_data,
    #     columns=None,  # Use all numeric columns
    #     time_series_columns=["close", "volume"],
    #     tickers=args.assets,
    #     save_dir=str(visualizations_dir),
    #     output_prefix="train_data"
    # )
    # 
    # data_manager.visualize_data(
    #     raw_data=raw_val_data,
    #     processed_data=val_data,
    #     columns=None,  # Use all numeric columns
    #     time_series_columns=["close", "volume"],
    #     tickers=args.assets,
    #     save_dir=str(visualizations_dir),
    #     output_prefix="val_data"
    # )
    
    # Save the data snapshots for future reference
    data_dir = experiment_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Save sample datasets
    logger.info("Saving data...")
    raw_train_data.to_csv(data_dir / "raw_train_data.csv", index=False)
    train_data.to_csv(data_dir / "normalized_train_data.csv", index=False)
    raw_val_data.to_csv(data_dir / "raw_val_data.csv", index=False)
    val_data.to_csv(data_dir / "normalized_val_data.csv", index=False)
    
    # Save data info
    with open(data_dir / "data_info.json", 'w') as f:
        json.dump({
            "train_data": {
                "shape": train_data.shape,
                "date_range": [
                    train_data["date"].min().strftime("%Y-%m-%d"),
                    train_data["date"].max().strftime("%Y-%m-%d")
                ],
                "tickers": train_data["ticker"].unique().tolist(),
            },
            "val_data": {
                "shape": val_data.shape,
                "date_range": [
                    val_data["date"].min().strftime("%Y-%m-%d"),
                    val_data["date"].max().strftime("%Y-%m-%d")
                ],
                "tickers": val_data["ticker"].unique().tolist(),
            }
        }, f, indent=4)
    
    # Create environment configuration dictionaries
    env_config = {
        "env_params": ENV_PARAMS,
        "friction_params": MARKET_FRIC_PARAMS,
        "constraint_params": CONSTRAINT_PARAMS,
        "reward_params": REWARD_PARAMS["returns_based"],
        "columns": columns
    }
    
    # Save environment configuration
    with open(config_dir / "environment_config.json", 'w') as f:
        json.dump(env_config, f, indent=4)
    
    # Create environments
    logger.info("Creating training and validation environments...")
    train_env = TradingEnv(
        processed_data=train_data,
        raw_data=raw_train_data,
        columns=columns,
        env_params=ENV_PARAMS,
        friction_params=MARKET_FRIC_PARAMS,
        constraint_params=CONSTRAINT_PARAMS,
        reward_params=REWARD_PARAMS["returns_based"],
        render_mode=None,
    )
    
    val_env = TradingEnv(
        processed_data=val_data,
        raw_data=raw_val_data,
        columns=columns,
        env_params=ENV_PARAMS,
        friction_params=MARKET_FRIC_PARAMS,
        constraint_params=CONSTRAINT_PARAMS,
        reward_params=REWARD_PARAMS["returns_based"],
        render_mode=None,
    )
    
    # Create agent configuration dictionary
    agent_config = {
        "agent_type": args.agent_type,
        "observation_processor": "TradingObservationProcessor"
    }
    
    # Add agent-specific parameters based on the agent type
    if args.agent_type in ["DirectionalDQNAgent", "DQNAgent"]:
        agent_config.update({
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.999,
            "target_update": 10,
            "memory_size": 10000,
            "batch_size": 256,
        })
    elif args.agent_type == "PPOAgent":
        agent_config.update({
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_param": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "batch_size": 64,
            "epochs": 10,
        })
    elif args.agent_type == "A2CAgent":
        agent_config.update({
            "learning_rate": 0.0007,
            "gamma": 0.99,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
        })
    
    # Save agent configuration
    with open(config_dir / "agent_config.json", 'w') as f:
        json.dump(agent_config, f, indent=4)
    
    # Create agent based on agent type
    logger.info(f"Initializing {args.agent_type}...")
    
    if args.agent_type == "DirectionalDQNAgent":
        agent = DirectionalDQNAgent(
            env=train_env,
            observation_processor_class=TradingObservationProcessor,
            learning_rate=agent_config["learning_rate"],
            gamma=agent_config["gamma"],
            epsilon_start=agent_config["epsilon_start"],
            epsilon_end=agent_config["epsilon_end"],
            epsilon_decay=agent_config["epsilon_decay"],
            target_update=agent_config["target_update"],
            memory_size=agent_config["memory_size"],
            batch_size=agent_config["batch_size"],
        )
    elif args.agent_type == "DQNAgent":
        agent = DQNAgent(
            env=train_env,
            observation_processor_class=TradingObservationProcessor,
            learning_rate=agent_config["learning_rate"],
            gamma=agent_config["gamma"],
            epsilon_start=agent_config["epsilon_start"],
            epsilon_end=agent_config["epsilon_end"],
            epsilon_decay=agent_config["epsilon_decay"],
            target_update=agent_config["target_update"],
            memory_size=agent_config["memory_size"],
            batch_size=agent_config["batch_size"],
        )
    elif args.agent_type == "PPOAgent":
        agent = PPOAgent(
            env=train_env,
            observation_processor_class=TradingObservationProcessor,
            learning_rate=agent_config["learning_rate"],
            gamma=agent_config["gamma"],
            gae_lambda=agent_config["gae_lambda"],
            clip_param=agent_config["clip_param"],
            value_loss_coef=agent_config["value_loss_coef"],
            entropy_coef=agent_config["entropy_coef"],
            max_grad_norm=agent_config["max_grad_norm"],
            batch_size=agent_config["batch_size"],
            epochs=agent_config["epochs"],
        )
    elif args.agent_type == "A2CAgent":
        agent = A2CAgent(
            env=train_env,
            observation_processor_class=TradingObservationProcessor,
            learning_rate=agent_config["learning_rate"],
            gamma=agent_config["gamma"],
            value_loss_coef=agent_config["value_loss_coef"],
            entropy_coef=agent_config["entropy_coef"],
            max_grad_norm=agent_config["max_grad_norm"],
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
    
    # Create and save experiment configuration
    experiment_config = {
        "experiment_name": args.experiment_name,
        "assets": args.assets,
        "train_start_date": args.train_start_date,
        "train_end_date": args.train_end_date,
        "val_start_date": args.val_start_date,
        "val_end_date": args.val_end_date,
        "processors": args.processors,
        "normalize": args.normalize,
        "max_train_time": 86400,  # 24 hours
        "eval_interval": 10,
        "save_interval": 50,
        "n_eval_episodes": 5,
        "early_stopping_patience": 50,
        "early_stopping_threshold": 0.01,
        "render_train": False,
        "render_eval": False,
        "agent_type": args.agent_type,
        "data_config": {
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "normalized_data_dir": "data/normalized",
            "processor_params": PROCESSOR_PARAMS,
            "normalization_params": NORMALIZATION_PARAMS
        }
    }
    
    # Save experiment config
    with open(config_dir / "experiment_config.json", 'w') as f:
        json.dump(experiment_config, f, indent=4)
    
    # Initialize experiment manager
    logger.info("Setting up experiment manager...")
    experiment_manager = ExperimentManager(
        experiment_name=args.experiment_name,
        train_env=train_env,
        val_env=val_env,
        agent=agent,
        backtester=backtester,
        max_train_time=86400,  # 24 hours
        eval_interval=10,
        save_interval=50,
        n_eval_episodes=5,
        early_stopping_patience=50,
        early_stopping_threshold=0.01,
        render_train=False,
        render_eval=False,
        base_dir=args.output_dir,
        save_data_visualization=False,
    )
    
    # Save the experiment manager using pickle
    logger.info("Saving experiment manager for later training...")
    with open(experiment_dir / "experiment_manager.pkl", 'wb') as f:
        pickle.dump(experiment_manager, f)
    
    logger.info(f"Experiment setup completed: {args.experiment_name}")
    logger.info(f"All required files have been saved to {experiment_dir}")
    logger.info("The experiment is ready to be trained with:")
    logger.info(f"  python -m scripts.train_experiment --experiment-name {args.experiment_name}")
    logger.info("To continue training from a checkpoint later, use:")
    logger.info(f"  python -m scripts.continue_experiment --experiment-name {args.experiment_name}")

if __name__ == "__main__":
    main() 