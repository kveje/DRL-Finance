import sys
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import torch

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
from models.agents.agent_factory import AgentFactory
from managers.experiment_manager import ExperimentManager
from environments.trading_env import TradingEnv
from visualization.data_visualization import DataVisualization
from config.networks import get_network_config
from config.temperature import get_temperature_config

# Config imports
from config.data import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    VIX_PARAMS,
    TURBULENCE_PARAMS,
    NORMALIZATION_PARAMS,
    NO_INDICATOR_PARAMS,
    SIMPLE_INDICATOR_PARAMS,
    ADVANCED_INDICATOR_PARAMS,
    ALL_INDICATOR_PARAMS,
)
from config.models import (
    A2C_PARAMS,
    PPO_PARAMS,
    DQN_PARAMS,
    SAC_PARAMS,
)
from config.env import ENV_PARAMS, MARKET_FRIC_PARAMS, CONSTRAINT_PARAMS, get_processor_config, get_reward_config
from config.tickers import DOW_30_TICKER, NASDAQ_100_TICKER, SP_500_TICKER
from config.interpreter import get_interpreter_config


def parse_args():
    parser = argparse.ArgumentParser(description="Setup an experiment with data visualization and exploration")
    parser.add_argument("--experiment-name", type=str, default=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--assets", type=str, nargs="+", default=DOW_30_TICKER)
    parser.add_argument("--train-start-date", type=str, default=TRAIN_START_DATE)
    parser.add_argument("--train-end-date", type=str, default=TRAIN_END_DATE)
    parser.add_argument("--val-start-date", type=str, default=TEST_START_DATE)
    parser.add_argument("--val-end-date", type=str, default=TEST_END_DATE)
    parser.add_argument("--use-vix", action="store_true", default=False,
                        help="Whether to include VIX data in the experiment")
    parser.add_argument("--use-turbulence", action="store_true", default=False,
                        help="Whether to include market turbulence data in the experiment")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--agent-type", type=str, 
                        choices=["dqn", "ppo", "a2c", "sac"], 
                        default="dqn", 
                        help="Type of agent to use for the experiment")
    parser.add_argument("--interpreter-type", type=str,
                        choices=["discrete", "confidence_scaled"],
                        default="discrete",
                        help="Type of action interpreter to use")
    parser.add_argument("--use-bayesian", action="store_true", default=False,
                        help="Whether to use Bayesian network heads")
    parser.add_argument("--visualize-data", action="store_true", default=False, 
                        help="Whether to generate data visualizations")
    parser.add_argument("--by-ticker", action="store_true", default=False,
                        help="Whether to generate stock-specific visualizations")
    parser.add_argument("--indicator-type", type=str,
                        choices=["no", "simple", "advanced", "full"],
                        default="no",
                        help="Type of technical indicators to use")
    parser.add_argument("--price-type", type=str,
                        choices=["price", "ohlcv", "both"],
                        default="price",
                        help="Type of price data to use")
    parser.add_argument("--reward-type", type=str,
                        choices=["returns", "log_returns", "risk_adjusted"],
                        default="return",
                        help="Type of reward to use")
    parser.add_argument("--reward-projection-period", type=int,
                        default=0,
                        help="Delay in reward calculation")
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create experiment directory
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Select technical indicator parameters based on indicator type
    if args.indicator_type == "no":
        indicator_params = NO_INDICATOR_PARAMS
    elif args.indicator_type == "simple":
        indicator_params = SIMPLE_INDICATOR_PARAMS
    elif args.indicator_type == "advanced":
        indicator_params = ADVANCED_INDICATOR_PARAMS
    else:  # full
        indicator_params = ALL_INDICATOR_PARAMS
    
    # Build processor list based on flags
    processors = ["technical_indicator"] if args.indicator_type != "no" else []
    if args.use_vix:
        processors.append("vix")
    if args.use_turbulence:
        processors.append("turbulence")
    
    # Update processor parameters with selected indicator parameters
    processor_params = {}
    if "technical_indicator" in processors:
        processor_params["technical_indicator"] = indicator_params
    if "vix" in processors:
        processor_params["vix"] = VIX_PARAMS
    if "turbulence" in processors:
        processor_params["turbulence"] = TURBULENCE_PARAMS
    
    # Process with technical indicators
    logger.info("Processing data with technical indicators...")
    processed_data = data_manager.process_data(
        data=raw_data,
        processors=processors,
        processor_params=processor_params,
        save_data=True,
    )
    
    # Normalize data
    logger.info("Normalizing data...")
    normalized_data = data_manager.simple_normalize_data(processed_data)

    # Split the data into training and validation sets
    logger.info("Splitting data into training and validation sets...")
    normalized_train_data, normalized_val_data = data_manager.split_data(
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

    # Split the processed data exactly the same way for consistent behavior
    processed_train_data, processed_val_data = data_manager.split_data(
        data=processed_data,
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
        "tech_cols": [col for col in normalized_train_data.columns if col not in ["ticker", "day", "open", "high", "low", "close", "volume", "date", "timestamp"]]
    }
    
    # Generate data visualizations if requested
    if args.visualize_data:
        visualizations_dir = experiment_dir / "visualizations"
        visualizations_dir.mkdir(exist_ok=True)
        
        logger.info("Generating data visualizations...")
        visualizer = DataVisualization(save_dir=str(visualizations_dir))
        
        # Generate visualizations for training data
        train_vis_prefix = "train_data"
        visualizer.visualize_all(
            raw_data=raw_train_data,
            processed_data=processed_train_data,
            normalized_data=normalized_train_data,
            columns=None,  # Use all numeric columns
            tickers=args.assets,
            output_prefix=train_vis_prefix,
            by_ticker=args.by_ticker,  # Option for stock-specific plots
            max_features_per_plot=4  # Limit features per plot for readability
        )
        
        # Generate visualizations for validation data
        val_vis_prefix = "val_data"
        visualizer.visualize_all(
            raw_data=raw_val_data,
            processed_data=processed_val_data,
            normalized_data=normalized_val_data,
            columns=None,  # Use all numeric columns
            tickers=args.assets,
            output_prefix=val_vis_prefix,
            by_ticker=args.by_ticker,  # Option for stock-specific plots
            max_features_per_plot=4  # Limit features per plot for readability
        )
        
        # Generate comparison between training and validation data
        visualizer.compare_train_val_datasets(
            train_data=normalized_train_data,
            val_data=normalized_val_data,
            key_columns=["close", "volume"],
            additional_columns=["open", "high", "low"],
            save_filename_prefix="train_val"
        )
        
        logger.info(f"Data visualizations saved to {visualizations_dir}")
    
    # Create processor
    processor_config = get_processor_config(
        price_type=args.price_type,
        n_assets=len(args.assets),
        asset_list=args.assets,
        tech_cols=columns["tech_cols"] if args.indicator_type != "no" else None,
        price_col="close",
    )
    
    # Create environments
    logger.info("Creating training and validation environments...")

    reward_params = get_reward_config(args.reward_type, None if args.reward_projection_period == 0 else args.reward_projection_period)
    train_env = TradingEnv(
        processed_data=normalized_train_data,
        raw_data=raw_train_data,
        columns=columns,
        env_params=ENV_PARAMS,
        friction_params=MARKET_FRIC_PARAMS,
        constraint_params=CONSTRAINT_PARAMS,
        reward_params=reward_params,
        render_mode=None,
        processor_configs=processor_config
    )
    
    val_env = TradingEnv(
        processed_data=normalized_val_data,
        raw_data=raw_val_data,
        columns=columns,
        env_params=ENV_PARAMS,
        friction_params=MARKET_FRIC_PARAMS,
        constraint_params=CONSTRAINT_PARAMS,
        reward_params=reward_params,
        render_mode=None,
        processor_configs=processor_config
    )

    # Create interpreter configuration
    interpreter_config = get_interpreter_config(
        type = args.interpreter_type,
        n_assets = len(args.assets),
    )

    # Create network configuration
    network_config = get_network_config(
        n_assets=len(args.assets),
        window_size=train_env.window_size,
        price_type=args.price_type,
        head_type="bayesian" if args.use_bayesian else "parametric",
        include_discrete=True,
        include_confidence=True if args.interpreter_type == "confidence_scaled" else False,
        include_value=True if args.agent_type in ["ppo", "a2c", "sac"] else False,
        technical_dim = len(columns["tech_cols"]) if args.indicator_type != "no" else None
    )

    # Log the selected configurations
    logger.info(f"Using {args.agent_type} agent")
    logger.info(f"Using {args.interpreter_type} interpreter")
    logger.info(f"Using {'Bayesian' if args.use_bayesian else 'Parametric'} network heads")
    if args.interpreter_type == "confidence_scaled":
        logger.info("Network includes both discrete and confidence heads")
    elif args.agent_type in ["ppo", "a2c", "sac"]:
        logger.info("Network includes value heads for policy evaluation")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create agent configuration dictionary
    agent_config = {
        "agent_type": args.agent_type,
        "interpreter_type": args.interpreter_type,
        "network_config": network_config,
        "interpreter_config": interpreter_config,
    }
    
    # Add agent-specific parameters based on the agent type
    if args.agent_type == "dqn":
        agent_config.update(DQN_PARAMS)
    elif args.agent_type == "ppo":
        agent_config.update(PPO_PARAMS)
    elif args.agent_type == "a2c":
        agent_config.update(A2C_PARAMS)
    elif args.agent_type == "sac":
        agent_config.update(SAC_PARAMS)
    
    # Add temperature configuration
    temperature_config = get_temperature_config(network_config, update_frequency=agent_config["update_frequency"])
    agent_config["temperature_config"] = temperature_config

    # add use_bayesian to agent_config
    if args.use_bayesian:
        agent_config["use_bayesian"] = True
    else:
        agent_config["use_bayesian"] = False

    # Create agent using factory
    agent = AgentFactory.create_agent(
        agent_type=args.agent_type,
        network_config=network_config,
        interpreter_type=args.interpreter_type,
        interpreter_config=interpreter_config,
        temperature_config=temperature_config,
        device=device,  # Pass the torch.device object directly
        **{k: v for k, v in agent_config.items() if k not in [
            "agent_type", "interpreter_type", "network_config", "interpreter_config", "temperature_config"
        ]}  # Only pass additional parameters, not the ones already specified
    )
    
    # Initialize experiment manager
    logger.info("Setting up experiment manager...")
    experiment_manager = ExperimentManager(
        experiment_name=args.experiment_name,
        train_env=train_env,
        val_env=val_env,
        agent=agent,
        max_train_time=86400,  # 24 hours
        eval_interval=10,
        save_interval=50,
        save_metric_interval=10,
        early_stopping_patience=50,
        early_stopping_threshold=0.01,
        early_stopping_metric="sharpe_ratio",  # Default to sharpe ratio for early stopping
        render_train=False,
        render_eval=False,
        base_dir=args.output_dir,
        is_setup=True  # This is a new experiment setup
    )
    
    logger.info(f"Experiment setup completed: {args.experiment_name}")
    logger.info(f"All required files have been saved to {experiment_dir}")
    logger.info("The experiment is ready to be trained with:")
    logger.info(f"  python -m scripts.train_experiment --experiment-name {args.experiment_name}")
    logger.info("To continue training from a checkpoint later, use:")
    logger.info(f"  python -m scripts.continue_experiment --experiment-name {args.experiment_name}")

if __name__ == "__main__":
    main() 