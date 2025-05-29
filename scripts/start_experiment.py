import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import json
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging first, before any other imports
from utils.setup_logging import setup_logging
logger = setup_logging(
    log_filename="start_experiment.log",
    console_level=logging.INFO
)

# Now import other modules
from managers.experiment_manager import ExperimentManager
from environments.trading_env import TradingEnv
from data.data_manager import DataManager
from models.agents.agent_factory import AgentFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Start training a previously set up experiment")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment to train")
    parser.add_argument("--n-episodes", type=int, default=1000, help="Maximum number of episodes to train")
    parser.add_argument("--max-train-time", type=int, default=86400, help="Maximum training time in seconds (24h default)")
    parser.add_argument("--experiments-dir", type=str, default="experiments", help="Directory containing experiments")
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    experiment_path = Path(args.experiments_dir) / args.experiment_name
    
    if not experiment_path.exists():
        # Use default logging before we know experiment exists
        logger = setup_logging(
            log_filename="start_experiment.log",
            console_level=logging.INFO
        )
        logger.error(f"Experiment directory not found: {experiment_path}")
        logger.error("Please set up the experiment first using setup_experiment.py")
        return
    
    # Now set up experiment-specific logging
    logger = setup_logging(
        log_filename=f"{args.experiment_name}.log",
        console_level=logging.INFO,
        experiment_dir=str(experiment_path),
        experiment_name=args.experiment_name
    )
    
    logger.info(f"Starting training for experiment: {args.experiment_name}")
    logger.info(f"Maximum episodes: {args.n_episodes}")
    logger.info(f"Maximum training time: {args.max_train_time} seconds")
    
    try:
        # Load experiment data
        experiment_data = ExperimentManager.load_experiment(str(experiment_path))
        configs = experiment_data["config"]
        experiment_config = configs["experiment"]
        
        # Create environments
        logger.info("Creating training and validation environments...")
        train_env = TradingEnv(
            processed_data=experiment_data["train_data"],
            raw_data=experiment_data["raw_train_data"],
            columns=experiment_data["data_info"]["columns_mapping"],
            env_params=configs["environment"].get("env_params", {}),
            friction_params=configs["environment"].get("friction_params", {}),
            constraint_params=configs["environment"].get("constraint_params", {}),
            reward_params=configs["environment"].get("reward_params", {}),
            processor_configs=configs["environment"].get("processor_configs", {}),
            render_mode=None,
        )
        
        val_env = TradingEnv(
            processed_data=experiment_data["val_data"],
            raw_data=experiment_data["raw_val_data"],
            columns=experiment_data["data_info"]["columns_mapping"],
            env_params=configs["environment"].get("env_params", {}),
            friction_params=configs["environment"].get("friction_params", {}),
            constraint_params=configs["environment"].get("constraint_params", {}),
            reward_params=configs["environment"].get("reward_params", {}),
            processor_configs=configs["environment"].get("processor_configs", {}),
            render_mode=None,
        )
        
        # Create agent
        logger.info("Creating agent...")
        agent = AgentFactory.create_agent(
            agent_type=experiment_config.get("agent_type", "dqn").lower(),
            update_frequency=experiment_config.get("update_frequency", 1),
            env=train_env,
            **configs["agent"]  # Pass all agent config parameters
        )
        
        # Create experiment manager
        experiment_manager = ExperimentManager(
            experiment_name=args.experiment_name,
            train_env=train_env,
            val_env=val_env,
            agent=agent,
            max_train_time=args.max_train_time,
            eval_interval=1, # experiment_config.get("eval_interval", 10),
            save_interval=1, #experiment_config.get("save_interval", 50),
            save_metric_interval=1, # experiment_config.get("save_metric_interval", 5),
            early_stopping_patience=experiment_config.get("early_stopping_patience", 50),
            early_stopping_threshold=experiment_config.get("early_stopping_threshold", 0.01),
            early_stopping_metric=experiment_config.get("early_stopping_metric", "sharpe_ratio"),
            render_train=False,
            render_eval=False,
            base_dir=args.experiments_dir,
            is_setup=False  # This is loading an existing experiment
        )
        logger.info("Successfully loaded experiment")
    except Exception as e:
        logger.error(f"Failed to load experiment: {str(e)}")
        raise
    
    # Start training
    logger.info("Starting training...")
    try:
        metrics = experiment_manager.train(n_episodes=args.n_episodes)
        
        # Print final metrics
        logger.info("Training completed!")
        if metrics:
            logger.info(f"Total episodes: {experiment_manager.current_episode + 1}")
            if metrics.get('episode_reward'):
                logger.info(f"Final training reward: {metrics['episode_reward']:.4f}")
            if metrics.get('eval_return'):
                logger.info(f"Final evaluation return: {metrics['eval_return']:.4f}")
            if metrics.get('sharpe_ratio'):
                logger.info(f"Final Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
            if metrics.get('total_return'):
                logger.info(f"Final total return: {metrics['total_return']:.2%}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main() 