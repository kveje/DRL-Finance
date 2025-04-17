import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging first, before any other imports
from utils.setup_logging import setup_logging
logger = setup_logging(
    log_filename="continue_experiment.log",
    console_level=logging.INFO
)

# Import other modules
from models.experiment_manager import ExperimentManager


def parse_args():
    parser = argparse.ArgumentParser(description="Continue a previously set up experiment from a checkpoint")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment to continue")
    parser.add_argument("--n-episodes", type=int, default=1000000, help="Maximum number of additional episodes to train")
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
            log_filename="continue_experiment.log",
            console_level=logging.INFO
        )
        logger.error(f"Experiment directory not found: {experiment_path}")
        logger.error("Please set up the experiment first using setup_experiment.py")
        return
    
    # Now set up experiment-specific logging
    logger = setup_logging(
        log_filename=f"continue_{args.experiment_name}.log",
        console_level=logging.INFO,
        experiment_dir=str(experiment_path),
        experiment_name=f"continue_{args.experiment_name}"
    )
    
    # Check for model checkpoints
    checkpoint_pattern = "model_episode_*.pt"
    checkpoint_files = list(experiment_path.glob(f"**/{checkpoint_pattern}"))
    
    if not checkpoint_files:
        logger.error(f"No model checkpoints found in {experiment_path}")
        logger.error("The experiment must have been trained at least once before continuing")
        return
    
    # Find the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(str(x).split("_episode_")[1].split(".")[0]))
    latest_episode = int(str(latest_checkpoint).split("_episode_")[1].split(".")[0])
    
    logger.info(f"Continuing experiment: {args.experiment_name}")
    logger.info(f"Found latest checkpoint at episode {latest_episode}")
    logger.info(f"Additional episodes: {args.n_episodes}")
    logger.info(f"Maximum training time: {timedelta(seconds=args.max_train_time)}")
    
    try:
        # Use the continue_experiment method to recreate the experiment
        experiment_manager = ExperimentManager.continue_experiment(
            experiment_dir=str(experiment_path),
            additional_episodes=args.n_episodes,
            max_train_time=args.max_train_time
        )
        logger.info("Successfully loaded experiment from checkpoint")
    except Exception as e:
        logger.error(f"Failed to continue experiment: {str(e)}")
        raise
    
    # Start training
    logger.info("Starting training from checkpoint...")
    try:
        metrics = experiment_manager.train(n_episodes=args.n_episodes)
        
        # Print final metrics
        logger.info("Training completed!")
        if metrics:
            logger.info(f"Total episodes: {len(metrics.get('episode', []))}")
            if metrics.get('episode_reward'):
                logger.info(f"Final training reward: {metrics['episode_reward'][-1]:.4f}")
            if metrics.get('eval_return'):
                logger.info(f"Final evaluation return: {metrics['eval_return'][-1]:.4f}")
            if metrics.get('sharpe_ratio'):
                logger.info(f"Final Sharpe ratio: {metrics['sharpe_ratio'][-1]:.4f}")
            if metrics.get('total_return'):
                logger.info(f"Final total return: {metrics['total_return'][-1]:.2%}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main() 