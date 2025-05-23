import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging first, before any other imports
from utils.setup_logging import setup_logging
logger = setup_logging(
    log_filename="start_experiment.log",
    console_level=logging.INFO
)

# Now import other modules
from models.experiment_manager import ExperimentManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train a previously set up experiment")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment to train")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from the latest checkpoint")
    parser.add_argument("--n-episodes", type=int, default=1000000, help="Maximum number of episodes to train")
    parser.add_argument("--max-train-time", type=int, default=86400, help="Maximum training time in seconds (24h default)")
    parser.add_argument("--experiments-dir", type=str, default="experiments", help="Directory containing experiments")
    parser.add_argument("--agent-type", type=str, 
                        choices=["DirectionalDQNAgent", "DQNAgent", "PPOAgent", "A2CAgent", "DiscreteDQNAgent", "DDPGAgent"],
                        help="Override the agent type from the experiment config")
    
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
        experiment_dir=str(experiment_path)
    )
    
    logger.info(f"Training experiment: {args.experiment_name}")
    logger.info(f"Maximum episodes: {args.n_episodes}")
    logger.info(f"Maximum training time: {args.max_train_time} seconds")
    
    if args.continue_training:
        logger.info("Continuing training from the latest checkpoint")
        try:
            # Use the improved continue_experiment method to recreate the experiment
            experiment_manager = ExperimentManager.continue_experiment(
                experiment_dir=str(experiment_path),
                additional_episodes=args.n_episodes,
                max_train_time=args.max_train_time
            )
            logger.info("Successfully continued experiment from checkpoint")
        except Exception as e:
            logger.error(f"Failed to continue experiment: {str(e)}")
            return
    else:
        # Look for config files in the config directory
        config_dir = experiment_path / "config"
        config_path = config_dir / "experiment_config.json"
        
        # Look for experiment_manager.py in the experiment directory
        experiment_file = experiment_path / "experiment_manager.pkl"
        
        if experiment_file.exists():
            import pickle
            with open(experiment_file, 'rb') as f:
                experiment_manager = pickle.load(f)
                
            logger.info(f"Loaded experiment manager from {experiment_file}")
            
            # Update max training time if provided
            if args.max_train_time:
                experiment_manager.max_train_time = args.max_train_time
                logger.info(f"Updated max training time to {args.max_train_time} seconds")
            
            # Override agent type if specified
            if args.agent_type:
                logger.info(f"Overriding agent type to {args.agent_type}")
                # Get the current agent's environment
                env = experiment_manager.train_env
                
                # Create new agent of specified type
                if args.agent_type == "DDPGAgent":
                    from models.agents.ddpg_agent import DDPGAgent
                    agent = DDPGAgent(
                        env=env,
                        observation_processor_class=TradingObservationProcessor,
                        learning_rate_actor=0.0001,
                        learning_rate_critic=0.001,
                        gamma=0.99,
                        tau=0.001,
                        memory_size=100000,
                        batch_size=64,
                    )
                else:
                    # Handle other agent types...
                    logger.error(f"Agent type override not implemented for {args.agent_type}")
                    return
                
                # Update experiment manager with new agent
                experiment_manager.agent = agent
                logger.info(f"Updated experiment manager with new {args.agent_type} agent")
        else:
            logger.error(f"Experiment manager file not found: {experiment_file}")
            logger.error("Please ensure that setup_experiment.py created this file")
            return
    
    # Start training
    logger.info("Starting training...")
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