# External imports
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import sys
import os
import pandas as pd
from datetime import datetime

# Add the project root to the Python path
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

# Internal imports
from environments.trading_env import TradingEnv
from models.agents.dqn_agent import DQNAgent
from models.agents.directional_dqn_agent import DirectionalDQNAgent
from models.processors.trading_processor import TradingObservationProcessor
from data.data_manager import DataManager
from environments.trading_env import TradingEnv

from config.data import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    PROCESSOR_PARAMS,
    NORMALIZATION_PARAMS,
)
from config.env import ENV_PARAMS, MARKET_FRIC_PARAMS, CONSTRAINT_PARAMS, REWARD_PARAMS

def train_dqn(
    env: TradingEnv,
    agent: DQNAgent,
    n_episodes: int = 100,
    eval_interval: int = 10,
    n_eval_episodes: int = 5,
    render: bool = False
):
    """
    Train the DQN agent.
    
    Args:
        env: Trading environment
        agent: DQN agent
        n_episodes: Number of training episodes
        eval_interval: Number of episodes between evaluations
        n_eval_episodes: Number of episodes for evaluation
        render: Whether to render the environment during training
    """
    # Training metrics
    metrics = defaultdict(list)
    
    # Training loop
    for episode in tqdm(range(n_episodes), desc="Training"):
        observation = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Select action
            intended_action = agent.get_intended_action(observation, current_position=env.get_current_position())
            
            # Take action in environment
            next_observation, reward, done, info = env.step(intended_action)
            
            # Store experience
            agent.add_to_memory(
                observation=observation,
                action=intended_action,
                reward=reward,
                next_observation=next_observation,
                done=done
            )

            # Update agent information in the environment for visualization
            if render:
                agent_info = agent.get_info()
                env.update_agent_info(agent_info)
                env.render()
            
            # Update agent if enough samples
            if len(agent.memory) >= agent.batch_size:
                batch = agent.get_batch()
                update_metrics = agent.update(batch)
                metrics["loss"].append(update_metrics["loss"])
                metrics["epsilon"].append(update_metrics["epsilon"])
            
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
            
            observation = next_observation
        
        # Log episode metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_steps"].append(episode_steps)
        
        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_rewards = []
            for _ in range(n_eval_episodes):
                observation = env.reset()
                episode_reward = 0
                while True:
                    intended_action = agent.get_intended_action(observation, current_position=env.get_current_position(), deterministic=True)
                    observation, reward, done, _ = env.step(intended_action)
                    episode_reward += reward
                    if done:
                        break
                eval_rewards.append(episode_reward)
            
            metrics["eval_rewards"].append(np.mean(eval_rewards))
            print(f"\nEpisode {episode + 1}")
            print(f"Average Reward: {np.mean(metrics['episode_rewards'][-eval_interval:]):.2f}")
            print(f"Average Eval Reward: {np.mean(eval_rewards):.2f}")
            print(f"Epsilon: {metrics['epsilon'][-1]:.2f}")
            print(f"Loss: {metrics['loss'][-1]:.2f}")
    
    return metrics

if __name__ == "__main__":
    # Data manager
    data_manager = DataManager(
        data_dir="data/raw",
        cache_dir="data/processed",
        save_raw_data=True,
        use_cache=True,
    )

    # Define test assets
    test_assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Download and process data
    raw_data = data_manager.download_data(
        tickers=test_assets,
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        source="yahoo",
        time_interval="1d",
        add_day_index_bool=True,
        force_download=True,
    )

    # Process data with technical indicators
    processed_data = data_manager.process_data(
        data=raw_data,
        processor_params=PROCESSOR_PARAMS,
    )

    # Normalize data
    normalized_data = data_manager.normalize_data(
        data=processed_data, method=NORMALIZATION_PARAMS["method"]
    )
    
    # Create environment
    columns = {
        "ticker": "ticker",
        "price": "close",
        "day": "day",
        "ohlcv": ["open", "high", "low", "close", "volume"],
        "tech_cols": [col for col in processed_data.columns if col not in ["ticker", "day", "open", "high", "low", "close", "volume", "date", "timestamp"]],
    }

    # Set visualization flag
    use_visualization = True

    # Create environment with render_mode if visualization is enabled
    env = TradingEnv(
        processed_data=normalized_data,
        raw_data=raw_data,
        columns=columns,
        env_params=ENV_PARAMS,
        friction_params=MARKET_FRIC_PARAMS,
        constraint_params=CONSTRAINT_PARAMS,
        reward_params=REWARD_PARAMS["returns_based"],
        seed=42,
        render_mode="human" if use_visualization else None,
    )
    
    # Create agent
    agent = DirectionalDQNAgent(
        env=env,
        observation_processor_class=TradingObservationProcessor,
    )

    # Train agent
    metrics = train_dqn(
        env=env,
        agent=agent,
        n_episodes=100,
        eval_interval=10,
        n_eval_episodes=5,
        render=use_visualization
    )