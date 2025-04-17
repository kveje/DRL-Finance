import os
import time
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import shutil

from models.backtesting import Backtester
from models.agents.base_agent import BaseAgent
from environments.trading_env import TradingEnv
from visualization.trading_visualizer import TradingVisualizer
from utils.logger import Logger, LogConfig
from models.agents.dqn_agent import DQNAgent
from models.agents.directional_dqn_agent import DirectionalDQNAgent
from models.agents.ppo_agent import PPOAgent
from models.agents.a2c_agent import A2CAgent

# Add import for data visualization
try:
    from visualization.data_visualization import DataVisualization
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False
    logger = Logger.get_logger("experiment_manager")
    logger.warning("DataVisualization class not found. Data visualization features will be disabled.")

class ExperimentManager:
    """
    Manages long-term training experiments with periodic evaluation, metrics tracking,
    and model checkpointing. Designed for running experiments over extended periods.
    """
    
    def __init__(
        self,
        experiment_name: str,
        train_env: TradingEnv,
        val_env: TradingEnv,
        agent: BaseAgent,
        backtester: Backtester,
        max_train_time: Optional[int] = None,  # Max training time in seconds (None for unlimited)
        eval_interval: int = 10,  # Episodes between evaluations
        save_interval: int = 5,  # Episodes between model saves
        n_eval_episodes: int = 5,  # Number of episodes for evaluation
        early_stopping_patience: int = 20,  # Episodes with no improvement before stopping
        early_stopping_threshold: float = 0.01,  # Minimum improvement to reset patience
        metrics_to_track: List[str] = [
            "episode_reward", "loss", "epsilon", "portfolio_value", 
            "sharpe_ratio", "max_drawdown", "total_return"
        ],
        render_train: bool = False,  # Whether to render training environment
        render_eval: bool = True,  # Whether to render evaluation environment
        base_dir: str = "experiments",  # Base directory for saving experiment data
        save_data_visualization: bool = True,  # Whether to save data visualizations
    ):
        """
        Initialize the experiment manager.
        
        Args:
            experiment_name: Name of the experiment
            train_env: Training environment
            val_env: Validation environment
            agent: Agent to train
            backtester: Backtester for evaluation
            max_train_time: Maximum training time in seconds (None for unlimited)
            eval_interval: Number of episodes between evaluations
            save_interval: Number of episodes between model saves
            n_eval_episodes: Number of episodes for evaluation
            early_stopping_patience: Episodes with no improvement before stopping
            early_stopping_threshold: Minimum improvement to reset patience
            metrics_to_track: List of metrics to track during training
            render_train: Whether to render training environment
            render_eval: Whether to render evaluation environment
            base_dir: Base directory for saving experiment data
            save_data_visualization: Whether to save data visualizations
        """
        self.experiment_name = experiment_name
        self.train_env = train_env
        self.val_env = val_env
        self.agent = agent
        self.backtester = backtester
        self.max_train_time = max_train_time
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.n_eval_episodes = n_eval_episodes
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metrics_to_track = metrics_to_track
        self.render_train = render_train
        self.render_eval = render_eval
        self.save_data_visualization = save_data_visualization
        
        # Create experiment directory
        self.experiment_dir = Path(base_dir) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging to use the experiment directory
        Logger.configure(
            EXPERIMENT_NAME=experiment_name,
            EXPERIMENT_DIR=str(self.experiment_dir)
        )
        
        # Get a logger for this experiment
        self.logger = Logger.get_logger(name=f"experiment_{experiment_name}")
        
        # Create subdirectories
        self.data_dir = self.experiment_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.config_dir = self.experiment_dir / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.visualizations_dir.mkdir(exist_ok=True)
        
        # Create visualization directory for backtests
        if self.backtester:
            # Set visualization directory for the backtester to be within the experiment directory
            backtest_vis_dir = self.experiment_dir / "backtest_visualizations"
            backtest_vis_dir.mkdir(exist_ok=True)
            self.backtester.visualization_dir = str(backtest_vis_dir)
            self.backtester.save_visualizations = True
        
        # Initialize metrics storage
        self.metrics = {metric: [] for metric in metrics_to_track}
        self.metrics["episode"] = []
        self.metrics["episode_length"] = []
        self.metrics["timestamp"] = []
        self.metrics["eval_return"] = []
        
        # Initialize early stopping tracking
        self.best_eval_return = -float('inf')
        self.patience_counter = 0
        
        # Initialize training state
        self.current_episode = 0
        self.total_steps = 0
        self.start_time = None
        self.is_finished = False
        
        # Save initial setup
        self._save_config()
        self._save_data_info()
        
        # Generate and save data visualizations
        if self.save_data_visualization:
            self._save_data_visualizations()
    
        
        self.logger.info(f"Initialized experiment: {experiment_name}")
        
    def _save_config(self):
        """Save experiment configuration to disk."""
        # Main config
        config = {
            "experiment_name": self.experiment_name,
            "max_train_time": self.max_train_time,
            "eval_interval": self.eval_interval,
            "save_interval": self.save_interval,
            "n_eval_episodes": self.n_eval_episodes,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "metrics_to_track": self.metrics_to_track,
            "render_train": self.render_train,
            "render_eval": self.render_eval,
            "agent_type": self.agent.get_model_name(),
            "train_env_params": {
                "n_assets": self.train_env.n_assets,
                "initial_balance": self.train_env.initial_balance,
                "window_size": self.train_env.window_size
            }
        }
        
        # Save to config directory
        with open(self.config_dir / "experiment_config.json", 'w') as f:
            json.dump(config, f, indent=4)
            
        # Save detailed environment configs
        # Extract the parameters directly from the env object
        env_config = {
            "env_params": getattr(self.train_env, "env_params", {}),
            "friction_params": getattr(self.train_env, "friction_params", {}),
            "constraint_params": getattr(self.train_env, "constraint_params", {}),
            "reward_params": getattr(self.train_env, "reward_params", {})
        }
        
        with open(self.config_dir / "environment_config.json", 'w') as f:
            json.dump(env_config, f, indent=4)
            
        # Save agent configuration
        agent_config = self.agent.get_config()
        with open(self.config_dir / "agent_config.json", 'w') as f:
            json.dump(agent_config, f, indent=4)
            
        self.logger.info(f"Saved experiment configuration to {self.config_dir}")
        
    def _save_data_info(self):
        """Save information about the datasets used in the experiment."""
        # Extract data info from environments
        train_data_info = {
            "shape": self.train_env.processed_data.shape,
            "columns": self.train_env.processed_data.columns.tolist(),
            "tickers": self.train_env.processed_data["ticker"].unique().tolist(),
            "date_range": [
                self.train_env.processed_data["date"].min().strftime("%Y-%m-%d"),
                self.train_env.processed_data["date"].max().strftime("%Y-%m-%d")
            ],
            "columns_mapping": self.train_env.columns
        }
        
        val_data_info = {
            "shape": self.val_env.processed_data.shape,
            "columns": self.val_env.processed_data.columns.tolist(),
            "tickers": self.val_env.processed_data["ticker"].unique().tolist(),
            "date_range": [
                self.val_env.processed_data["date"].min().strftime("%Y-%m-%d"),
                self.val_env.processed_data["date"].max().strftime("%Y-%m-%d")
            ],
            "columns_mapping": self.val_env.columns
        }
        
        # Save data info
        with open(self.data_dir / "train_data_info.json", 'w') as f:
            json.dump(train_data_info, f, indent=4)
            
        with open(self.data_dir / "val_data_info.json", 'w') as f:
            json.dump(val_data_info, f, indent=4)
            
        # Save column statistics for both datasets
        train_stats = self._compute_data_statistics(self.train_env.processed_data)
        val_stats = self._compute_data_statistics(self.val_env.processed_data)
        
        train_stats.to_csv(self.data_dir / "train_data_statistics.csv")
        val_stats.to_csv(self.data_dir / "val_data_statistics.csv")
        
        self.logger.info(f"Saved data information to {self.data_dir}")
        
    def _compute_data_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute statistics for each column in the dataset."""
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Compute statistics for each column
        stats = data[numeric_cols].describe().T
        
        # Add additional statistics
        stats['skew'] = data[numeric_cols].skew()
        stats['kurtosis'] = data[numeric_cols].kurtosis()
        stats['missing'] = data[numeric_cols].isnull().sum()
        stats['missing_pct'] = data[numeric_cols].isnull().mean() * 100
        
        return stats
        
    def _save_data_visualizations(self):
        """Generate and save visualizations of the training and validation data."""
        if not _HAS_VISUALIZATION:
            self.logger.warning("DataVisualization not available. Skipping data visualizations.")
            return
            
        self.logger.info("Generating data visualizations...")
        
        # Create data visualization object
        visualizer = DataVisualization(save_dir=str(self.visualizations_dir))
        
        # Generate visualizations for training data
        train_vis_prefix = "train_data"
        visualizer.visualize_all(
            raw_data=self.train_env.raw_data,
            processed_data=self.train_env.processed_data,
            columns=None,  # Use all numeric columns
            time_series_columns=["close", "volume"],
            tickers=self.train_env.processed_data["ticker"].unique().tolist(),
            output_prefix=train_vis_prefix
        )
        
        # Generate visualizations for validation data
        val_vis_prefix = "val_data"
        visualizer.visualize_all(
            raw_data=self.val_env.raw_data,
            processed_data=self.val_env.processed_data,
            columns=None,  # Use all numeric columns
            time_series_columns=["close", "volume"],
            tickers=self.val_env.processed_data["ticker"].unique().tolist(),
            output_prefix=val_vis_prefix
        )
        
        # Generate comparison between training and validation data
        self._generate_train_val_comparison(visualizer)
        
        self.logger.info(f"Saved data visualizations to {self.visualizations_dir}")
        
    def _generate_train_val_comparison(self, visualizer: Any):
        """Generate visualizations comparing training and validation data."""
        # Extract key tickers to compare (use the first ticker if multiple exist)
        train_tickers = self.train_env.processed_data["ticker"].unique()
        val_tickers = self.val_env.processed_data["ticker"].unique()
        
        # Find common tickers
        common_tickers = list(set(train_tickers) & set(val_tickers))
        
        if not common_tickers:
            self.logger.warning("No common tickers between training and validation data")
            return
            
        # Select the first common ticker for comparison
        ticker = common_tickers[0]
        
        # Filter data for the selected ticker
        train_ticker_data = self.train_env.processed_data[self.train_env.processed_data["ticker"] == ticker]
        val_ticker_data = self.val_env.processed_data[self.val_env.processed_data["ticker"] == ticker]
        
        # Create a figure for price comparison
        plt.figure(figsize=(12, 6))
        plt.plot(train_ticker_data["date"], train_ticker_data["close"], label="Training")
        plt.plot(val_ticker_data["date"], val_ticker_data["close"], label="Validation")
        plt.title(f"Price Comparison: {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"train_val_price_comparison_{ticker}.png")
        plt.close()
    
    def train(self, n_episodes: int = 10000):
        """
        Run the training experiment for the specified number of episodes or until max_train_time is reached.
        
        Args:
            n_episodes: Maximum number of episodes to train for
        
        Returns:
            Dictionary containing training metrics
        """
        self.start_time = time.time()
        self.logger.info(f"Starting training experiment: {self.experiment_name}")
        self.logger.info(f"Planning to train for up to {n_episodes} episodes")
        
        if self.max_train_time:
            end_time = self.start_time + self.max_train_time
            self.logger.info(f"Maximum training time: {timedelta(seconds=self.max_train_time)}")
            self.logger.info(f"Training will end by: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Main training loop
            for episode in range(n_episodes):
                self.current_episode = episode
                
                # Check if we've exceeded max training time
                if self.max_train_time and (time.time() - self.start_time) > self.max_train_time:
                    self.logger.info(f"Reached maximum training time. Stopping training.")
                    break                
                # Run a single training episode
                episode_metrics = self._run_training_episode(episode)
                
                # Update metrics
                self._update_metrics(episode_metrics)
                
                # Log episode results
                self._log_episode_results(episode, episode_metrics)
                
                # Evaluate if needed
                if (episode + 1) % self.eval_interval == 0:
                    eval_metrics = self._run_evaluation()
                    self._update_metrics(eval_metrics)
                    self._check_early_stopping(eval_metrics["eval_return"])
                    
                    # Log evaluation results
                    self._log_eval_results(episode, eval_metrics)
                
                # Save model if needed
                if (episode + 1) % self.save_interval == 0:
                    self._save_model(episode)
                
                # Check if we should stop early
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {episode + 1} episodes due to lack of improvement.")
                    break
                
                # Save metrics periodically
                if (episode + 1) % 10 == 0:
                    self._save_metrics()
                    self._plot_metrics()
            
            # Training finished
            self.is_finished = True
            self.logger.info(f"Training completed after {self.current_episode + 1} episodes")
            self.logger.info(f"Total training time: {timedelta(seconds=time.time() - self.start_time)}")
            
            # Final evaluation and save
            eval_metrics = self._run_evaluation()
            self._update_metrics(eval_metrics)
            self._save_model(self.current_episode, is_final=True)
            self._save_metrics()
            self._plot_metrics()
            
            return self.metrics
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user.")
            self._save_model(self.current_episode, is_final=False, is_interrupted=True)
            self._save_metrics()
            self._plot_metrics()
            return self.metrics
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self._save_model(self.current_episode, is_final=False, is_error=True)
            self._save_metrics()
            self._plot_metrics()
            raise
    
    def _run_training_episode(self, episode: int) -> Dict[str, float]:
        """
        Run a single training episode.
        
        Args:
            episode: Current episode number
        
        Returns:
            Dictionary with episode metrics
        """
        # Reset environment
        observation = self.train_env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_loss = []
        episode_epsilon = []
        done = False
        
        # Run episode
        while not done:
            # Select action
            intended_action = self.agent.get_intended_action(
                observation, 
                current_position=self.train_env.get_current_position()
            )
            
            # Take action in environment
            next_observation, reward, done, info = self.train_env.step(intended_action)
            
            # Store experience
            self.agent.add_to_memory(
                observation=observation,
                action=intended_action,
                reward=reward,
                next_observation=next_observation,
                done=done
            )

            # Update agent if enough samples
            if len(self.agent.memory) >= self.agent.batch_size:
                batch = self.agent.get_batch()
                update_metrics = self.agent.update(batch)
                episode_loss.append(update_metrics["loss"])
                episode_epsilon.append(update_metrics["epsilon"])
            
            # Update visualization if enabled
            if self.render_train:
                agent_info = self.agent.get_info()
                self.train_env.update_agent_info(agent_info)
                self.train_env.render()
            
            # Update tracking
            episode_reward += reward
            episode_steps += 1
            observation = next_observation
            self.total_steps += 1
        
        # Calculate final portfolio value
        final_portfolio_value = info["portfolio_value"]
        
        # Compile episode metrics
        metrics = {
            "episode": episode,
            "episode_length": episode_steps,
            "episode_reward": episode_reward,
            "portfolio_value": final_portfolio_value,
            "timestamp": time.time()
        }
        
        # Add loss and epsilon if available
        if episode_loss:
            metrics["loss"] = np.mean(episode_loss)
            metrics["epsilon"] = episode_epsilon[-1]
        
        return metrics
    
    def _run_evaluation(self) -> Dict[str, float]:
        """
        Run evaluation episodes on the validation environment.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Running evaluation after episode {self.current_episode}")
        
        # Set agent to evaluation mode
        self.agent.eval()
        
        eval_rewards = []
        eval_portfolio_values = []
        backtest_results = None
        
        # Run multiple evaluation episodes
        for eval_episode in range(self.n_eval_episodes):
            observation = self.val_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get deterministic action
                intended_action = self.agent.get_intended_action(
                    observation, 
                    current_position=self.val_env.get_current_position(),
                    deterministic=True
                )
                
                # Take action
                observation, reward, done, info = self.val_env.step(intended_action)
                episode_reward += reward
                
                # Render if enabled
                if self.render_eval:
                    agent_info = self.agent.get_info()
                    self.val_env.update_agent_info(agent_info)
                    self.val_env.render()
            
            # Record results
            eval_rewards.append(episode_reward)
            eval_portfolio_values.append(info["portfolio_value"])
        
        # Run full backtest if backtester is available
        if self.backtester:
            # Make sure backtester visualization directory is set correctly
            backtest_vis_dir = self.experiment_dir / "backtest_visualizations"
            backtest_vis_dir.mkdir(exist_ok=True)
            self.backtester.visualization_dir = str(backtest_vis_dir)
            self.backtester.save_visualizations = True
            
            # Include episode_id for the visualization
            backtest_results = self.backtester.run_backtest(
                model_state_dict=self.agent.get_model().state_dict(),
                deterministic=True,
                episode_id=self.current_episode,
                save_visualization=True,
                visualization_filename=f"backtest_ep{self.current_episode}.png"
            )
            
            # Save detailed backtest results to a separate file
            self._save_backtest_results(backtest_results)
        
        # Set agent back to training mode
        self.agent.train()
        
        # Compile evaluation metrics
        metrics = {
            "eval_return": np.mean(eval_rewards),
            "eval_portfolio_value": np.mean(eval_portfolio_values)
        }
        
        # Add backtest metrics if available
        if backtest_results:
            metrics["sharpe_ratio"] = backtest_results["sharpe_ratio"]
            metrics["max_drawdown"] = backtest_results["max_drawdown"]
            metrics["total_return"] = backtest_results["total_return"]
        
        return metrics
    
    def _save_backtest_results(self, backtest_results: Dict[str, Any]) -> None:
        """
        Save backtest results to disk.
        
        Args:
            backtest_results: Dictionary with backtest results
        """
        backtest_dir = self.experiment_dir / "backtests"
        backtest_dir.mkdir(exist_ok=True)
        
        # Create a filename with timestamp and episode number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_ep{self.current_episode}_{timestamp}.json"
        
        # Add metadata
        backtest_results["episode"] = self.current_episode
        backtest_results["timestamp"] = timestamp
        backtest_results["total_steps"] = self.total_steps
        
        # Convert numpy objects to Python native types recursively
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                  np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.complex64, np.complex128)):
                return {'real': obj.real, 'imag': obj.imag}
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(i) for i in obj]
            else:
                # If it's a non-serializable object, try to convert to string
                try:
                    return str(obj)
                except:
                    return None
        
        # Convert all backtest results to JSON serializable format
        serializable_results = convert_to_serializable(backtest_results)
        
        # Save to file
        with open(backtest_dir / filename, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        # Also save a CSV with portfolio values for easy plotting
        if "portfolio_values" in backtest_results:
            portfolio_values = convert_to_serializable(backtest_results["portfolio_values"])
            returns = convert_to_serializable(backtest_results["returns"]) if "returns" in backtest_results else []
            
            portfolio_df = pd.DataFrame({
                "step": range(len(portfolio_values)),
                "portfolio_value": portfolio_values,
                "returns": [0] + returns if returns else []
            })
            portfolio_df.to_csv(backtest_dir / f"portfolio_values_ep{self.current_episode}_{timestamp}.csv", index=False)
        
        self.logger.info(f"Saved backtest results to {backtest_dir / filename}")
    
    def _update_metrics(self, new_metrics: Dict[str, float]):
        """
        Update stored metrics with new values.
        
        Args:
            new_metrics: Dictionary with new metric values
        """
        for metric, value in new_metrics.items():
            if metric in self.metrics:
                self.metrics[metric].append(value)
    
    def _check_early_stopping(self, eval_return: float) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            eval_return: Current evaluation return
        
        Returns:
            True if training should stop, False otherwise
        """
        if eval_return > (self.best_eval_return + self.early_stopping_threshold):
            # Improvement detected
            self.best_eval_return = eval_return
            self.patience_counter = 0
            self.logger.info(f"New best evaluation return: {eval_return:.4f}")
            return False
        else:
            # No significant improvement
            self.patience_counter += 1
            self.logger.info(f"No improvement in evaluation return. Patience: {self.patience_counter}/{self.early_stopping_patience}")
            return self.patience_counter >= self.early_stopping_patience
    
    def _save_model(self, episode: int, is_final: bool = False, is_interrupted: bool = False, is_error: bool = False):
        """
        Save the current model.
        
        Args:
            episode: Current episode number
            is_final: Whether this is the final model save
            is_interrupted: Whether training was interrupted
            is_error: Whether an error occurred during training
        """
        prefix = "final" if is_final else "checkpoint"
        if is_interrupted:
            prefix = "interrupted"
        if is_error:
            prefix = "error"
            
        save_path = self.experiment_dir / f"{prefix}_model_episode_{episode}.pt"
        
        # Get recent metrics for saving
        recent_metrics = {metric: self.metrics[metric][-1] if self.metrics[metric] else None 
                         for metric in self.metrics_to_track if metric in self.metrics}
        
        # Save the model
        torch.save({
            'episode': episode,
            'total_steps': self.total_steps,
            'model_state_dict': self.agent.get_model().state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'metrics': recent_metrics,
            'training_time': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        
        self.logger.info(f"Saved model to {save_path}")
    
    def _save_metrics(self):
        """Save training metrics to disk."""
        # Ensure all lists have the same length by padding shorter lists with None
        max_length = max(len(values) for values in self.metrics.values() if len(values) > 0)
        padded_metrics = {}
        
        for key, values in self.metrics.items():
            if not values:  # Skip empty lists
                continue
            # Pad shorter lists with None values
            padded_metrics[key] = values + [None] * (max_length - len(values))
        
        # Convert to DataFrame for easier analysis
        metrics_df = pd.DataFrame(padded_metrics)
        metrics_df.to_csv(self.experiment_dir / "metrics.csv", index=False)
        
        # Define a conversion function if it doesn't exist already in the class
        if not hasattr(self, 'convert_to_serializable'):
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                    np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, (np.complex64, np.complex128)):
                    return {'real': obj.real, 'imag': obj.imag}
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                elif isinstance(obj, tuple):
                    return [convert_to_serializable(i) for i in obj]
                else:
                    # If it's a non-serializable object, try to convert to string
                    try:
                        return str(obj)
                    except:
                        return None
            self.convert_to_serializable = convert_to_serializable
        
        # Also save as JSON for easier loading
        with open(self.experiment_dir / "metrics.json", 'w') as f:
            # Convert all metrics to JSON serializable format
            json_metrics = {k: self.convert_to_serializable(vals) for k, vals in self.metrics.items()}
            json.dump(json_metrics, f, indent=4)
    
    def _plot_metrics(self):
        """Generate and save plots of training metrics."""
        if not self.metrics["episode"]:
            return  # No metrics to plot yet
            
        # Filter metrics to those that have data and match the episodes length
        episodes = self.metrics["episode"]
        episode_count = len(episodes)
        
        # Find metrics that can be plotted (non-empty and length matches episodes or can be aligned)
        metrics_to_plot = []
        for metric in self.metrics_to_track:
            if metric in self.metrics and len(self.metrics[metric]) > 0:
                if len(self.metrics[metric]) == episode_count:
                    # Length matches exactly - can be plotted directly
                    metrics_to_plot.append(metric)
                elif len(self.metrics[metric]) < episode_count:
                    # This means the metric was only captured for some episodes
                    # We'll handle this by aligning when plotting
                    metrics_to_plot.append(metric)
        
        n_plots = len(metrics_to_plot)
        
        if n_plots == 0:
            return  # No metrics to plot
            
        # Create figure
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]  # Make it indexable if there's only one plot
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            metric_data = self.metrics[metric]
            metric_len = len(metric_data)
            
            if metric_len == episode_count:
                # Direct plot - lengths match
                axes[i].plot(episodes, metric_data)
            else:
                # Metric length doesn't match episodes length
                # We'll plot the metric against the most recent episodes
                # This is often needed for metrics that are only calculated periodically
                recent_episodes = episodes[-metric_len:] if metric_len > 0 else []
                axes[i].plot(recent_episodes, metric_data)
                axes[i].set_title(f"{metric} vs Episode (Partial Data)")
            
            axes[i].set_xlabel("Episode")
            axes[i].set_ylabel(metric)
            if metric_len == episode_count:
                axes[i].set_title(f"{metric} vs Episode")
            axes[i].grid(True)
        
        # Add title and adjust layout
        plt.suptitle(f"Training Metrics for {self.experiment_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        plt.savefig(self.experiment_dir / "metrics_plot.png")
        plt.close(fig)
    
    def _log_episode_results(self, episode: int, metrics: Dict[str, float]):
        """
        Log episode results.
        
        Args:
            episode: Current episode number
            metrics: Dictionary with episode metrics
        """
        self.logger.info(f"Episode {episode + 1} completed")
        self.logger.info(f"  Steps: {metrics['episode_length']}")
        self.logger.info(f"  Reward: {metrics['episode_reward']:.4f}")
        self.logger.info(f"  Portfolio Value: ${metrics['portfolio_value']:.2f}")
        
        if "loss" in metrics:
            self.logger.info(f"  Loss: {metrics['loss']:.6f}")
        if "epsilon" in metrics:
            self.logger.info(f"  Epsilon: {metrics['epsilon']:.4f}")
    
    def _log_eval_results(self, episode: int, metrics: Dict[str, float]):
        """
        Log evaluation results.
        
        Args:
            episode: Current episode number
            metrics: Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluation after episode {episode + 1}")
        self.logger.info(f"  Evaluation Return: {metrics['eval_return']:.4f}")
        self.logger.info(f"  Evaluation Portfolio Value: ${metrics['eval_portfolio_value']:.2f}")
        
        if "sharpe_ratio" in metrics:
            self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        if "max_drawdown" in metrics:
            self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        if "total_return" in metrics:
            self.logger.info(f"  Total Return: {metrics['total_return']:.2%}")
    
    @classmethod
    def load_experiment(cls, experiment_dir: str):
        """
        Load an existing experiment from disk.
        
        Args:
            experiment_dir: Path to the experiment directory
        
        Returns:
            Dictionary with experiment data
        """
        # Configure logging with experiment name
        experiment_name = os.path.basename(experiment_dir)
        Logger.configure(
            EXPERIMENT_NAME=f"load_{experiment_name}",
            EXPERIMENT_DIR=str(Path(experiment_dir))
        )
        logger = Logger.get_logger("load_experiment")
        
        # Load configuration from the config directory
        config_dir = Path(experiment_dir) / "config"
        config_path = config_dir / "experiment_config.json"
        
        # Check for older config path for backward compatibility
        if not config_path.exists():
            old_config_path = Path(experiment_dir) / "config.json"
            if old_config_path.exists():
                config_path = old_config_path
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load metrics
        try:
            metrics_df = pd.read_csv(Path(experiment_dir) / "metrics.csv")
            metrics = {col: metrics_df[col].tolist() for col in metrics_df.columns}
        except FileNotFoundError:
            # Try JSON instead
            try:
                with open(Path(experiment_dir) / "metrics.json", 'r') as f:
                    metrics = json.load(f)
            except FileNotFoundError:
                metrics = {}
        
        # Load data information
        data_info = {}
        data_dir = Path(experiment_dir) / "data"
        if data_dir.exists():
            for file in data_dir.glob("*.json"):
                with open(file, 'r') as f:
                    data_info[file.stem] = json.load(f)
        
        # Load latest model checkpoint
        model_files = list(Path(experiment_dir).glob("*model_episode_*.pt"))
        latest_model = None
        if model_files:
            latest_model = max(model_files, key=lambda x: int(str(x).split("_episode_")[1].split(".")[0]))
        
        experiment_data = {
            "config": config,
            "metrics": metrics,
            "latest_model_path": latest_model,
            "data_info": data_info
        }
        
        return experiment_data

    @classmethod
    def continue_experiment(cls, experiment_dir: str, additional_episodes: int, max_train_time: Optional[int] = None):
        """
        Continue a previously saved experiment from a checkpoint.
        
        Args:
            experiment_dir: Directory containing the experiment
            additional_episodes: Number of additional episodes to train
            max_train_time: Maximum training time in seconds (if None, keeps original)
            
        Returns:
            ExperimentManager instance initialized from checkpoint
        """
        from data.data_manager import DataManager
        from environments.trading_env import TradingEnv
        from models.agents.dqn_agent import DQNAgent
        from models.agents.directional_dqn_agent import DirectionalDQNAgent
        from models.agents.ppo_agent import PPOAgent
        from models.agents.a2c_agent import A2CAgent
        from models.processors.trading_processor import TradingObservationProcessor
        from models.backtesting import Backtester
        import pandas as pd
        
        experiment_path = Path(experiment_dir)
        
        # Set experiment name for logging
        experiment_name = os.path.basename(experiment_dir)
        Logger.configure(
            EXPERIMENT_NAME=f"continue_{experiment_name}",
            EXPERIMENT_DIR=str(experiment_path)
        )
        exp_logger = Logger.get_logger("continue_experiment")
        
        exp_logger.info(f"Continuing experiment from: {experiment_dir}")
        
        # Load configuration
        config_dir = experiment_path / "config"
        config_path = config_dir / "experiment_config.json"
        
        if not os.path.exists(config_path):
            # Fall back to old path for backward compatibility
            config_path = experiment_path / "config.json"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Find the latest model checkpoint
        model_files = list(experiment_path.glob("*model_episode_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model checkpoints found in {experiment_path}")
            
        latest_model = max(model_files, key=lambda x: int(str(x).split("_episode_")[1].split(".")[0]))
        latest_episode = int(str(latest_model).split("_episode_")[1].split(".")[0])
        
        exp_logger.info(f"Found latest checkpoint at episode {latest_episode}: {latest_model}")
        
        # Load the checkpoint
        checkpoint = torch.load(latest_model)
        
        # Load metrics if available
        metrics_path = experiment_path / "metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            exp_logger.info(f"Loaded existing metrics with {len(metrics_df)} entries")
        else:
            exp_logger.warning(f"No metrics file found at {metrics_path}")
            metrics_df = None
            
        # Try to load the experiment manager directly
        experiment_manager_path = experiment_path / "experiment_manager.pkl"
        if experiment_manager_path.exists():
            try:
                import pickle
                with open(experiment_manager_path, 'rb') as f:
                    experiment_manager = pickle.load(f)
                
                # Update training parameters
                if max_train_time is not None:
                    experiment_manager.max_train_time = max_train_time
                
                exp_logger.info(f"Successfully loaded experiment manager from {experiment_manager_path}")
                return experiment_manager
            except Exception as e:
                exp_logger.warning(f"Failed to load experiment manager from pickle: {e}")
                exp_logger.info("Will reconstruct experiment from saved files")
        
        # Load data information
        data_info = {}
        data_dir = experiment_path / "data"
        
        # Look for environment and agent configurations in the config directory
        env_config_path = config_dir / "environment_config.json"
        agent_config_path = config_dir / "agent_config.json"
        
        # Check for older file paths for backward compatibility
        if not env_config_path.exists():
            old_env_config_path = experiment_path / "env_config.json"
            if old_env_config_path.exists():
                env_config_path = old_env_config_path
                exp_logger.warning(f"Using legacy environment config path: {env_config_path}")
            
        if not agent_config_path.exists():
            old_agent_config_path = experiment_path / "agent_config.json"
            if old_agent_config_path.exists():
                agent_config_path = old_agent_config_path
                exp_logger.warning(f"Using legacy agent config path: {agent_config_path}")
        
        if not data_dir.exists() or not env_config_path.exists() or not agent_config_path.exists():
            raise FileNotFoundError(f"Required data or config files not found in {experiment_path}")
        
        # Load environment and agent configurations
        with open(env_config_path, 'r') as f:
            env_config = json.load(f)
            
        with open(agent_config_path, 'r') as f:
            agent_config = json.load(f)
            
        # Load data samples to determine structure
        train_sample_path = data_dir / "train_data_sample.csv"
        val_sample_path = data_dir / "val_data_sample.csv"
        
        if not train_sample_path.exists() or not val_sample_path.exists():
            raise FileNotFoundError(f"Data samples not found in {data_dir}")
            
        # Load data info to reconstruct full datasets
        data_info_path = data_dir / "data_info.json"
        if data_info_path.exists():
            with open(data_info_path, 'r') as f:
                data_info = json.load(f)
                
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
        
        exp_logger.info("Loading training and validation data...")
        
        # Get date ranges and tickers from data info
        train_info = data_info.get("train_data", {})
        val_info = data_info.get("val_data", {})
        
        # Extract parameters from data info
        tickers = train_info.get("tickers", [])
        train_date_range = train_info.get("date_range", ["", ""])
        val_date_range = val_info.get("date_range", ["", ""])
        
        if not tickers or not train_date_range[0] or not train_date_range[1]:
            raise ValueError("Missing required data information (tickers or date ranges)")
            
        # Load the full datasets
        train_data = data_manager.prepare_data(
            tickers=tickers,
            start_date=train_date_range[0],
            end_date=train_date_range[1],
            normalize=True,
            add_day_index_bool=True,
        )
        
        val_data = data_manager.prepare_data(
            tickers=tickers,
            start_date=val_date_range[0] if val_date_range[0] else train_date_range[1],
            end_date=val_date_range[1] if val_date_range[1] else None,
            normalize=True,
            add_day_index_bool=True,
        )
        
        # Get raw data as well
        train_raw_data = data_manager.download_data(
            tickers=tickers,
            start_date=train_date_range[0],
            end_date=train_date_range[1],
            add_day_index_bool=True,
        )
        
        val_raw_data = data_manager.download_data(
            tickers=tickers,
            start_date=val_date_range[0] if val_date_range[0] else train_date_range[1],
            end_date=val_date_range[1] if val_date_range[1] else None,
            add_day_index_bool=True,
        )
        
        # Extract columns config from env_config or infer from data
        train_env_params = config.get("train_env_params", {})
        columns = {
            "ticker": "ticker",
            "price": "close",
            "day": "day",
            "ohlcv": ["open", "high", "low", "close", "volume"],
            "tech_cols": [col for col in train_data.columns if col not in ["ticker", "day", "open", "high", "low", "close", "volume", "date", "timestamp"]]
        }
        
        # Create environments
        exp_logger.info("Recreating training and validation environments...")
        train_env = TradingEnv(
            processed_data=train_data,
            raw_data=train_raw_data,
            columns=columns,
            env_params=env_config.get("env_params", {}),
            friction_params=env_config.get("friction_params", {}),
            constraint_params=env_config.get("constraint_params", {}),
            reward_params=env_config.get("reward_params", {}),
            render_mode=None,
        )
        
        val_env = TradingEnv(
            processed_data=val_data,
            raw_data=val_raw_data,
            columns=columns,
            env_params=env_config.get("env_params", {}),
            friction_params=env_config.get("friction_params", {}),
            constraint_params=env_config.get("constraint_params", {}),
            reward_params=env_config.get("reward_params", {}),
            render_mode=None,
        )
        
        # Create agent based on type
        exp_logger.info("Recreating agent and loading model weights...")
        agent_type = config.get("agent_type", "DirectionalDQNAgent")
        
        if agent_type == "DirectionalDQNAgent":
            agent = DirectionalDQNAgent(
                env=train_env,
                observation_processor_class=TradingObservationProcessor,
                learning_rate=agent_config.get("learning_rate", 0.001),
                gamma=agent_config.get("gamma", 0.99),
                epsilon_start=agent_config.get("epsilon", 0.01),  # Continue with low exploration
                epsilon_end=agent_config.get("epsilon_end", 0.01),
                epsilon_decay=agent_config.get("epsilon_decay", 0.999),
                target_update=agent_config.get("target_update", 10),
                memory_size=agent_config.get("memory_size", 10000),
                batch_size=agent_config.get("batch_size", 256),
            )
        elif agent_type == "DQNAgent":
            agent = DQNAgent(
                env=train_env,
                observation_processor_class=TradingObservationProcessor,
                learning_rate=agent_config.get("learning_rate", 0.001),
                gamma=agent_config.get("gamma", 0.99),
                epsilon_start=agent_config.get("epsilon", 0.01),  # Continue with low exploration
                epsilon_end=agent_config.get("epsilon_end", 0.01),
                epsilon_decay=agent_config.get("epsilon_decay", 0.999),
                target_update=agent_config.get("target_update", 10),
                memory_size=agent_config.get("memory_size", 10000),
                batch_size=agent_config.get("batch_size", 256),
            )
        elif agent_type == "PPOAgent":
            agent = PPOAgent(
                env=train_env,
                observation_processor_class=TradingObservationProcessor,
                learning_rate=agent_config.get("learning_rate", 0.0003),
                gamma=agent_config.get("gamma", 0.99),
                gae_lambda=agent_config.get("gae_lambda", 0.95),
                clip_param=agent_config.get("clip_param", 0.2),
                value_loss_coef=agent_config.get("value_loss_coef", 0.5),
                entropy_coef=agent_config.get("entropy_coef", 0.01),
                max_grad_norm=agent_config.get("max_grad_norm", 0.5),
                batch_size=agent_config.get("batch_size", 64),
                epochs=agent_config.get("epochs", 10),
            )
        elif agent_type == "A2CAgent":
            agent = A2CAgent(
                env=train_env,
                observation_processor_class=TradingObservationProcessor, 
                learning_rate=agent_config.get("learning_rate", 0.0007),
                gamma=agent_config.get("gamma", 0.99),
                value_loss_coef=agent_config.get("value_loss_coef", 0.5),
                entropy_coef=agent_config.get("entropy_coef", 0.01),
                max_grad_norm=agent_config.get("max_grad_norm", 0.5),
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        # Load model weights
        agent.get_model().load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Initialize backtester
        backtester = Backtester(
            env=val_env,
            agent=agent,
            asset_names=tickers,
            visualizer=None,
            save_visualizations=True,
            visualization_dir=os.path.join(experiment_dir, "backtest_visualizations")
        )
        
        # Create a new experiment manager
        exp_logger.info("Creating new experiment manager with loaded components...")
        experiment_manager = cls(
            experiment_name=os.path.basename(experiment_dir),
            train_env=train_env,
            val_env=val_env,
            agent=agent,
            backtester=backtester,
            max_train_time=max_train_time if max_train_time is not None else config.get("max_train_time", 86400),
            eval_interval=config.get("eval_interval", 10),
            save_interval=config.get("save_interval", 100),
            n_eval_episodes=config.get("n_eval_episodes", 5),
            early_stopping_patience=config.get("early_stopping_patience", 20),
            early_stopping_threshold=config.get("early_stopping_threshold", 0.01),
            metrics_to_track=config.get("metrics_to_track", []),
            render_train=config.get("render_train", False),
            render_eval=config.get("render_eval", False),
            base_dir=os.path.dirname(experiment_dir),
            save_data_visualization=False,  # Skip initial visualization on continuation
            save_data_snapshots=False,      # Skip data snapshots on continuation
        )
        
        # Set the current episode to continue from where we left off
        experiment_manager.current_episode = latest_episode
        
        # Transfer metrics if available
        if metrics_df is not None:
            for column in metrics_df.columns:
                if column in experiment_manager.metrics:
                    experiment_manager.metrics[column] = metrics_df[column].tolist()
                    
        # Note: resuming from the exact state would require more memory state
        # but this gives a reasonable continuation point
                    
        exp_logger.info(f"Successfully reconstructed experiment from checkpoint at episode {latest_episode}")
        return experiment_manager 