"""Enhanced experiment manager that integrates metrics, checkpoint, and visualization management"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

from data.data_manager import DataManager
from models.agents.base_agent import BaseAgent
from environments.trading_env import TradingEnv
from managers.backtest_manager import BacktestManager
from utils.logger import Logger
from models.agents.agent_factory import AgentFactory

from managers.metrics_manager import MetricsManager, MetricCategory
from managers.checkpoint_manager import CheckpointManager
from managers.visualization_manager import VisualizationManager
from managers.config_manager import ConfigManager
from managers.data_manager import ExperimentDataManager

class ExperimentManager:
    """
    Enhanced experiment manager that integrates metrics, checkpoint, and visualization management.
    Provides a unified interface for running and managing trading experiments.
    """
    
    def __init__(
        self,
        experiment_name: str,
        train_env: TradingEnv,
        val_env: TradingEnv,
        agent: BaseAgent,
        max_train_time: Optional[int] = None,  # Max training time in seconds (None for unlimited)
        eval_interval: int = 10,  # Episodes between evaluations
        save_interval: int = 50,  # Episodes between model saves
        save_metric_interval: int = 2,  # Episodes between metric saves
        n_eval_episodes: int = 5,  # Number of episodes for evaluation
        early_stopping_patience: int = 100,  # Episodes with no improvement before stopping
        early_stopping_threshold: float = 0.01,  # Minimum improvement to reset patience
        early_stopping_metric: str = "sharpe_ratio",  # Metric to use for early stopping
        render_train: bool = False,  # Whether to render training environment
        render_eval: bool = False,  # Whether to render evaluation environment
        base_dir: str = "experiments",  # Base directory for saving experiment data
        is_setup: bool = False,  # Whether this is a new experiment setup
    ):
        """
        Initialize the experiment manager.
        
        Args:
            experiment_name: Name of the experiment
            train_env: Training environment
            val_env: Validation environment
            agent: Agent to train
            max_train_time: Maximum training time in seconds (None for unlimited)
            eval_interval: Number of episodes between evaluations
            save_interval: Number of episodes between model saves
            save_metric_interval: Number of episodes between metric saves
            n_eval_episodes: Number of episodes for evaluation
            early_stopping_patience: Episodes with no improvement before stopping
            early_stopping_threshold: Minimum improvement to reset patience
            early_stopping_metric: Metric to use for early stopping (e.g., "sharpe_ratio", "total_return")
            render_train: Whether to render training environment
            render_eval: Whether to render evaluation environment
            base_dir: Base directory for saving experiment data
            is_setup: Whether this is a new experiment setup
        """
        self.experiment_name = experiment_name
        self.train_env = train_env
        self.val_env = val_env
        self.agent = agent
        self.max_train_time = max_train_time
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.save_metric_interval = save_metric_interval
        self.n_eval_episodes = n_eval_episodes
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_metric = early_stopping_metric
        self.render_train = render_train
        self.render_eval = render_eval
        self.is_setup = is_setup  # Store the setup flag
        
        # Create experiment directory
        self.experiment_dir = Path(base_dir) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        Logger.configure(
            EXPERIMENT_NAME=experiment_name,
            EXPERIMENT_DIR=str(self.experiment_dir)
        )
        self.logger = Logger.get_logger(name=f"experiment_{experiment_name}")
        
        # Initialize managers
        self._init_managers()
        
        # Initialize training state
        self.current_episode = 0
        self.total_steps = 0
        self.start_time = None
        self.is_finished = False
        
        # Initialize early stopping tracking
        self.best_metric_value = float('-inf')
        self.patience_counter = 0
        
        # Save initial setup only if this is a new experiment
        if is_setup:
            self._save_initial_config()
        
        self.logger.info(f"Initialized experiment: {experiment_name}")
    
    def _init_managers(self):
        """Initialize all managers used by the experiment."""
        # Initialize config manager
        self.config_manager = ConfigManager(self.experiment_dir, self.logger)
        
        # Initialize data manager
        self.data_manager = ExperimentDataManager(self.experiment_dir, self.logger)
        
        # Initialize metrics manager
        self.metrics_manager = MetricsManager(
            experiment_dir=self.experiment_dir,
            logger=self.logger,
        )
        
        # Register agent-specific metrics
        agent_type = self.agent.get_model_name().lower()
        agent_info = self.agent.get_info()
        self.metrics_manager.register_agent_metrics(agent_type, list(agent_info.keys()))
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            experiment_dir=str(self.experiment_dir),
            agent=self.agent,
            max_checkpoints=5,
            save_best_only=False,
            best_metric="eval_return",
            logger=self.logger
        )
        
        # Initialize visualization manager
        self.visualizer = VisualizationManager()
        
        # Initialize backtester
        backtest_dir = self.experiment_dir / "backtest"
        backtest_dir.mkdir(exist_ok=True)
        self.backtester = BacktestManager(
            train_env=self.train_env,
            val_env=self.val_env,
            agent=self.agent,
            save_dir=str(backtest_dir),
            save_visualizations=True,
            asset_names=self.train_env.processed_data["ticker"].unique().tolist()
        )
    
    def _save_initial_config(self):
        """Save initial experiment configuration."""
        # Save experiment config
        self.config_manager.save_experiment_config(
            experiment_name=self.experiment_name,
            max_train_time=self.max_train_time,
            eval_interval=self.eval_interval,
            save_interval=self.save_interval,
            save_metric_interval=self.save_metric_interval,
            n_eval_episodes=self.n_eval_episodes,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_threshold=self.early_stopping_threshold,
            early_stopping_metric=self.early_stopping_metric,
            render_train=self.render_train,
            render_eval=self.render_eval,
            agent_type=self.agent.get_model_name(),
            train_env_params={
                "n_assets": self.train_env.n_assets,
                "initial_balance": self.train_env.initial_balance,
                "window_size": self.train_env.window_size
            }
        )
        
        # Save environment config
        self.config_manager.save_environment_config(
            env_params=getattr(self.train_env, "env_params", {}),
            friction_params=getattr(self.train_env, "friction_params", {}),
            constraint_params=getattr(self.train_env, "constraint_params", {}),
            reward_params=getattr(self.train_env, "reward_params", {}),
            processor_configs=getattr(self.train_env, "processor_configs", {})
        )
        
        # Get agent config and add interpreter info
        agent_config = self.agent.get_config()
        agent_config["interpreter_type"] = self.agent.interpreter.get_type()
        agent_config["interpreter_config"] = self.agent.interpreter.get_config()
        
        # Save agent config
        self.config_manager.save_agent_config(agent_config)
        
        # Save data config
        self.config_manager.save_data_config(
            normalization_params=getattr(self.train_env, "normalization_params", {}),
            processor_params={
                "vix_params": getattr(self.train_env, "vix_params", {}),
                "turbulence_params": getattr(self.train_env, "turbulence_params", {}),
                "indicator_params": getattr(self.train_env, "indicator_params", {})
            }
        )
        
        # Save experiment data using data manager
        self.data_manager.save_data(
            train_data=self.train_env.raw_data,
            val_data=self.val_env.raw_data,
            data_type="raw"
        )
        
        self.data_manager.save_data(
            train_data=self.train_env.processed_data,
            val_data=self.val_env.processed_data,
            data_type="normalized"
        )
        
        self.data_manager.save_data_info(
            train_data=self.train_env.processed_data,
            val_data=self.val_env.processed_data,
            columns_mapping=self.train_env.columns,
            processor_params={
                "vix_params": getattr(self.train_env, "vix_params", {}),
                "turbulence_params": getattr(self.train_env, "turbulence_params", {}),
                "indicator_params": getattr(self.train_env, "indicator_params", {})
            },
            normalization_params=getattr(self.train_env, "normalization_params", {})
        )

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
                
                # Update metrics - only training metrics
                self.metrics_manager.update(episode_metrics, agent_type=self.agent.get_model_name().lower())
                
                # Log episode results
                self.metrics_manager.log_episode_results(episode, episode_metrics)
                
                # Evaluate if needed
                if (episode + 1) % self.eval_interval == 0:
                    eval_metrics = self._run_evaluation()
                    self._check_early_stopping(eval_metrics)
                
                # Save model if needed
                if (episode + 1) % self.save_interval == 0:
                    self.checkpoint_manager.save_checkpoint(
                        episode=episode,
                        metrics=self.metrics_manager.get_latest_metrics()
                    )
                
                # Save metrics if needed
                if (episode + 1) % self.save_metric_interval == 0:
                    self.metrics_manager.save()
                    self.metrics_manager.plot()
                
                # Check if we should stop early
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {episode + 1} episodes due to lack of improvement.")
                    break
            
            # Training finished
            self.is_finished = True
            self.logger.info(f"Training completed after {self.current_episode + 1} episodes")
            self.logger.info(f"Total training time: {timedelta(seconds=time.time() - self.start_time)}")
            
            # Final evaluation and save
            eval_metrics = self._run_evaluation()
            self.checkpoint_manager.save_checkpoint(
                episode=self.current_episode,
                metrics=self.metrics_manager.get_latest_metrics(),
                force_save=True
            )
            self.metrics_manager.save()
            self.metrics_manager.plot()
            
            return self.metrics_manager.get_latest_metrics()
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user.")
            self.checkpoint_manager.save_checkpoint(
                episode=self.current_episode,
                metrics=self.metrics_manager.get_latest_metrics(),
                force_save=True
            )
            self.metrics_manager.save()
            self.metrics_manager.plot()
            return self.metrics_manager.get_latest_metrics()
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.checkpoint_manager.save_checkpoint(
                episode=self.current_episode,
                metrics=self.metrics_manager.get_latest_metrics(),
                force_save=True
            )
            self.metrics_manager.save()
            self.metrics_manager.plot()
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
        episode_metrics = {}
        done = False
        
        # Track portfolio values and trades for performance metrics
        portfolio_values = [self.train_env.initial_balance]
        
        # Get agent type to handle different update patterns
        agent_type = self.agent.get_model_name().lower()
        
        # Run episode
        while not done:
            # Select action
            intended_action, action_choice = self.agent.get_intended_action(
                observation, 
                current_position=self.train_env.get_current_position(),
                deterministic=False  # Allow exploration during training
            )
            
            # Take action in environment
            next_observation, reward, done, info = self.train_env.step(intended_action)
            
            # Track portfolio value
            portfolio_values.append(info["portfolio_value"])
            
            # Handle different agent types differently
            if agent_type == "a2c":
                # A2C uses a different update pattern (on-policy)
                self.agent.add_to_rollout(
                    observation=observation,
                    action=intended_action,
                    action_choice=action_choice,
                    reward=reward,
                    next_observation=next_observation,
                    done=done
                )
            elif agent_type == "ddpg":
                # DDPG uses experience replay
                self.agent.add_to_memory(
                    observation=observation,
                    action=intended_action,
                    action_choice=action_choice,
                    reward=reward,
                    next_observation=next_observation,
                    done=done
                )
                
                # Update DDPG if enough samples
                if len(self.agent.memory) >= self.agent.batch_size:
                    update_metrics = self.agent.update()
                    episode_loss.append(update_metrics.get("actor_loss", 0))
                    # Store all metrics for later use
                    for k, v in update_metrics.items():
                        episode_metrics[k] = episode_metrics.get(k, []) + [v]
            else:
                # DQN-style agents use experience replay
                self.agent.add_to_memory(
                    observation=observation,
                    action=intended_action,
                    action_choice=action_choice,
                    reward=reward,
                    next_observation=next_observation,
                    done=done
                )

                # Update DQN-style agents if enough samples
                if hasattr(self.agent, 'memory') and hasattr(self.agent, 'batch_size') and len(self.agent.memory) >= self.agent.batch_size:
                    batch = self.agent.get_batch()
                    update_metrics = self.agent.update(batch)
                    episode_loss.append(update_metrics.get("loss", 0))
                    # Store all metrics for later use
                    for k, v in update_metrics.items():
                        episode_metrics[k] = episode_metrics.get(k, []) + [v]
            
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
        
        # For A2C, update at the end of episode
        if agent_type == "a2c":
            update_metrics = self.agent.update()
            for k, v in update_metrics.items():
                episode_metrics[k] = [v]  # Store as single-item list for consistency
        
        # Calculate performance metrics
        portfolio_values = np.array(portfolio_values)
        
        # Calculate returns as percentage changes in portfolio value
        returns = np.zeros_like(portfolio_values[1:], dtype=float)
        for i in range(1, len(portfolio_values)):
            returns[i-1] = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0  # Annualized volatility
        
        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = np.mean(returns) * np.sqrt(252) / volatility if volatility > 0 else 0
        
        # Sortino Ratio (using negative returns only)
        negative_returns = returns[returns < 0]
        sortino_ratio = np.mean(returns) * np.sqrt(252) / np.std(negative_returns) if len(negative_returns) > 0 and np.std(negative_returns) > 0 else 0
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Calmar Ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Compile episode metrics
        metrics = {
            "episode": episode,
            "episode_length": episode_steps,
            "episode_reward": episode_reward,
            "portfolio_value": portfolio_values[-1],
            "timestamp": time.time(),
            
            # Performance metrics
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "volatility": volatility,
        }
        
        # Add reward components from info
        if "reward_components" in info:
            for reward_type, value in info["reward_components"].items():
                metrics[f"reward_{reward_type}"] = value
        
        # Add metrics from update if available
        for metric_name, values in episode_metrics.items():
            if values:
                metrics[metric_name] = np.mean(values)
        
        # Get additional info from the agent
        agent_info = self.agent.get_info()
        metrics.update(agent_info)
        
        return metrics
    
    def _run_evaluation(self) -> Dict[str, float]:
        """
        Run evaluation using backtesting on the validation environment.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Running evaluation after episode {self.current_episode}")
        
        # Set agent to evaluation mode
        self.agent.eval()
        
        # Run backtest if backtester is available
        if self.backtester:
            # Make sure backtester save directory is set correctly
            backtest_dir = self.experiment_dir / "backtest"
            backtest_dir.mkdir(exist_ok=True)
            self.backtester.save_dir = str(backtest_dir)
            self.backtester.save_visualizations = True
            
            # Run backtest
            backtest_results = self.backtester.run_full_backtest(
                model_state_dict=self.agent.get_state_dict(),
                deterministic=True,
                episode_id=self.current_episode,
                include_train=True,  # Only run on validation data during evaluation
                include_val=True,
                save_visualization=True
            )
            
            # Extract validation results
            if "validation" in backtest_results and "metrics" in backtest_results["validation"]:
                metrics = backtest_results["validation"]["metrics"]
                self.logger.info(f"Evaluation metrics: {metrics}")
            else:
                self.logger.warning(f"Backtest results structure: {backtest_results.keys()}")
                metrics = {}
                self.logger.warning("No validation metrics found in backtest results")
        else:
            metrics = {}
            self.logger.warning("No backtester available for evaluation")
        
        # Set agent back to training mode
        self.agent.train()
        
        return metrics
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.early_stopping_metric not in metrics:
            self.logger.warning(f"Early stopping metric '{self.early_stopping_metric}' not found in metrics. Skipping early stopping check.")
            return False
            
        current_metric = metrics[self.early_stopping_metric]
        
        if current_metric > (self.best_metric_value + self.early_stopping_threshold):
            # Improvement detected
            self.best_metric_value = current_metric
            self.patience_counter = 0
            self.logger.info(f"New best {self.early_stopping_metric}: {current_metric:.4f}")
            return False
        else:
            # No significant improvement
            self.patience_counter += 1
            self.logger.info(f"No improvement in {self.early_stopping_metric}. Patience: {self.patience_counter}/{self.early_stopping_patience}")
            return self.patience_counter >= self.early_stopping_patience
    
    def save_experiment_data(
        self,
        raw_train_data: pd.DataFrame,
        raw_val_data: pd.DataFrame,
        processed_train_data: pd.DataFrame,
        processed_val_data: pd.DataFrame,
        normalized_train_data: pd.DataFrame,
        normalized_val_data: pd.DataFrame,
        columns: Dict,
        processor_params: Dict,
        normalization_params: Dict
    ) -> None:
        """
        Save all experiment data using the internal data manager.
        This method should only be called during experiment setup.
        
        Args:
            raw_train_data: Raw training data
            raw_val_data: Raw validation data
            processed_train_data: Processed training data
            processed_val_data: Processed validation data
            normalized_train_data: Normalized training data
            normalized_val_data: Normalized validation data
            columns: Column mapping dictionary
            processor_params: Parameters used for data processing
            normalization_params: Parameters used for normalization
        """
        if not self.is_setup:
            self.logger.warning("save_experiment_data should only be called during experiment setup")
            return
            
        # Save raw data
        self.data_manager.save_data(
            train_data=raw_train_data,
            val_data=raw_val_data,
            data_type="raw"
        )
        
        # Save normalized data
        self.data_manager.save_data(
            train_data=normalized_train_data,
            val_data=normalized_val_data,
            data_type="normalized"
        )
        
        # Save comprehensive data info
        self.data_manager.save_data_info(
            train_data=processed_train_data,
            val_data=processed_val_data,
            columns_mapping=columns,
            processor_params=processor_params,
            normalization_params=normalization_params
        )
        
        self.logger.info(f"Saved all experiment data to {self.data_manager.data_dir}")

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
        
        # Load data info
        data_dir = Path(experiment_dir) / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
        with open(data_dir / "data_info.json", 'r') as f:
            data_info = json.load(f)
        
        # Load data
        train_data = pd.read_csv(data_dir / "normalized/train_data.csv")
        val_data = pd.read_csv(data_dir / "normalized/val_data.csv")
        raw_train_data = pd.read_csv(data_dir / "raw/train_data.csv")
        raw_val_data = pd.read_csv(data_dir / "raw/val_data.csv")
        
        # Initialize config manager
        config_manager = ConfigManager(Path(experiment_dir), logger)
        
        # Load all configurations
        configs = config_manager.load_all_configs()
        
        # Load metrics
        metrics_manager = MetricsManager(Path(experiment_dir), logger)
        
        # Load latest model checkpoint
        checkpoint_manager = CheckpointManager(
            experiment_dir,
            None,  # Agent will be set later
            logger=logger
        )
        latest_model = checkpoint_manager.get_latest_checkpoint()
        
        experiment_data = {
            "config": configs,
            "metrics": metrics_manager.get_latest_metrics(),
            "latest_model_path": latest_model,
            "data_info": data_info,
            "raw_train_data": raw_train_data,
            "raw_val_data": raw_val_data,
            "train_data": train_data,
            "val_data": val_data
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
        experiment_path = Path(experiment_dir)
        
        # Set experiment name for logging
        experiment_name = os.path.basename(experiment_dir)
        Logger.configure(
            EXPERIMENT_NAME=f"continue_{experiment_name}",
            EXPERIMENT_DIR=str(experiment_path)
        )
        exp_logger = Logger.get_logger("continue_experiment")
        
        exp_logger.info(f"Continuing experiment from: {experiment_dir}")
        
        # Initialize config manager and load configurations
        config_manager = ConfigManager(experiment_path, exp_logger)
        configs = config_manager.load_all_configs()
        
        # Get experiment config
        experiment_config = configs["experiment"]
        
        # Find the latest model checkpoint
        checkpoint_manager = CheckpointManager(
            str(experiment_path),
            None,  # Agent will be set later
            logger=exp_logger
        )
        latest_model = checkpoint_manager.get_latest_checkpoint()
        if not latest_model:
            raise FileNotFoundError(f"No model checkpoints found in {experiment_path}")
            
        latest_episode = int(str(latest_model).split("_episode_")[1].split(".")[0])
        
        exp_logger.info(f"Found latest checkpoint at episode {latest_episode}: {latest_model}")
        
        # Load the checkpoint
        checkpoint = torch.load(latest_model)
        
        # Load metrics if available
        metrics_manager = MetricsManager(experiment_path, exp_logger)
        
        # Initialize data manager
        data_manager = ExperimentDataManager(experiment_path, exp_logger)
        
        # Verify data integrity
        if not data_manager.verify_data_integrity():
            raise FileNotFoundError("Data integrity check failed. Required data files are missing or inconsistent.")
        
        # Load data info
        data_info = data_manager.get_data_info()
        
        # Load raw and normalized data
        exp_logger.info("Loading training and validation data...")
        train_raw_data, val_raw_data, raw_metadata = data_manager.load_data(data_type="raw")
        train_data, val_data, normalized_metadata = data_manager.load_data(data_type="normalized")
        
        # Extract columns config from data info
        columns = data_info.get("columns_mapping", {
            "ticker": "ticker",
            "price": "close",
            "day": "day",
            "ohlcv": ["open", "high", "low", "close", "volume"],
            "tech_cols": [col for col in train_data.columns if col not in ["ticker", "day", "open", "high", "low", "close", "volume", "date", "timestamp"]]
        })
        
        # Create environments
        exp_logger.info("Recreating training and validation environments...")
        train_env = TradingEnv(
            processed_data=train_data,
            raw_data=train_raw_data,
            columns=columns,
            env_params=configs["environment"].get("env_params", {}),
            friction_params=configs["environment"].get("friction_params", {}),
            constraint_params=configs["environment"].get("constraint_params", {}),
            reward_params=configs["environment"].get("reward_params", {}),
            processor_configs=configs["environment"].get("processor_configs", {}),
            render_mode=None,
        )
        
        val_env = TradingEnv(
            processed_data=val_data,
            raw_data=val_raw_data,
            columns=columns,
            env_params=configs["environment"].get("env_params", {}),
            friction_params=configs["environment"].get("friction_params", {}),
            constraint_params=configs["environment"].get("constraint_params", {}),
            reward_params=configs["environment"].get("reward_params", {}),
            processor_configs=configs["environment"].get("processor_configs", {}),
            render_mode=None,
        )
        
        # Create agent based on type
        exp_logger.info(f"Recreating agent and loading model weights...")
        agent_type = experiment_config.get("agent_type", "dqn").lower()  # Convert to lowercase for factory
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create agent using factory
        agent = AgentFactory.create_agent(
            agent_type=agent_type,
            env=train_env,
            network_config=configs["agent"].get("network_config", {}),
            interpreter_type=configs["agent"].get("interpreter_type", "discrete"),
            interpreter_config=configs["agent"].get("interpreter_config", {}),
            device=device,
            **configs["agent"]
        )
        
        # Load model weights
        agent.load_state_dict(checkpoint["agent_state"])
        
        # Create a new experiment manager
        exp_logger.info("Creating new experiment manager with loaded components...")
        experiment_manager = cls(
            experiment_name=os.path.basename(experiment_dir),
            train_env=train_env,
            val_env=val_env,
            agent=agent,
            max_train_time=max_train_time if max_train_time is not None else experiment_config.get("max_train_time", 86400),
            eval_interval=experiment_config.get("eval_interval", 10),
            save_interval=experiment_config.get("save_interval", 100),
            save_metric_interval=experiment_config.get("save_metric_interval", 1),
            n_eval_episodes=experiment_config.get("n_eval_episodes", 5),
            early_stopping_patience=experiment_config.get("early_stopping_patience", 20),
            early_stopping_threshold=experiment_config.get("early_stopping_threshold", 0.01),
            early_stopping_metric=experiment_config.get("early_stopping_metric", "sharpe_ratio"),
            render_train=experiment_config.get("render_train", False),
            render_eval=experiment_config.get("render_eval", False),
            base_dir=os.path.dirname(experiment_dir),
            is_setup=True,
        )
        
        # Set the current episode to continue from where we left off
        experiment_manager.current_episode = latest_episode
        
        # Transfer metrics if available
        metrics = metrics_manager.get_latest_metrics()
        for metric, value in metrics.items():
            if metric in experiment_manager.metrics_manager.metrics:
                experiment_manager.metrics_manager.metrics[metric].append(value)
                    
        exp_logger.info(f"Successfully reconstructed experiment from checkpoint at episode {latest_episode}")
        return experiment_manager 