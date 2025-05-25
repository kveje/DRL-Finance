import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum, auto

from utils.logger import Logger

class MetricCategory(Enum):
    """Categories for different types of metrics."""
    TRAINING = auto()      # Basic training metrics (episode reward, length, etc.)
    EVALUATION = auto()    # Evaluation metrics (eval return, portfolio value)
    PERFORMANCE = auto()   # Performance metrics (sharpe ratio, drawdown)
    AGENT = auto()         # Agent-specific metrics (loss, epsilon, etc.)
    REWARD = auto()        # Reward component metrics
    CUSTOM = auto()        # Custom metrics added by the user

class MetricsManager:
    """
    Manages metrics collection, storage, visualization, and analysis for experiments.
    Tracks training metrics including performance metrics calculated from training data.
    """
    
    # Base metrics that are always tracked
    BASE_METRICS = {
        MetricCategory.TRAINING: {
            "episode": [],
            "episode_length": [],
            "episode_reward": [],
            "portfolio_value": [],
            "timestamp": []
        },
        MetricCategory.PERFORMANCE: {
            "sharpe_ratio": [],
            "max_drawdown": [],
            "total_return": [],
            "sortino_ratio": [],
            "calmar_ratio": [],
            "volatility": [],
        },
        MetricCategory.REWARD: {
            "total_reward": []
        },
        MetricCategory.AGENT: {}
    }
    
    def __init__(
        self,
        experiment_dir: Path,
        logger: Logger,
    ):
        """
        Initialize the metrics manager.
        
        Args:
            experiment_dir: Directory to save metrics
            logger: Logger instance for logging
        """
        self.experiment_dir = experiment_dir
        self.logger = logger
        
        # Initialize metrics storage with base metrics
        self.metrics = self.BASE_METRICS.copy()
        
        # Create metrics directory
        self.metrics_dir = experiment_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Track which metrics are being used
        self.active_metrics: Set[str] = set()
    
    def register_agent_metrics(self, agent_type: str, metrics: List[str]) -> None:
        """
        Register metrics specific to an agent type.
        
        Args:
            agent_type: Type of agent (e.g., 'dqn', 'a2c', 'ddpg')
            metrics: List of metric names specific to this agent
        """
        # Initialize agent metrics if not exists
        if agent_type not in self.metrics[MetricCategory.AGENT]:
            self.metrics[MetricCategory.AGENT][agent_type] = {}
        
        # Add each metric
        for metric in metrics:
            if metric not in self.metrics[MetricCategory.AGENT][agent_type]:
                self.metrics[MetricCategory.AGENT][agent_type][metric] = []
                self.active_metrics.add(metric)
    
    def register_custom_metric(self, metric_name: str, category: MetricCategory = MetricCategory.CUSTOM) -> None:
        """
        Register a custom metric to track.
        
        Args:
            metric_name: Name of the metric
            category: Category to place the metric in
        """
        if category not in self.metrics:
            self.metrics[category] = {}
        
        if metric_name not in self.metrics[category]:
            self.metrics[category][metric_name] = []
            self.active_metrics.add(metric_name)
    
    def update(self, new_metrics: Dict[str, Any], agent_type: str = None) -> None:
        """Update metrics with new values.
        
        Args:
            new_metrics: Dictionary with new metric values
            agent_type: Optional agent type for agent-specific metrics
        """
        # Update basic metrics
        for metric, value in new_metrics.items():
            # Try to find the metric in each category
            metric_found = False
            
            # Check training metrics
            if metric in self.metrics[MetricCategory.TRAINING]:
                self.metrics[MetricCategory.TRAINING][metric].append(value)
                metric_found = True
            
            # Check performance metrics
            if metric in self.metrics[MetricCategory.PERFORMANCE]:
                self.metrics[MetricCategory.PERFORMANCE][metric].append(value)
                metric_found = True
            
            # Check reward components
            if metric.startswith('reward_'):
                component_name = metric[7:]  # Remove 'reward_' prefix
                if component_name not in self.metrics[MetricCategory.REWARD]:
                    self.metrics[MetricCategory.REWARD][component_name] = []
                self.metrics[MetricCategory.REWARD][component_name].append(value)
                metric_found = True
            
            # Check agent-specific metrics
            if not metric_found and agent_type and agent_type in self.metrics[MetricCategory.AGENT]:
                if metric in self.metrics[MetricCategory.AGENT][agent_type]:
                    self.metrics[MetricCategory.AGENT][agent_type][metric].append(value)
                    metric_found = True

    def save(self) -> None:
        """Save metrics to disk in both CSV and JSON formats."""
        # Flatten metrics for CSV
        flat_metrics = {}
        for category, metrics in self.metrics.items():
            if isinstance(metrics, dict):
                for metric_name, values in metrics.items():
                    if isinstance(values, dict):  # Handle nested dicts (agent metrics)
                        for agent_metric, agent_values in values.items():
                            flat_metrics[f"{metric_name}_{agent_metric}"] = agent_values
                    else:
                        flat_metrics[metric_name] = values
        
        # Find the maximum length of any metric array
        max_length = max(len(values) for values in flat_metrics.values()) if flat_metrics else 0
        
        # Pad all arrays to the same length
        padded_metrics = {}
        for metric_name, values in flat_metrics.items():
            if len(values) < max_length:
                padded_metrics[metric_name] = values + [None] * (max_length - len(values))
            else:
                padded_metrics[metric_name] = values
        
        # Save as CSV
        try:
            metrics_df = pd.DataFrame(padded_metrics)
            metrics_df.to_csv(self.metrics_dir / "metrics.csv", index=False)
        except Exception as e:
            self.logger.error(f"Error saving metrics to CSV: {e}")
        
        # Convert metrics to serializable format for JSON
        serializable_metrics = {}
        for category, metrics in self.metrics.items():
            category_name = category.name if isinstance(category, MetricCategory) else str(category)
            serializable_metrics[category_name] = {}
            
            if isinstance(metrics, dict):
                for metric_name, values in metrics.items():
                    if isinstance(values, dict):  # Handle nested dicts (agent metrics)
                        serializable_metrics[category_name][metric_name] = {
                            agent_metric: self._convert_to_serializable(agent_values)
                            for agent_metric, agent_values in values.items()
                        }
                    else:
                        serializable_metrics[category_name][metric_name] = self._convert_to_serializable(values)

        # Save as JSON with category structure
        try:
            with open(self.metrics_dir / "metrics.json", 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving metrics to JSON: {e}")
            self.logger.error(f"Metrics: {serializable_metrics}")
    
    def plot(self) -> None:
        """Generate and save plots of training metrics."""
        if not self.metrics[MetricCategory.TRAINING]["episode"]:
            return
            
        # Plot training metrics
        self._plot_category_metrics(MetricCategory.TRAINING, "Training Metrics")
        
        # Plot performance metrics
        self._plot_category_metrics(MetricCategory.PERFORMANCE, "Training Performance Metrics")
        
        # Plot reward components
        self._plot_category_metrics(MetricCategory.REWARD, "Reward Components")
        
        # Plot agent-specific metrics if any
        for agent_type, metrics in self.metrics[MetricCategory.AGENT].items():
            if metrics:  # Only plot if there are metrics
                self._plot_category_metrics(
                    MetricCategory.AGENT,
                    f"{agent_type.upper()} Agent Metrics",
                    agent_type=agent_type
                )
    
    def _plot_category_metrics(self, category: MetricCategory, title: str, agent_type: Optional[str] = None) -> None:
        """Plot metrics for a specific category."""
        # Get metrics for this category
        if category == MetricCategory.AGENT and agent_type:
            metrics = self.metrics[category].get(agent_type, {})
        else:
            metrics = self.metrics[category]
        
        if not metrics:
            return
            
        # Get episode numbers for x-axis
        episodes = self.metrics[MetricCategory.TRAINING]["episode"]
        episode_count = len(episodes)
        
        # Find metrics that can be plotted
        metrics_to_plot = []
        for metric, values in metrics.items():
            if len(values) > 0:
                if len(values) == episode_count:
                    metrics_to_plot.append((metric, values))
                elif len(values) < episode_count:
                    metrics_to_plot.append((metric, values))
        
        if not metrics_to_plot:
            return
            
        # Create figure
        n_plots = len(metrics_to_plot)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Plot each metric
        for i, (metric, values) in enumerate(metrics_to_plot):
            if len(values) == episode_count:
                axes[i].plot(episodes, values)
            else:
                recent_episodes = episodes[-len(values):] if values else []
                axes[i].plot(recent_episodes, values)
                axes[i].set_title(f"{metric} vs Episode (Partial Data)")
            
            axes[i].set_xlabel("Episode")
            axes[i].set_ylabel(metric)
            if len(values) == episode_count:
                axes[i].set_title(f"{metric} vs Episode")
            axes[i].grid(True)
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Create filename based on category and agent type
        filename = f"{category.name.lower()}"
        if agent_type:
            filename += f"_{agent_type}"
        filename += "_metrics.png"
        
        plt.savefig(self.metrics_dir / filename)
        plt.close(fig)
    
    def log_episode_results(self, episode: int, metrics: Dict[str, float]) -> None:
        """Log episode results."""
        self.logger.info(f"Episode {episode + 1} completed")
        
        # Log training metrics
        if "episode_length" in metrics:
            self.logger.info(f"  Steps: {metrics['episode_length']}")
        if "episode_reward" in metrics:
            self.logger.info(f"  Total Reward: {metrics['episode_reward']:.4f}")
        if "portfolio_value" in metrics:
            self.logger.info(f"  Portfolio Value: ${metrics['portfolio_value']:.2f}")
        
        # Log reward components
        for metric, value in metrics.items():
            if metric.startswith('reward_'):
                component_name = metric[7:]  # Remove 'reward_' prefix
                self.logger.info(f"  {component_name} Reward: {value:.4f}")
        
        # Log agent-specific metrics
        for metric in self.active_metrics:
            if metric in metrics:
                self.logger.info(f"  {metric}: {metrics[metric]:.4f}")
    
    def log_eval_results(self, episode: int, metrics: Dict[str, float]) -> None:
        """Log evaluation results."""
        self.logger.info(f"Evaluation after episode {episode + 1}")
        
        # Log evaluation metrics
        if "eval_return" in metrics:
            self.logger.info(f"  Evaluation Return: {metrics['eval_return']:.4f}")
        if "eval_portfolio_value" in metrics:
            self.logger.info(f"  Evaluation Portfolio Value: ${metrics['eval_portfolio_value']:.2f}")
        
        # Log performance metrics
        if "sharpe_ratio" in metrics:
            self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        if "max_drawdown" in metrics:
            self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        if "total_return" in metrics:
            self.logger.info(f"  Total Return: {metrics['total_return']:.2%}")
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest values for all metrics."""
        latest_metrics = {}
        for category, metrics in self.metrics.items():
            if isinstance(metrics, dict):
                for metric_name, values in metrics.items():
                    if isinstance(values, dict):  # Handle nested dicts (agent metrics)
                        for agent_metric, agent_values in values.items():
                            if agent_values:
                                latest_metrics[f"{metric_name}_{agent_metric}"] = agent_values[-1]
                    elif values:
                        latest_metrics[metric_name] = values[-1]
        return latest_metrics
    
    def get_metric_history(self, metric_name: str, agent_type: Optional[str] = None) -> List[float]:
        """Get the history of a specific metric."""
        # Search through all categories
        for category, metrics in self.metrics.items():
            if isinstance(metrics, dict):
                if agent_type and category == MetricCategory.AGENT:
                    if agent_type in metrics and metric_name in metrics[agent_type]:
                        return metrics[agent_type][metric_name]
                elif metric_name in metrics:
                    return metrics[metric_name]
        return []
    
    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert numpy types to Python native types."""
        try:
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
                return {k: MetricsManager._convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [MetricsManager._convert_to_serializable(i) for i in obj]
            elif isinstance(obj, tuple):
                return [MetricsManager._convert_to_serializable(i) for i in obj]
            elif isinstance(obj, Enum):
                return obj.name
            else:
                try:
                    return str(obj)
                except:
                    return None 
        except:
            return None 