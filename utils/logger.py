"""
Logging utility for the financial RL project.
Provides unified logging functionality across the application.
"""

import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
import numpy as np

# Try to import optional dependencies
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Unified logger class that handles console, file, and optional
    experiment tracking (Weights & Biases and/or TensorBoard).
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_dir: Optional[str] = None,
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_to_wandb: bool = False,
        log_to_tensorboard: bool = False,
        config: Optional[Dict[str, Any]] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name (usually module or class name)
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            log_dir: Directory to store log files
            log_to_file: Whether to save logs to a file
            log_to_console: Whether to output logs to console
            log_to_wandb: Whether to log metrics to Weights & Biases
            log_to_tensorboard: Whether to log metrics to TensorBoard
            config: Configuration dictionary to log
            wandb_project: W&B project name
            wandb_entity: W&B entity (team) name
            wandb_tags: Tags for W&B run
            experiment_name: Name for this experiment run
        """
        # Set up the experiment name and directories
        self.name = name
        self.level = getattr(logging, level.upper())

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{name}_{timestamp}"
        else:
            self.experiment_name = experiment_name

        if log_dir is None:
            self.log_dir = Path("logs") / self.experiment_name
        else:
            self.log_dir = Path(log_dir) / self.experiment_name

        # Create log directory if needed
        if log_to_file or log_to_tensorboard:
            os.makedirs(self.log_dir, exist_ok=True)

        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers = []  # Clear existing handlers

        # Set up formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Set up console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Set up file handler
        if log_to_file:
            log_file = self.log_dir / f"{self.experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Initialize experiment tracking
        self.use_wandb = log_to_wandb and WANDB_AVAILABLE
        self.use_tensorboard = log_to_tensorboard and TENSORBOARD_AVAILABLE

        # Log the config
        if config is not None:
            config_file = self.log_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)
            self.info(f"Configuration saved to {config_file}")

        # Initialize W&B
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                self.warning(
                    "Weights & Biases not available. Please install with 'pip install wandb'"
                )
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=config,
                    name=self.experiment_name,
                    tags=wandb_tags,
                    dir=self.log_dir,
                )
                self.info("Initialized Weights & Biases logging")

        # Initialize TensorBoard
        if self.use_tensorboard:
            if not TENSORBOARD_AVAILABLE:
                self.warning(
                    "TensorBoard not available. Please install with 'pip install tensorboard'"
                )
                self.use_tensorboard = False
            else:
                self.tb_writer = SummaryWriter(log_dir=self.log_dir / "tensorboard")
                self.info("Initialized TensorBoard logging")

    def debug(self, msg: str):
        """Log a debug message."""
        self.logger.debug(msg)

    def info(self, msg: str):
        """Log an info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log a warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log an error message."""
        self.logger.error(msg)

    def critical(self, msg: str):
        """Log a critical message."""
        self.logger.critical(msg)

    def log_metric(
        self, name: str, value: Union[float, int], step: Optional[int] = None
    ):
        """
        Log a numerical metric to experiment tracking platforms.

        Args:
            name: Name of the metric
            value: Value of the metric
            step: Training step or episode number
        """
        if self.use_wandb:
            data = {name: value}
            if step is not None:
                wandb.log(data, step=step)
            else:
                wandb.log(data)

        if self.use_tensorboard and self.tb_writer is not None:
            if step is not None:
                self.tb_writer.add_scalar(name, value, step)
            else:
                self.tb_writer.add_scalar(name, value, 0)

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step or episode number
        """
        if self.use_wandb:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

        if self.use_tensorboard and self.tb_writer is not None:
            for name, value in metrics.items():
                if step is not None:
                    self.tb_writer.add_scalar(name, value, step)
                else:
                    self.tb_writer.add_scalar(name, value, 0)

    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """
        Log a histogram of values.

        Args:
            name: Name of the histogram
            values: Array of values
            step: Training step or episode number
        """
        if self.use_tensorboard and self.tb_writer is not None:
            if step is not None:
                self.tb_writer.add_histogram(name, values, step)
            else:
                self.tb_writer.add_histogram(name, values, 0)

        # W&B also supports histograms but with different API
        if self.use_wandb:
            if step is not None:
                wandb.log({name: wandb.Histogram(values)}, step=step)
            else:
                wandb.log({name: wandb.Histogram(values)})

    def log_figure(self, name: str, figure, step: Optional[int] = None):
        """
        Log a matplotlib figure.

        Args:
            name: Name of the figure
            figure: Matplotlib figure object
            step: Training step or episode number
        """
        if self.use_wandb:
            if step is not None:
                wandb.log({name: wandb.Image(figure)}, step=step)
            else:
                wandb.log({name: wandb.Image(figure)})

        if self.use_tensorboard and self.tb_writer is not None:
            if step is not None:
                self.tb_writer.add_figure(name, figure, step)
            else:
                self.tb_writer.add_figure(name, figure, 0)

    def close(self):
        """Close all log handlers and experiment tracking sessions."""
        # Close file handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

        # Close TensorBoard writer
        if (
            self.use_tensorboard
            and hasattr(self, "tb_writer")
            and self.tb_writer is not None
        ):
            self.tb_writer.close()

        # Finish W&B run
        if self.use_wandb:
            wandb.finish()


# Create a default logger factory function
def get_logger(
    name: str, level: str = "INFO", log_dir: Optional[str] = None, **kwargs
) -> Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        **kwargs: Additional arguments to pass to Logger

    Returns:
        Configured Logger instance
    """
    return Logger(name=name, level=level, log_dir=log_dir, **kwargs)
