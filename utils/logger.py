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
import inspect

from torch.utils.tensorboard import SummaryWriter
import wandb

class Logger:
    """
    Unified logger class that handles console, file, and optional
    experiment tracking (Weights & Biases and/or TensorBoard).
    """

    _instance = None
    _loggers = {}  # Dictionary to store module-specific loggers
    _initialized = False  # Flag to track if the main logger has been initialized
    _file_handler = None  # Shared file handler for all loggers
    _project_root = None  # Project root directory
    _experiment_name = None  # Shared experiment name for all loggers

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

        # Set project root if not set
        if Logger._project_root is None:
            # Get the project root (assuming this file is in utils/)
            Logger._project_root = str(Path(__file__).parent.parent)

        # Set experiment name if not set
        if Logger._experiment_name is None:
            if experiment_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                Logger._experiment_name = f"DRL_Finance_{timestamp}"
            else:
                Logger._experiment_name = experiment_name

        self.experiment_name = Logger._experiment_name

        if log_dir is None:
            self.log_dir = Path(Logger._project_root) / "logs" / self.experiment_name
        else:
            self.log_dir = Path(log_dir) / self.experiment_name

        # Create log directory if needed
        if log_to_file or log_to_tensorboard:
            os.makedirs(self.log_dir, exist_ok=True)

        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers = []  # Clear existing handlers
        self.logger.propagate = False  # Prevent propagation to root logger

        # Set up formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(caller_file)s:%(caller_line)d - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Set up console handler
        if log_to_console and not Logger._initialized:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Set up file handler only for the first logger instance
        if log_to_file and not Logger._initialized:
            log_file = self.log_dir / f"{self.experiment_name}.log"
            Logger._file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            Logger._file_handler.setFormatter(formatter)
            Logger._file_handler.setLevel(self.level)  # Set the level for the file handler
            Logger._initialized = True

        # Add the shared file handler to this logger
        if Logger._file_handler is not None:
            self.logger.addHandler(Logger._file_handler)
            # Ensure the logger's level is set correctly
            self.logger.setLevel(self.level)

        # Initialize experiment tracking only for the first logger instance
        if not Logger._initialized:
            self.use_wandb = log_to_wandb
            self.use_tensorboard = log_to_tensorboard

            # Log the config
            if config is not None:
                config_file = self.log_dir / "config.json"
                with open(config_file, "w") as f:
                    json.dump(config, f, indent=4)
                self.info(f"Configuration saved to {config_file}")

            # Initialize W&B
            if self.use_wandb:
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
                self.tb_writer = SummaryWriter(log_dir=self.log_dir / "tensorboard")
                self.info("Initialized TensorBoard logging")
        else:
            self.use_wandb = False
            self.use_tensorboard = False
            self.tb_writer = None

    def _get_caller_info(self):
        """Get information about the caller of the logging method."""
        caller_frame = inspect.stack()[2]  # Skip this method and the logging method
        # Get the absolute path and convert to relative path from project root
        abs_path = caller_frame.filename
        try:
            # Try to get the relative path from the project root
            rel_path = os.path.relpath(abs_path, Logger._project_root)
        except ValueError:
            # If we can't get the relative path, just use the filename
            rel_path = os.path.basename(abs_path)
            
        return {
            'caller_file': rel_path,
            'caller_line': caller_frame.lineno,
            'caller_func': caller_frame.function
        }

    def debug(self, msg: str):
        """Log a debug message."""
        self.logger.debug(msg, extra=self._get_caller_info())

    def info(self, msg: str):
        """Log an info message."""
        self.logger.info(msg, extra=self._get_caller_info())

    def warning(self, msg: str):
        """Log a warning message."""
        self.logger.warning(msg, extra=self._get_caller_info())

    def error(self, msg: str):
        """Log an error message."""
        self.logger.error(msg, extra=self._get_caller_info())

    def critical(self, msg: str):
        """Log a critical message."""
        self.logger.critical(msg, extra=self._get_caller_info())

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
            if handler != Logger._file_handler:  # Don't close the shared file handler
                handler.close()
                self.logger.removeHandler(handler)

        # Close the shared file handler only if this is the last logger
        if Logger._file_handler is not None and len(Logger._loggers) == 1:
            Logger._file_handler.close()
            Logger._file_handler = None
            Logger._initialized = False

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

    @classmethod
    def get_logger(cls, name: Optional[str] = None, level: str = "INFO", log_dir: Optional[str] = None, **kwargs) -> 'Logger':
        """
        Get or create a logger instance for the specified name.
        
        Args:
            name: Optional logger name. If None, uses the caller's module name
            level: Logging level
            log_dir: Directory for log files
            **kwargs: Additional arguments for Logger initialization
        """
        # If no name provided, get the caller's module name
        if name is None:
            # Get the caller's module name
            stack = inspect.stack()
            # Skip the first frame (this function) and get the caller's frame
            caller_frame = stack[1]
            # Get the module name from the frame's globals
            name = caller_frame.filename.split('\\')[-1].replace('.py', '')
        
        # If we don't have a logger for this name, create one
        if name not in cls._loggers:
            cls._loggers[name] = cls(name, level, log_dir, **kwargs)
            
        return cls._loggers[name]
