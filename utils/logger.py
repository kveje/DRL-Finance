"""
Simplified logging utility for the DRL-Finance project.
Provides unified logging functionality across the application.
"""

import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
import inspect

# Optional imports for metrics logging
try:
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    import wandb
    _HAS_METRICS_LOGGING = True
except ImportError:
    _HAS_METRICS_LOGGING = False

class LogConfig:
    """
    Centralized configuration for all loggers.
    Modify these settings to change the behavior of all loggers.
    """
    # Log levels
    CONSOLE_LOG_LEVEL = logging.INFO  # Set to logging.WARNING to reduce console output
    FILE_LOG_LEVEL = logging.INFO
    
    # Output control
    LOG_TO_CONSOLE = True            # Set to False to disable all console output
    LOG_TO_FILE = True               # Set to False to disable file logging
    LOG_TO_WANDB = False             # Set to True to enable Weights & Biases logging
    LOG_TO_TENSORBOARD = False       # Set to True to enable TensorBoard logging
    
    # Paths and naming
    LOG_DIR = "logs"                 # Base directory for log files
    LOG_FILENAME = "drl_finance.log" # Single log file for all modules
    EXPERIMENT_DIR = None            # If set, logs will go to this directory instead of LOG_DIR
    EXPERIMENT_NAME = None           # Will be auto-generated if None
    PROJECT_ROOT = None              # Will be auto-detected if None
    
    # W&B Configuration
    WANDB_PROJECT = "drl-finance"
    WANDB_ENTITY = None
    WANDB_TAGS = None
    
    # Formatting
    LOG_FORMAT = "%(asctime)s - %(name)s - %(caller_file)s:%(caller_line)d - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Internal state
    _initialized = False
    _file_handler = None
    _loggers = {}

    @classmethod
    def init(cls):
        """Initialize global logging configuration."""
        if cls._initialized:
            # Close existing file handler if we're re-initializing
            if cls._file_handler:
                cls._file_handler.close()
                cls._file_handler = None
            
        # Set project root if not manually set
        if cls.PROJECT_ROOT is None:
            # Get the project root (assuming this file is in utils/)
            cls.PROJECT_ROOT = str(Path(__file__).parent.parent)
            
        # Set experiment name if not manually set
        if cls.EXPERIMENT_NAME is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls.EXPERIMENT_NAME = f"DRL_Finance_{timestamp}"
            
        # Create log directory if needed
        if cls.EXPERIMENT_DIR:
            # Use experiment-specific directory
            cls.log_path = Path(cls.EXPERIMENT_DIR) / "logs"
            log_filename = f"{cls.EXPERIMENT_NAME}.log"
        else:
            # Use default log directory
            cls.log_path = Path(cls.PROJECT_ROOT) / cls.LOG_DIR
            log_filename = cls.LOG_FILENAME
            
        os.makedirs(cls.log_path, exist_ok=True)
        
        # Set up file handler if enabled
        if cls.LOG_TO_FILE:
            log_file = cls.log_path / log_filename
            if cls._file_handler:
                cls._file_handler.close()
            cls._file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            formatter = logging.Formatter(cls.LOG_FORMAT, datefmt=cls.DATE_FORMAT)
            cls._file_handler.setFormatter(formatter)
            cls._file_handler.setLevel(cls.FILE_LOG_LEVEL)

        cls._initialized = True

# Initialize logging configuration
LogConfig.init()

class Logger:
    """
    Simplified logger class with a focus on ease of use and consistency.
    Uses a single log file by default and provides better control over console output.
    """

    def __init__(
        self,
        name: str,
        level: Optional[int] = None,
        log_to_console: Optional[bool] = None,
        log_to_file: Optional[bool] = None,
        log_to_wandb: Optional[bool] = None,
        log_to_tensorboard: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name (usually module or class name)
            level: Logging level for this specific logger (overrides global config)
            log_to_console: Whether to output logs to console (overrides global config)
            log_to_file: Whether to save logs to a file (overrides global config)
            log_to_wandb: Whether to log metrics to W&B (overrides global config)
            log_to_tensorboard: Whether to log metrics to TensorBoard (overrides global config)
            config: Configuration dictionary to log (for W&B/TensorBoard)
        """
        self.name = name
        self.level = level if level is not None else LogConfig.FILE_LOG_LEVEL
        
        # Use instance settings if provided, otherwise use global config
        self.log_to_console = log_to_console if log_to_console is not None else LogConfig.LOG_TO_CONSOLE
        self.log_to_file = log_to_file if log_to_file is not None else LogConfig.LOG_TO_FILE
        self.log_to_wandb = log_to_wandb if log_to_wandb is not None else LogConfig.LOG_TO_WANDB
        self.log_to_tensorboard = log_to_tensorboard if log_to_tensorboard is not None else LogConfig.LOG_TO_TENSORBOARD
        
        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers = []  # Clear existing handlers
        self.logger.propagate = False  # Prevent propagation to root logger

        # Set up formatter
        formatter = logging.Formatter(LogConfig.LOG_FORMAT, datefmt=LogConfig.DATE_FORMAT)

        # Set up console handler
        if self.log_to_console:
            console_level = level if level is not None else LogConfig.CONSOLE_LOG_LEVEL
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(console_level)
            self.logger.addHandler(console_handler)

        # Add the shared file handler if logging to file
        if self.log_to_file and LogConfig._file_handler is not None:
            self.logger.addHandler(LogConfig._file_handler)
        
        # Initialize metrics logging if needed and supported
        self.tb_writer = None
        if _HAS_METRICS_LOGGING:
            if self.log_to_wandb and not wandb.run:
                wandb.init(
                    project=LogConfig.WANDB_PROJECT,
                    entity=LogConfig.WANDB_ENTITY,
                    config=config,
                    name=LogConfig.EXPERIMENT_NAME,
                    tags=LogConfig.WANDB_TAGS,
                    dir=LogConfig.log_path,
                )
                
            if self.log_to_tensorboard:
                tensorboard_dir = LogConfig.log_path / "tensorboard"
                os.makedirs(tensorboard_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)

    def _get_caller_info(self):
        """Get information about the caller of the logging method."""
        caller_frame = inspect.stack()[2]  # Skip this method and the logging method
        abs_path = caller_frame.filename
        try:
            rel_path = os.path.relpath(abs_path, LogConfig.PROJECT_ROOT)
        except ValueError:
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

    def log_metric(self, name: str, value: Union[float, int], step: Optional[int] = None):
        """
        Log a numerical metric to experiment tracking platforms.
        No-op if metrics logging is not enabled or supported.

        Args:
            name: Name of the metric
            value: Value of the metric
            step: Training step or episode number
        """
        if not _HAS_METRICS_LOGGING:
            return
            
        if self.log_to_wandb and wandb.run:
            data = {name: value}
            if step is not None:
                wandb.log(data, step=step)
            else:
                wandb.log(data)

        if self.log_to_tensorboard and self.tb_writer is not None:
            if step is not None:
                self.tb_writer.add_scalar(name, value, step)
            else:
                self.tb_writer.add_scalar(name, value, 0)

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """
        Log multiple metrics at once.
        No-op if metrics logging is not enabled or supported.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step or episode number
        """
        if not _HAS_METRICS_LOGGING:
            return
            
        if self.log_to_wandb and wandb.run:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

        if self.log_to_tensorboard and self.tb_writer is not None:
            for name, value in metrics.items():
                if step is not None:
                    self.tb_writer.add_scalar(name, value, step)
                else:
                    self.tb_writer.add_scalar(name, value, 0)

    def close(self):
        """Close all log handlers and experiment tracking sessions."""
        # Close file handlers
        for handler in self.logger.handlers[:]:
            if handler != LogConfig._file_handler:  # Don't close the shared file handler
                handler.close()
                self.logger.removeHandler(handler)
        
        # Close TensorBoard writer
        if _HAS_METRICS_LOGGING and self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None

    @classmethod
    def get_logger(cls, name: Optional[str] = None, level: Optional[int] = None, **kwargs) -> 'Logger':
        """
        Get or create a logger instance for the specified name.
        
        Args:
            name: Optional logger name. If None, uses the caller's module name
            level: Logging level
            **kwargs: Additional arguments for Logger initialization
        """
        # If no name provided, get the caller's module name
        if name is None:
            caller_frame = inspect.stack()[1]  # Get the caller's frame
            # Get the module name from the frame's filename
            name = os.path.basename(caller_frame.filename).replace('.py', '')
        
        # If we don't have a logger for this name, create one
        if name not in LogConfig._loggers:
            LogConfig._loggers[name] = cls(name, level, **kwargs)
            
        return LogConfig._loggers[name]
        
    @classmethod
    def configure(cls, **kwargs):
        """
        Update the global logging configuration.
        
        Example:
            Logger.configure(
                LOG_TO_CONSOLE=False,
                CONSOLE_LOG_LEVEL=logging.WARNING,
                LOG_FILENAME="my_custom_log.log"
            )
        """
        for key, value in kwargs.items():
            if hasattr(LogConfig, key):
                setattr(LogConfig, key, value)
            else:
                print(f"Warning: Unknown configuration option '{key}'")
        
        # Re-initialize with new settings
        LogConfig.init()
        
        # Update existing loggers with new settings
        for logger in LogConfig._loggers.values():
            # Reconnect file handler if needed
            if LogConfig._file_handler is not None and LogConfig.LOG_TO_FILE and logger.log_to_file:
                if LogConfig._file_handler not in logger.logger.handlers:
                    logger.logger.addHandler(LogConfig._file_handler)
        
        return cls
