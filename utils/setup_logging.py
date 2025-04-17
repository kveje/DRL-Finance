"""
Utility to set up logging for the application.
Import and run setup_logging() at the beginning of your application.
"""

import logging
import os
from pathlib import Path
from utils.logger import Logger, LogConfig

def setup_logging(
    log_to_console=False,
    console_level=logging.INFO,
    log_to_file=True,
    file_level=logging.DEBUG,
    log_dir="logs",
    log_filename=None,
    experiment_name=None,
    experiment_dir=None
):
    """
    Set up logging for the application.
    
    This function should be called at the beginning of your application to configure
    the logging system for all modules.
    
    Args:
        log_to_console: Whether to output logs to console
        console_level: Logging level for console output
        log_to_file: Whether to save logs to a file
        file_level: Logging level for file output
        log_dir: Directory for log files (used if experiment_dir is None)
        log_filename: Custom log filename (default is 'drl_finance.log' or '{experiment_name}.log')
        experiment_name: Custom experiment name
        experiment_dir: If specified, logs will be saved in this directory instead of log_dir
    """
    # Configure the logger system
    Logger.configure(
        LOG_TO_CONSOLE=log_to_console,
        CONSOLE_LOG_LEVEL=console_level,
        LOG_TO_FILE=log_to_file,
        FILE_LOG_LEVEL=file_level,
        LOG_DIR=log_dir,
        LOG_FILENAME=log_filename or "drl_finance.log",
        EXPERIMENT_NAME=experiment_name,
        EXPERIMENT_DIR=experiment_dir
    )
    
    # Get the root logger to apply basic configuration
    root_logger = logging.getLogger()
    
    # Create a logger for this module
    logger = Logger.get_logger("setup_logging")
    logger.info("Logging system initialized")
    
    # Log the location where logs will be saved
    if experiment_dir:
        logger.info(f"Experiment logs will be saved to: {Path(experiment_dir)}")
    else:
        logger.info(f"Log directory: {Path(LogConfig.PROJECT_ROOT) / LogConfig.LOG_DIR}")
    
    # Log the filename being used
    if experiment_dir:
        log_filename = f"{LogConfig.EXPERIMENT_NAME}.log" if not log_filename else log_filename
    else:
        log_filename = LogConfig.LOG_FILENAME
    
    logger.info(f"Log file: {log_filename}")
    
    return logger

def disable_console_logging():
    """
    Disable all console output for all loggers.
    Useful when running in background or as a service.
    """
    Logger.configure(LOG_TO_CONSOLE=False)
    
def set_console_level(level):
    """
    Set the console logging level.
    
    Args:
        level: Logging level (e.g., logging.WARNING)
    """
    Logger.configure(CONSOLE_LOG_LEVEL=level)

# Example usage
if __name__ == "__main__":
    # Set up logging with custom settings
    setup_logging(
        log_to_console=True,
        console_level=logging.INFO,
        log_filename="my_application.log"
    )
    
    # Get a logger for this module
    logger = Logger.get_logger()
    
    # Use the logger
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Disable console output
    disable_console_logging()
    
    # This will only go to the log file, not to console
    logger.info("This message will only appear in the log file") 