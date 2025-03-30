from utils.logger import get_logger

# Initialize the logger once at the start of your application
logger = get_logger(
    level="INFO",
    log_dir="logs",
    log_to_file=True,
    log_to_console=True,
    # Add other configuration as needed
)
