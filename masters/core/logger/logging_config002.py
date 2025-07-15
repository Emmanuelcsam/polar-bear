
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(log_level="INFO"):
    """
    Configures a robust logging system for the application.

    This setup creates a logger that outputs to both the console and a
    timestamped file in the 'logs' directory. It includes detailed context
    in each log message.

    Args:
        log_level (str): The minimum logging level to capture (e.g., "DEBUG", "INFO").
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create a unique, timestamped log file for each run
    log_file = log_dir / f"neural_connector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Define the detailed format for log messages
    log_format = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s"
    )

    # Get the root logger
    logger = logging.getLogger("neural_connector")
    logger.setLevel(log_level)

    # --- Console Handler ---
    # Outputs logs to the standard system output (the terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # --- File Handler ---
    # Writes logs to the timestamped file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logger.info("Logging system initialized. Logs will be saved to %s", log_file)
    return logger

# Example of how to get the logger in other modules:
# import logging
# logger = logging.getLogger("neural_connector")
