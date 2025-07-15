import logging
import os
from datetime import datetime

LOG_DIR = "neural_framework/logs"

def setup_logger(name="NeuralFramework", log_level=logging.DEBUG):
    """
    Sets up a logger that logs to both the console and a file.
    """
    # Get the absolute path for the log directory
    abs_log_dir = os.path.abspath(LOG_DIR)

    if not os.path.exists(abs_log_dir):
        os.makedirs(abs_log_dir)

    log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".log"
    log_filepath = os.path.join(abs_log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent duplicate logs in parent loggers

    # Remove existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG) # Changed to DEBUG
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# A global logger instance
log = setup_logger()

if __name__ == '__main__':
    # Example Usage
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    log.critical("This is a critical message.")