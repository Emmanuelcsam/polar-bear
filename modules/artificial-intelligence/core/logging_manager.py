
import logging
import sys
from datetime import datetime
from pathlib import Path

# Color codes for console output
class ColorFormatter(logging.Formatter):
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    GREEN = "\x1b[32m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.GREY + self.fmt + self.RESET,
            logging.INFO: self.GREEN + self.fmt + self.RESET,
            logging.WARNING: self.YELLOW + self.fmt + self.RESET,
            logging.ERROR: self.RED + self.fmt + self.RESET,
            logging.CRITICAL: self.BOLD_RED + self.fmt + self.RESET
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging():
    """
    Sets up dual logging to a timestamped file and the console.
    """
    log_directory = Path("logs")
    log_directory.mkdir(exist_ok=True)
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_filepath = log_directory / log_filename

    # General log format
    log_format = "[%(asctime)s] [%(levelname)-8s] [%(name)-15s] --- %(message)s"
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter(log_format))
    root_logger.addHandler(console_handler)

    logging.getLogger("main").info("Logging system initialized.")
    logging.getLogger("main").info(f"Log file available at: {log_filepath}")

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with the specified name.
    """
    return logging.getLogger(name)
