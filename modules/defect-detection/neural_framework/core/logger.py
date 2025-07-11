
import logging
import sys
import os
from datetime import datetime

# Check if colorlog is installed, if not, we'll fall back to the standard logger.
try:
    from colorlog import ColoredFormatter
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

# Define the base directory for the neural framework
# This makes the logger path independent of where the script is run from
FRAMEWORK_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGS_DIR = os.path.join(FRAMEWORK_BASE_DIR, 'neural_framework', 'logs')

def setup_logger():
    """
    Sets up a logger that outputs to both a timestamped file and the console.
    Console output is color-coded if the 'colorlog' package is available.
    """
    # Ensure the logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = logging.getLogger("NeuralFramework")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.hasHandlers():
        return logger

    # 1. File Handler
    log_filename = f"framework_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_filepath = os.path.join(LOGS_DIR, log_filename)
    file_handler = logging.FileHandler(log_filepath)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # 2. Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    
    if COLORLOG_AVAILABLE:
        console_formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
            reset=True,
            style='%'
        )
    else:
        console_formatter = logging.Formatter("%(levelname)-8s - %(message)s")

    stream_handler.setFormatter(console_formatter)
    stream_handler.setLevel(logging.INFO) # Keep console output less verbose by default

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if not COLORLOG_AVAILABLE:
        logger.warning("`colorlog` package not found. Console output will not be colored.")
        logger.warning("For a better experience, please consider installing it: pip install colorlog")

    return logger

# Create a default logger instance to be imported by other modules
log = setup_logger()
