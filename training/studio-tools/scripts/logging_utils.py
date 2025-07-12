
import logging
import sys
import os

def setup_logger(name, log_file="script.log"):
    """Sets up a logger for a script."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Formatter
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] - %(message)s")

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        print(f"Could not write to log file {log_file}: {e}", file=sys.stderr)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    return logger

def get_logger(script_path):
    """Gets a logger for a given script path."""
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    log_file = f"{script_name}.log"
    return setup_logger(script_name, log_file)
