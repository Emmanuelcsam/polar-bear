"""
Simple logging setup module.
This provides basic logging functionality without complex async features.
"""

import sys
import logging


def setup_logging(log_level="DEBUG"):
    """
    Configures a simple logging system.
    """
    # Convert string log level to logging constant
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Formatter for all log messages
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(threadName)-12s - %(levelname)-8s - '
        '%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler to send logs to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    root_logger.addHandler(console_handler)

    logging.info("Logging system initialized.")
    return None, None  # Return None for compatibility with existing code 