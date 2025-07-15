import logging
import os
from pathlib import Path

def setup_logging(name: str = __name__):
    """Configure logging with optional file output when DEBUG_MODE is on."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level = logging.DEBUG if os.getenv("DEBUG_MODE") else logging.INFO
    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if os.getenv("DEBUG_MODE"):
        log_file = Path(os.getenv("DEBUG_LOG_FILE", "debug.log"))
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
