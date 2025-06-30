#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Logging Module for Neural Nexus
Supports both loguru (preferred) and standard logging as fallback.
"""
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .config import config, LOGS_DIR
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import config, LOGS_DIR


class Logger:
    """Enhanced logger with structured output and performance optimization."""

    def __init__(self):
        self.use_loguru = config.features.get('loguru', False)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logger based on available libraries."""
        if self.use_loguru:
            return self._setup_loguru()
        else:
            return self._setup_standard_logging()

    def _setup_loguru(self):
        """Setup loguru-based logging."""
        try:
            from loguru import logger

            # Remove default logger
            logger.remove()

            # Add structured JSON logging for production
            log_file = LOGS_DIR / f"server_{datetime.now():%Y%m%d_%H%M%S}.log"
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                level="INFO",
                rotation="10 MB",
                compression="zip",
                serialize=True,  # JSON format for easier parsing
                retention="30 days"
            )

            # Console output with colors
            logger.add(
                sys.stderr,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
                level="DEBUG" if config.debug else "INFO",
                colorize=True
            )

            return logger
        except ImportError:
            return self._setup_standard_logging()

    def _setup_standard_logging(self):
        """Setup standard Python logging as fallback."""
        log_file = LOGS_DIR / f"server_{datetime.now():%Y%m%d_%H%M%S}.log"

        logging.basicConfig(
            level=logging.DEBUG if config.debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        return logging.getLogger(__name__)

    def info(self, message: str, **kwargs):
        """Log info message."""
        if self.use_loguru:
            self.logger.info(message, **kwargs)
        else:
            self.logger.info(message)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if self.use_loguru:
            self.logger.warning(message, **kwargs)
        else:
            self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """Log error message."""
        if self.use_loguru:
            self.logger.error(message, **kwargs)
        else:
            self.logger.error(message)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        if self.use_loguru:
            self.logger.debug(message, **kwargs)
        else:
            self.logger.debug(message)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        if self.use_loguru:
            self.logger.critical(message, **kwargs)
        else:
            self.logger.critical(message)


# Global logger instance
logger = Logger()
