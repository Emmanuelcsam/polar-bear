"""
Core functionality for fiber optic defect detection
"""

from .utils.logging import get_logger
from .utils.config import get_config, get_config_manager

__all__ = ['get_logger', 'get_config', 'get_config_manager']