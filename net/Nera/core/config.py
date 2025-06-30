#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Nexus Configuration Module
Centralized configuration management with environment detection.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Core directories
HOME_DIR = Path.home() / ".neural_nexus_server"
SCRIPTS_DIR = HOME_DIR / "scripts"
LOGS_DIR = HOME_DIR / "logs"
TEMP_DIR = HOME_DIR / "temp"
PROJECTS_DIR = HOME_DIR / "projects"
CACHE_DIR = HOME_DIR / "cache"

# Create directories
for dir_path in [SCRIPTS_DIR, LOGS_DIR, TEMP_DIR, PROJECTS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Feature availability detection
def check_feature_availability() -> Dict[str, bool]:
    """Check availability of optional enhancement features."""
    features = {}

    # Performance enhancements
    try:
        import uvloop
        features['uvloop'] = True
    except ImportError:
        features['uvloop'] = False

    try:
        import orjson
        features['orjson'] = True
    except ImportError:
        features['orjson'] = False

    # Logging
    try:
        import loguru
        features['loguru'] = True
    except ImportError:
        features['loguru'] = False

    # Security and rate limiting
    try:
        import slowapi
        features['slowapi'] = True
    except ImportError:
        features['slowapi'] = False

    # Code analysis tools
    try:
        import semgrep
        features['semgrep'] = True
    except ImportError:
        features['semgrep'] = False

    try:
        import bandit
        features['bandit'] = True
    except ImportError:
        features['bandit'] = False

    try:
        import ruff
        features['ruff'] = True
    except ImportError:
        features['ruff'] = False

    try:
        import mypy
        features['mypy'] = True
    except ImportError:
        features['mypy'] = False

    try:
        import pylint
        features['pylint'] = True
    except ImportError:
        features['pylint'] = False

    try:
        import friendly_traceback
        features['friendly_traceback'] = True
    except ImportError:
        features['friendly_traceback'] = False

    # Serialization
    try:
        import msgspec
        features['msgspec'] = True
    except ImportError:
        features['msgspec'] = False

    # AI/OpenAI
    try:
        import openai
        features['openai'] = True
    except ImportError:
        features['openai'] = False

    return features

# Server configuration
class ServerConfig:
    """Server configuration with environment-based settings."""

    def __init__(self):
        self.features = check_feature_availability()
        self.host = os.getenv("NEURAL_NEXUS_HOST", "127.0.0.1")
        self.port = int(os.getenv("NEURAL_NEXUS_PORT", "8765"))
        self.debug = os.getenv("NEURAL_NEXUS_DEBUG", "false").lower() == "true"
        self.max_workers = int(os.getenv("NEURAL_NEXUS_MAX_WORKERS", "4"))
        self.max_heal_attempts = int(os.getenv("NEURAL_NEXUS_MAX_HEAL_ATTEMPTS", "10"))

        # Security settings
        self.security_enabled = os.getenv("NEURAL_NEXUS_SECURITY_ENABLED", "true").lower() == "true"
        self.rate_limit_enabled = self.features['slowapi'] and self.security_enabled

        # Analysis settings
        self.analysis_cache_ttl = int(os.getenv("NEURAL_NEXUS_ANALYSIS_CACHE_TTL", "3600"))  # 1 hour
        self.auto_format_enabled = self.features['ruff']
        self.security_scan_enabled = self.features['semgrep'] or self.features['bandit']

        # Performance settings
        self.use_uvloop = self.features['uvloop']
        self.use_orjson = self.features['orjson']

        # Add paths to sys.path
        self._setup_python_paths()

    def _setup_python_paths(self):
        """Add necessary directories to Python path."""
        paths_to_add = [str(TEMP_DIR), str(PROJECTS_DIR)]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.append(path)

    def get_feature_status(self) -> Dict[str, str]:
        """Get formatted feature status for display."""
        status = {}
        feature_descriptions = {
            'uvloop': 'Enhanced event loop (4x faster)',
            'orjson': 'Ultra-fast JSON (6x faster)',
            'loguru': 'Structured logging',
            'slowapi': 'Rate limiting & security',
            'semgrep': 'Security scanning',
            'bandit': 'Vulnerability detection',
            'ruff': 'Fast linting & formatting',
            'mypy': 'Type checking',
            'pylint': 'Advanced code analysis',
            'friendly_traceback': 'Enhanced error messages',
            'msgspec': 'Fast serialization',
            'openai': 'AI-powered analysis',
        }

        for feature, description in feature_descriptions.items():
            available = self.features.get(feature, False)
            status[description] = '✓' if available else '✗'

        return status

# Global configuration instance
config = ServerConfig()
