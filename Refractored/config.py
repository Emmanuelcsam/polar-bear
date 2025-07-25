# config.py
# Configuration management and setup utilities

import yaml
import os
import logging
from pathlib import Path
try:
    from box import Box
except ImportError:
    # Fallback if box is not installed
    class Box(dict):
        def __getattr__(self, key):
            return self[key]
        def __setattr__(self, key, value):
            self[key] = value

import torch
import torch.distributed as dist

class ConfigManager:
    """Centralized configuration management for the fiber optics analysis system."""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Loads the YAML configuration file into a Box object for dot notation access."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return Box(config_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'model.backbone')."""
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default
    
    def update(self, key_path, value):
        """Update configuration value using dot notation."""
        keys = key_path.split('.')
        config_ref = self.config
        for key in keys[:-1]:
            config_ref = config_ref[key]
        config_ref[keys[-1]] = value
    
    def save_config(self, output_path=None):
        """Save current configuration to file."""
        output_path = output_path or self.config_path
        with open(output_path, 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

def setup_logging(level=logging.INFO):
    """Sets up a standardized logger for the entire system."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def setup_distributed():
    """Initializes the distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False

def ensure_directories(config):
    """Ensure all required directories exist."""
    checkpoint_dir = Path(config.system.checkpoints_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(config.data.path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    return checkpoint_dir
