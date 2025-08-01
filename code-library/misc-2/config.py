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
            try:
                value = self[key]
                if isinstance(value, dict):
                    return Box(value)
                return value
            except KeyError:
                raise AttributeError(f"'Box' object has no attribute '{key}'")
        
        def __setattr__(self, key, value):
            self[key] = value
        
        def __getitem__(self, key):
            value = super().__getitem__(key)
            if isinstance(value, dict) and not isinstance(value, Box):
                # Convert to Box and store it back
                box_value = Box(value)
                super().__setitem__(key, box_value)
                return box_value
            return value

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock objects for when torch is not available
    class MockDist:
        @staticmethod
        def init_process_group(*args, **kwargs):
            pass
    dist = MockDist()

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
            if key not in config_ref:
                config_ref[key] = Box({})
            elif not isinstance(config_ref[key], (dict, Box)):
                raise ValueError(f"Cannot access '{key}' as a dictionary")
            config_ref = config_ref[key]
        config_ref[keys[-1]] = value
    
    def save_config(self, output_path=None):
        """Save current configuration to file."""
        output_path = output_path or self.config_path
        
        # Convert Box to dict recursively
        def box_to_dict(obj):
            if isinstance(obj, Box):
                return {k: box_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: box_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [box_to_dict(item) for item in obj]
            else:
                return obj
        
        config_dict = box_to_dict(self.config)
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

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
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and TORCH_AVAILABLE:
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
