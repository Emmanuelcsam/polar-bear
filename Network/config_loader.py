#!/usr/bin/env python3
"""
Advanced Configuration Loader for Fiber Optics Neural Network
Loads and manages all parameters from YAML configuration file
"I want the entire program to run from a config file where I can tweak 
every variable, numeric, parameter, weight etc."
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import os
from collections import OrderedDict

# from logger import get_logger  # Avoid circular import


class ConfigLoader:
    """
    Configuration loader with validation and dynamic updates
    Allows real-time parameter modification
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to YAML configuration file
        """
        print(f"[{datetime.now()}] Initializing ConfigLoader")
        
        # Logger removed to avoid circular import
        self.config_path = Path(config_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Validate configuration
        self._validate_config()
        
        # Setup watchers for dynamic updates
        self._setup_watchers()
        
        # Track modifications
        self.modifications = []
        self.original_config = self._deep_copy(self.config)
        
        # Legacy compatibility - fixed parameters
        self.INPUT_SIZE = (256, 256)
        self.NUM_REGIONS = 3
        self.BATCH_SIZE = self.config.training.batch_size
        self.LEARNING_RATE = self.config.optimizer.learning_rate
        self.NUM_EPOCHS = self.config.training.num_epochs
        self.SIMILARITY_THRESHOLD = self.config.similarity.threshold
        self.ANOMALY_THRESHOLD = self.config.anomaly.threshold
        self.FEATURE_CHANNELS = [64, 128, 256, 512]  # Feature channels at each scale
        self.REFERENCE_PATH = Path(self.config.system.reference_data_path)
        self.TENSORIZED_DATA_PATH = Path(self.config.system.tensorized_data_path)
        
        # Region categories for data organization
        self.REGION_CATEGORIES = {
            'core': ['core-batch-1', 'core-batch-2', 'core-batch-3', 'core-batch-4',
                    'core-batch-5', 'core-batch-6', 'core-batch-7', 'core-batch-8'],
            'cladding': ['cladding-batch-1', 'cladding-batch-3', 'cladding-batch-4',
                        'cladding-batch-5', 'cladding-features-batch-1', '50-cladding', '91-cladding'],
            'ferrule': ['ferrule-batch-1', 'ferrule-batch-2', 'ferrule-batch-3', 'ferrule-batch-4'],
            'defects': ['dirty-image', 'scratch-library-bmp', '91-scratched']
        }
        
        print(f"[{datetime.now()}] Configuration loaded from {config_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            print(f"[{datetime.now()}] ERROR: Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Convert to nested dictionary with dot notation access
            config = self._create_nested_dict(config)
            
            return config
            
        except yaml.YAMLError as e:
            print(f"[{datetime.now()}] ERROR: Error parsing YAML configuration: {e}")
            raise
    
    def _create_nested_dict(self, d: Dict) -> 'NestedDict':
        """Create nested dictionary with dot notation access"""
        if isinstance(d, dict):
            return NestedDict({k: self._create_nested_dict(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self._create_nested_dict(item) for item in d]
        else:
            return d
    
    def _validate_config(self):
        """Validate configuration values"""
        print(f"[{datetime.now()}] Validating configuration...")
        
        # Check required sections
        required_sections = ['system', 'model', 'equation', 'optimizer', 
                           'loss', 'similarity', 'training']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate specific parameters
        # Learning rate
        if self.config.optimizer.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Batch size
        if self.config.training.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        # Similarity threshold
        if not 0 <= self.config.similarity.threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        # Equation coefficients
        for coef_name, coef_value in self.config.equation.coefficients.items():
            if not self.config.equation.min_coefficient <= coef_value <= self.config.equation.max_coefficient:
                raise ValueError(f"Coefficient {coef_name} out of bounds")
        
        print(f"[{datetime.now()}] Configuration validation successful")
    
    def _setup_watchers(self):
        """Setup file watchers for dynamic configuration updates"""
        # This would use watchdog or similar library in production
        self.last_modified = os.path.getmtime(self.config_path)
    
    def _deep_copy(self, obj):
        """Deep copy configuration object"""
        if isinstance(obj, NestedDict):
            return NestedDict({k: self._deep_copy(v) for k, v in obj.items()})
        elif isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def reload(self):
        """Reload configuration from file"""
        print(f"[{datetime.now()}] Reloading configuration...")
        self.config = self._load_config()
        self._validate_config()
        print(f"[{datetime.now()}] Configuration reloaded")
    
    def check_for_updates(self) -> bool:
        """Check if configuration file has been modified"""
        current_modified = os.path.getmtime(self.config_path)
        if current_modified > self.last_modified:
            self.last_modified = current_modified
            return True
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'model.base_channels')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            value = self.config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value dynamically
        
        Args:
            key: Configuration key (e.g., 'model.base_channels')
            value: New value
        """
        print(f"[{datetime.now()}] Setting {key} = {value}")
        
        # Track modification
        self.modifications.append({
            'timestamp': datetime.now(),
            'key': key,
            'old_value': self.get(key),
            'new_value': value
        })
        
        # Set value
        parts = key.split('.')
        obj = self.config
        for part in parts[:-1]:
            obj = obj[part]
        obj[parts[-1]] = value
        
        # Validate after change
        try:
            self._validate_config()
        except ValueError as e:
            # Rollback change
            obj[parts[-1]] = self.modifications[-1]['old_value']
            self.modifications.pop()
            raise ValueError(f"Invalid configuration change: {e}")
    
    def get_device(self) -> torch.device:
        """Get torch device based on configuration"""
        return self._get_device()
    
    def get_optimizer_config(self) -> Dict:
        """Get optimizer configuration as dictionary"""
        return {
            'type': self.config.optimizer.type,
            'lr': self.config.optimizer.learning_rate,
            'weight_decay': self.config.optimizer.weight_decay,
            'betas': self.config.optimizer.betas,
            'eps': self.config.optimizer.eps,
            'sam_rho': self.config.optimizer.sam_rho,
            'sam_adaptive': self.config.optimizer.sam_adaptive,
            'lookahead_k': self.config.optimizer.lookahead_k,
            'lookahead_alpha': self.config.optimizer.lookahead_alpha,
        }
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss function weights"""
        return dict(self.config.loss.weights)
    
    def get_equation_coefficients(self) -> Dict[str, float]:
        """Get equation coefficients"""
        return dict(self.config.equation.coefficients)
    
    def save_modifications(self, path: Optional[str] = None):
        """Save configuration modifications to file"""
        if path is None:
            path = self.config_path.with_suffix('.modifications.json')
        
        with open(path, 'w') as f:
            json.dump(self.modifications, f, indent=2, default=str)
        
        print(f"[{datetime.now()}] Saved {len(self.modifications)} modifications to {path}")
    
    def export_current_config(self, path: Optional[str] = None):
        """Export current configuration to new file"""
        if path is None:
            path = self.config_path.with_suffix('.current.yaml')
        
        # Convert NestedDict back to regular dict
        config_dict = self._to_dict(self.config)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"[{datetime.now()}] Exported current configuration to {path}")
    
    def _to_dict(self, obj):
        """Convert NestedDict to regular dict"""
        if isinstance(obj, NestedDict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_dict(item) for item in obj]
        else:
            return obj
    
    def reset_to_original(self):
        """Reset configuration to original values"""
        print(f"[{datetime.now()}] Resetting configuration to original values")
        self.config = self._deep_copy(self.original_config)
        self.modifications = []
    
    def get_diff(self) -> Dict:
        """Get differences between current and original configuration"""
        return self._compare_configs(self.original_config, self.config)
    
    def _compare_configs(self, original, current, prefix=''):
        """Recursively compare configurations"""
        diff = {}
        
        if isinstance(original, (dict, NestedDict)) and isinstance(current, (dict, NestedDict)):
            all_keys = set(original.keys()) | set(current.keys())
            for key in all_keys:
                sub_prefix = f"{prefix}.{key}" if prefix else key
                if key not in original:
                    diff[sub_prefix] = {'status': 'added', 'value': current[key]}
                elif key not in current:
                    diff[sub_prefix] = {'status': 'removed', 'value': original[key]}
                else:
                    sub_diff = self._compare_configs(original[key], current[key], sub_prefix)
                    diff.update(sub_diff)
        elif original != current:
            diff[prefix] = {
                'status': 'modified',
                'original': original,
                'current': current
            }
        
        return diff
    
    def get_region_folders(self, region: str) -> List[str]:
        """Get folder names for a specific region category"""
        return self.REGION_CATEGORIES.get(region, [])
    
    def _get_device(self) -> torch.device:
        """Get PyTorch device based on configuration"""
        device_config = self.config.system.device
        
        if device_config == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_config == "cuda":
            if torch.cuda.is_available():
                return torch.device(f"cuda:{self.config.system.gpu_id}")
            else:
                print(f"[{datetime.now()}] WARNING: CUDA requested but not available, using CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")
    
    def save_config_json(self, path: Optional[str] = None):
        """Save configuration as JSON (alternative format)"""
        if path is None:
            path = self.config_path.with_suffix('.json')
        
        with open(path, 'w') as f:
            json.dump(self._to_dict(self.config), f, indent=2)
        
        print(f"[{datetime.now()}] Configuration saved to JSON: {path}")
    
    def load_config_json(self, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        self.config = self._create_nested_dict(config_dict)
        print(f"[{datetime.now()}] Configuration loaded from JSON: {path}")
    
    def _to_dict(self, nested_dict):
        """Convert NestedDict to regular dict"""
        if isinstance(nested_dict, NestedDict):
            return {k: self._to_dict(v) for k, v in nested_dict.items()}
        elif isinstance(nested_dict, list):
            return [self._to_dict(item) for item in nested_dict]
        else:
            return nested_dict


class NestedDict(OrderedDict):
    """
    Dictionary subclass that allows dot notation access
    Makes configuration more convenient to use
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = NestedDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")


class ConfigManager:
    """
    Global configuration manager singleton
    Provides centralized access to configuration
    """
    
    _instance = None
    _config_loader = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __getattr__(self, name):
        """Forward attribute access to the config object"""
        if self._config_loader is None:
            self.initialize()
        return getattr(self._config_loader.config, name)
    
    def initialize(self, config_path: str = "config.yaml"):
        """Initialize configuration manager"""
        if self._config_loader is None:
            self._config_loader = ConfigLoader(config_path)
    
    @property
    def config(self):
        """Get configuration object"""
        if self._config_loader is None:
            self.initialize()
        return self._config_loader.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if self._config_loader is None:
            self.initialize()
        return self._config_loader.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        if self._config_loader is None:
            self.initialize()
        self._config_loader.set(key, value)
    
    def reload(self):
        """Reload configuration"""
        if self._config_loader is None:
            self.initialize()
        self._config_loader.reload()
    
    def get_device(self) -> torch.device:
        """Get PyTorch device"""
        if self._config_loader is None:
            self.initialize()
        return self._config_loader._get_device()
    
    def get_region_folders(self, region: str) -> List[str]:
        """Get folder names for a specific region category"""
        if self._config_loader is None:
            self.initialize()
        return self._config_loader.get_region_folders(region)


# Global configuration instance
config_manager = ConfigManager()


def get_config():
    """Get global configuration manager"""
    return config_manager


def update_config(key: str, value: Any):
    """Update configuration value"""
    config_manager.set(key, value)


# Test the configuration system
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing configuration loader")
    
    # Load configuration
    loader = ConfigLoader("config.yaml")
    
    # Test dot notation access
    print(f"\nBase channels: {loader.config.model.base_channels}")
    print(f"Learning rate: {loader.config.optimizer.learning_rate}")
    print(f"Similarity threshold: {loader.config.similarity.threshold}")
    
    # Test getting values
    print(f"\nGet test: {loader.get('model.use_se_blocks')}")
    print(f"Get with default: {loader.get('nonexistent.key', 'default_value')}")
    
    # Test setting values
    print(f"\nOriginal LR: {loader.get('optimizer.learning_rate')}")
    loader.set('optimizer.learning_rate', 0.0001)
    print(f"Updated LR: {loader.get('optimizer.learning_rate')}")
    
    # Test equation coefficients
    print(f"\nEquation coefficients: {loader.get_equation_coefficients()}")
    
    # Test modifications tracking
    loader.set('training.batch_size', 32)
    print(f"\nModifications: {len(loader.modifications)}")
    
    # Test diff
    diff = loader.get_diff()
    print(f"\nConfiguration differences:")
    for key, change in diff.items():
        print(f"  {key}: {change}")
    
    # Save modifications
    loader.save_modifications()
    
    # Export current config
    loader.export_current_config()
    
    print(f"\n[{datetime.now()}] Configuration loader test completed")
    print(f"[{datetime.now()}] Next script: fiber_visualization_ui.py")