#!/usr/bin/env python3
"""
Statistical Configuration Loader for Fiber Optics Neural Network.
Loads statistical parameters from a dedicated YAML file and merges them
into the main configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from core.config_loader import ConfigManager

# FIX: This entire file has been refactored.
# Instead of hardcoding parameters, this module now loads them from
# "config/statistical_config.yaml", adhering to the principle of keeping
# all configuration in external, editable files.

def get_statistical_config(config_path: str = "config/statistical_config.yaml") -> Dict[str, Any]:
    """
    Loads the statistical configuration from its YAML file.

    Args:
        config_path: Path to the statistical YAML configuration file.

    Returns:
        A dictionary containing the statistical configuration.
    """
    print(f"[{datetime.now()}] Loading StatisticalConfig from {config_path}")
    
    stat_config_path = Path(config_path)
    
    if not stat_config_path.exists():
        print(f"[{datetime.now()}] ERROR: Statistical config file not found: {stat_config_path}")
        print("Please ensure 'config/statistical_config.yaml' exists for research mode.")
        raise FileNotFoundError(f"Statistical config file not found: {stat_config_path}")
        
    try:
        with open(stat_config_path, 'r') as f:
            statistical_config = yaml.safe_load(f)
        return statistical_config
    except yaml.YAMLError as e:
        print(f"[{datetime.now()}] ERROR: Error parsing statistical YAML configuration: {e}")
        raise

def merge_with_base_config(config_manager: 'ConfigManager', 
                           statistical_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Merges statistical configuration directly into the main ConfigManager instance.
    This makes the statistical parameters available via the global config object
    (e.g., config.statistical.feature_dim).

    Args:
        config_manager: The global ConfigManager instance to merge into.
        statistical_config: A dictionary of statistical config (loads default if None).
    """
    if statistical_config is None:
        statistical_config = get_statistical_config()
    
    print(f"[{datetime.now()}] Merging statistical config into base config.")

    # FIX: Merging is now done by updating the live config object.
    # This uses the existing `set` method in ConfigLoader, which is safer
    # as it can trigger validation and handles nested creation.
    
    # Add the entire statistical block under a 'statistical' key
    config_manager.set('statistical', statistical_config)
    
    # Overwrite specific values in the base config for seamless integration
    # This allows other parts of the code to not worry about which mode is active.
    if 'model' in config_manager.config:
        stat_features_config = statistical_config.get('statistical_features', {})
        config_manager.set('model.use_statistical_features', stat_features_config.get('enabled', False))
        config_manager.set('model.statistical_feature_dim', stat_features_config.get('feature_dim', 88))
    
    if 'training' in config_manager.config:
        loss_settings = statistical_config.get('loss_settings', {})
        config_manager.set('training.use_statistical_loss', loss_settings.get('loss_type') == 'composite')
    
    # Update loss weights and other settings from the statistical config
    if 'loss_settings' in statistical_config:
        # This replaces the 'loss' section of the main config with the statistical one
        config_manager.set('loss', statistical_config['loss_settings'])
        
    print(f"[{datetime.now()}] Statistical config merged successfully.")

if __name__ == '__main__':
    print("Testing statistical config loading and merging.")
    from core.config_loader import get_config, config_manager

    # Ensure dummy configs exist
    dummy_main_path = Path("config/config.yaml")
    dummy_stat_path = Path("config/statistical_config.yaml")
    dummy_main_path.parent.mkdir(exist_ok=True)
    
    if not dummy_main_path.exists():
        with open(dummy_main_path, "w") as f:
            yaml.dump({'runtime': {}, 'model': {}, 'training': {}, 'loss': {}}, f)
            
    if not dummy_stat_path.exists():
        with open(dummy_stat_path, "w") as f:
            yaml.dump({
                'statistical_features': {'enabled': True, 'feature_dim': 88},
                'loss_settings': {'loss_type': 'composite', 'weights': {'stat_loss': 1.0}}
            }, f)

    # 1. Load the base config
    config = get_config()
    print("\n--- Base Config Loaded ---")
    print(f"use_statistical_features: {config.model.use_statistical_features}")
    print(f"loss type: {config.loss.get('loss_type', 'standard')}")

    # 2. Merge the statistical config
    merge_with_base_config(config_manager)
    print("\n--- Statistical Config Merged ---")
    print(f"use_statistical_features: {config.model.use_statistical_features}")
    print(f"statistical_feature_dim: {config.model.statistical_feature_dim}")
    print(f"loss type: {config.loss.loss_type}")
    print(f"statistical loss weights: {config.loss.weights.stat_loss}")
    print(f"Full statistical block accessible: {config.statistical.statistical_features.enabled}")

    print("\nTest complete.")