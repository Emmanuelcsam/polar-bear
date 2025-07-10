#!/usr/bin/env python3
"""
Modular Configuration Functions
==============================
Standalone configuration loading, validation, and management functions
for fiber inspection applications.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config_from_file(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Dictionary containing the loaded configuration
    """
    config_path_obj = Path(config_path)
    
    if not config_path_obj.exists():
        logger.warning(f"Configuration file '{config_path}' not found.")
        return get_default_config()
    
    try:
        with open(config_path_obj, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        logger.info(f"Configuration loaded successfully from '{config_path}'")
        return config_data
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading configuration from '{config_path}': {e}")
        return get_default_config()

def save_config_to_file(config: Dict[str, Any], config_path: str = "config.json") -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where the configuration will be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config_path_obj = Path(config_path)
        config_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path_obj, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to '{config_path}'")
        return True
    except IOError as e:
        logger.error(f"Error saving configuration to '{config_path}': {e}")
        return False

def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration structure.
    
    Returns:
        Dictionary containing default configuration
    """
    return {
        "general_settings": {
            "output_dir_name": "output",
            "log_level": "INFO",
            "log_to_console": True
        },
        "processing_profiles": {
            "fast_scan": {
                "description": "Quick scan with minimal algorithms",
                "preprocessing": {
                    "clahe_clip_limit": 1.0,
                    "clahe_tile_grid_size": [4, 4],
                    "gaussian_blur_kernel_size": [3, 3]
                },
                "localization": {
                    "hough_dp": 1.5,
                    "hough_min_dist_factor": 0.2,
                    "hough_param1": 50,
                    "hough_param2": 25,
                    "hough_min_radius_factor": 0.1,
                    "hough_max_radius_factor": 0.4
                },
                "defect_detection": {
                    "region_algorithms": ["black_hat"],
                    "linear_algorithms": ["lei_simple"],
                    "confidence_threshold": 0.6,
                    "min_defect_area_px": 15
                }
            },
            "deep_inspection": {
                "description": "Thorough scan with full algorithm suite",
                "preprocessing": {
                    "clahe_clip_limit": 2.0,
                    "clahe_tile_grid_size": [8, 8],
                    "gaussian_blur_kernel_size": [5, 5]
                },
                "localization": {
                    "hough_dp": 1.2,
                    "hough_min_dist_factor": 0.15,
                    "hough_param1": 70,
                    "hough_param2": 35,
                    "hough_min_radius_factor": 0.08,
                    "hough_max_radius_factor": 0.45
                },
                "defect_detection": {
                    "region_algorithms": ["morph_gradient", "black_hat"],
                    "linear_algorithms": ["lei_advanced", "skeletonization"],
                    "confidence_threshold": 0.9,
                    "min_defect_area_px": 5,
                    "scratch_aspect_ratio_threshold": 3.0,
                    "algorithm_weights": {
                        "morph_gradient": 0.4,
                        "black_hat": 0.6,
                        "lei_advanced": 0.7,
                        "skeletonization": 0.3
                    }
                }
            }
        },
        "algorithm_parameters": {
            "flat_field_image_path": None,
            "morph_gradient_kernel_size": [5, 5],
            "black_hat_kernel_size": [11, 11],
            "lei_kernel_lengths": [11, 17],
            "lei_angle_step_deg": 15,
            "sobel_scharr_ksize": 3,
            "skeletonization_dilation_kernel_size": [3, 3]
        },
        "zone_definitions_iec61300_3_35": {
            "single_mode_pc": [
                {
                    "name": "Core",
                    "r_min_factor": 0.0,
                    "r_max_factor_core_relative": 1.0,
                    "color_bgr": [255, 0, 0],
                    "pass_fail_rules": {
                        "max_scratches": 0,
                        "max_defects": 0,
                        "max_defect_size_um": 3
                    }
                },
                {
                    "name": "Cladding",
                    "r_min_factor_cladding_relative": 0.0,
                    "r_max_factor_cladding_relative": 1.0,
                    "color_bgr": [0, 255, 0],
                    "pass_fail_rules": {
                        "max_scratches": 5,
                        "max_scratches_gt_5um": 0,
                        "max_defects": 5,
                        "max_defect_size_um": 10
                    }
                },
                {
                    "name": "Adhesive",
                    "r_min_factor_cladding_relative": 1.0,
                    "r_max_factor_cladding_relative": 1.15,
                    "color_bgr": [0, 255, 255],
                    "pass_fail_rules": {
                        "max_defects": "unlimited",
                        "max_defect_size_um": 50
                    }
                },
                {
                    "name": "Contact",
                    "r_min_factor_cladding_relative": 1.15,
                    "r_max_factor_cladding_relative": 2.0,
                    "color_bgr": [255, 0, 255],
                    "pass_fail_rules": {
                        "max_defects": "unlimited",
                        "max_defect_size_um": 100
                    }
                }
            ],
            "multi_mode_pc": [
                {
                    "name": "Core",
                    "r_min_factor": 0.0,
                    "r_max_factor_core_relative": 1.0,
                    "color_bgr": [255, 100, 100],
                    "pass_fail_rules": {
                        "max_scratches": 1,
                        "max_scratch_length_um": 10,
                        "max_defects": 3,
                        "max_defect_size_um": 5
                    }
                },
                {
                    "name": "Cladding",
                    "r_min_factor_cladding_relative": 0.0,
                    "r_max_factor_cladding_relative": 1.0,
                    "color_bgr": [100, 255, 100],
                    "pass_fail_rules": {
                        "max_scratches": "unlimited",
                        "max_defects": "unlimited",
                        "max_defect_size_um": 20
                    }
                },
                {
                    "name": "Adhesive",
                    "r_min_factor_cladding_relative": 1.0,
                    "r_max_factor_cladding_relative": 1.15,
                    "color_bgr": [100, 255, 255],
                    "pass_fail_rules": {
                        "max_defects": "unlimited"
                    }
                },
                {
                    "name": "Contact",
                    "r_min_factor_cladding_relative": 1.15,
                    "r_max_factor_cladding_relative": 2.0,
                    "color_bgr": [255, 100, 255],
                    "pass_fail_rules": {
                        "max_defects": "unlimited"
                    }
                }
            ]
        },
        "reporting": {
            "annotated_image_dpi": 150,
            "defect_label_font_scale": 0.4,
            "defect_label_thickness": 1,
            "pass_fail_stamp_font_scale": 1.5,
            "pass_fail_stamp_thickness": 2,
            "zone_outline_thickness": 2,
            "defect_outline_thickness": 2,
            "display_timestamp_on_image": True,
            "timestamp_format": "%Y-%m-%d %H:%M:%S"
        }
    }

def validate_config_structure(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate configuration structure and return validation results.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Dictionary with 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []
    
    # Required top-level sections
    required_sections = [
        "general_settings",
        "processing_profiles",
        "algorithm_parameters",
        "zone_definitions_iec61300_3_35",
        "reporting"
    ]
    
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate processing profiles
    if "processing_profiles" in config:
        profiles = config["processing_profiles"]
        if not profiles:
            warnings.append("No processing profiles defined")
        
        for profile_name, profile_data in profiles.items():
            if not isinstance(profile_data, dict):
                errors.append(f"Profile '{profile_name}' must be a dictionary")
                continue
            
            required_profile_sections = ["preprocessing", "localization", "defect_detection"]
            for section in required_profile_sections:
                if section not in profile_data:
                    errors.append(f"Profile '{profile_name}' missing section: {section}")
    
    # Validate zone definitions
    if "zone_definitions_iec61300_3_35" in config:
        zone_defs = config["zone_definitions_iec61300_3_35"]
        if not zone_defs:
            warnings.append("No zone definitions found")
        
        for fiber_type, zones in zone_defs.items():
            if not isinstance(zones, list):
                errors.append(f"Zone definition for '{fiber_type}' must be a list")
                continue
            
            for zone in zones:
                if "name" not in zone:
                    errors.append(f"Zone in '{fiber_type}' missing 'name' field")
                if "color_bgr" not in zone:
                    warnings.append(f"Zone '{zone.get('name', 'unknown')}' in '{fiber_type}' missing color")
    
    return {"errors": errors, "warnings": warnings}

def get_config_value(config: Dict[str, Any], key_path: str, default_value: Any = None) -> Any:
    """
    Get a configuration value using dot notation (e.g., 'general_settings.log_level').
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the configuration value
        default_value: Default value if key path not found
        
    Returns:
        Configuration value or default_value
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default_value

def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> bool:
    """
    Set a configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the configuration value
        value: Value to set
        
    Returns:
        True if successful, False otherwise
    """
    keys = key_path.split('.')
    current = config
    
    try:
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
        return True
    except (TypeError, KeyError) as e:
        logger.error(f"Error setting config value '{key_path}': {e}")
        return False

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to merge/override with
        
    Returns:
        Merged configuration dictionary
    """
    def _merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    return _merge_dicts(base_config, override_config)

def create_profile_config(base_profile: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new processing profile based on an existing one with overrides.
    
    Args:
        base_profile: Name of the base profile to copy from
        overrides: Dictionary of values to override
        
    Returns:
        New profile configuration
    """
    default_config = get_default_config()
    base_profile_config = default_config["processing_profiles"].get(base_profile, {})
    
    if not base_profile_config:
        logger.warning(f"Base profile '{base_profile}' not found, using fast_scan")
        base_profile_config = default_config["processing_profiles"]["fast_scan"]
    
    return merge_configs(base_profile_config, overrides)

# Test function
def test_config_functions():
    """Test the configuration functions."""
    logger.info("Testing configuration functions...")
    
    # Test default config
    default_config = get_default_config()
    logger.info(f"Default config has {len(default_config)} sections")
    
    # Test validation
    validation = validate_config_structure(default_config)
    logger.info(f"Validation errors: {len(validation['errors'])}, warnings: {len(validation['warnings'])}")
    
    # Test get/set config values
    log_level = get_config_value(default_config, "general_settings.log_level", "DEBUG")
    logger.info(f"Log level: {log_level}")
    
    set_config_value(default_config, "general_settings.test_value", "test")
    test_value = get_config_value(default_config, "general_settings.test_value")
    logger.info(f"Test value: {test_value}")
    
    # Test profile creation
    custom_profile = create_profile_config("fast_scan", {
        "preprocessing": {"clahe_clip_limit": 2.5},
        "description": "Custom fast scan"
    })
    logger.info(f"Custom profile CLAHE limit: {custom_profile['preprocessing']['clahe_clip_limit']}")
    
    # Test file operations
    test_config_path = "test_config.json"
    if save_config_to_file(default_config, test_config_path):
        loaded_config = load_config_from_file(test_config_path)
        logger.info(f"Loaded config matches: {len(loaded_config) == len(default_config)}")
        
        # Clean up
        try:
            Path(test_config_path).unlink()
        except:
            pass
    
    logger.info("Configuration function tests completed")

if __name__ == "__main__":
    test_config_functions()
