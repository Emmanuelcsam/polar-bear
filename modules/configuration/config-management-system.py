#!/usr/bin/env python3
"""
Configuration Management System Module
=====================================
Advanced configuration loading, validation, and management system
with support for multiple profiles and dynamic parameter adjustment.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import copy

class ConfigurationManager:
    """
    Advanced configuration management system with validation and profiles.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = Path(config_path) if config_path else Path("config.json")
        self.config_data = {}
        self.default_config = self._get_default_configuration()
        self.load_configuration()
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration structure."""
        return {
            "general_settings": {
                "output_dir_name": "output",
                "log_level": "INFO",
                "log_to_console": True,
                "supported_image_extensions": [".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
                "max_processing_threads": 4,
                "enable_gpu_acceleration": False
            },
            "processing_profiles": {
                "fast_scan": {
                    "description": "Quick scan with minimal processing",
                    "preprocessing": {
                        "clahe_clip_limit": 1.5,
                        "clahe_tile_grid_size": [4, 4],
                        "gaussian_blur_kernel_size": [3, 3],
                        "enable_illumination_correction": False
                    },
                    "localization": {
                        "hough_dp": 1.5,
                        "hough_min_dist_factor": 0.2,
                        "hough_param1": 50,
                        "hough_param2": 25,
                        "hough_min_radius_factor": 0.1,
                        "hough_max_radius_factor": 0.4,
                        "enable_contour_fallback": True,
                        "enable_circle_fit_fallback": False
                    },
                    "defect_detection": {
                        "algorithms": ["do2mr"],
                        "algorithm_weights": {"do2mr": 1.0},
                        "multi_scale_detection": False,
                        "min_defect_area_px": 10,
                        "confidence_threshold": 0.7
                    }
                },
                "deep_inspection": {
                    "description": "Comprehensive inspection with all algorithms",
                    "preprocessing": {
                        "clahe_clip_limit": 2.0,
                        "clahe_tile_grid_size": [8, 8],
                        "gaussian_blur_kernel_size": [5, 5],
                        "enable_illumination_correction": True,
                        "illumination_method": "rolling_ball"
                    },
                    "localization": {
                        "hough_dp": 1.2,
                        "hough_min_dist_factor": 0.15,
                        "hough_param1": 70,
                        "hough_param2": 35,
                        "hough_min_radius_factor": 0.08,
                        "hough_max_radius_factor": 0.45,
                        "enable_contour_fallback": True,
                        "enable_circle_fit_fallback": True
                    },
                    "defect_detection": {
                        "algorithms": ["do2mr", "lei", "matrix_variance"],
                        "algorithm_weights": {
                            "do2mr": 1.0,
                            "lei": 0.8,
                            "matrix_variance": 0.6
                        },
                        "multi_scale_detection": True,
                        "min_defect_area_px": 3,
                        "confidence_threshold": 0.5,
                        "enable_validation": True
                    }
                }
            },
            "algorithm_parameters": {
                "do2mr": {
                    "kernel_size": 5,
                    "gamma_core": 1.2,
                    "gamma_cladding": 1.5,
                    "multi_scale_kernels": [3, 5, 7, 9]
                },
                "lei": {
                    "kernel_lengths": [7, 11, 15, 21, 31],
                    "angle_step_deg": 10,
                    "multi_scale_enabled": True,
                    "scales": [0.75, 1.0, 1.25]
                },
                "matrix_variance": {
                    "variance_threshold": 15.0,
                    "local_window_size": 3,
                    "segment_grid": [3, 3]
                }
            },
            "fiber_type_definitions": {
                "single_mode_pc": {
                    "description": "Single-mode PC connector",
                    "typical_core_diameter_um": 9.0,
                    "typical_cladding_diameter_um": 125.0,
                    "pass_fail_rules": {
                        "Core": {
                            "max_scratches": 0,
                            "max_defects": 0,
                            "max_defect_size_um": 3,
                            "critical_zone": True
                        },
                        "Cladding": {
                            "max_scratches": 5,
                            "max_scratches_gt_5um": 0,
                            "max_defects": 5,
                            "max_defect_size_um": 10,
                            "critical_zone": False
                        }
                    }
                },
                "multi_mode_pc": {
                    "description": "Multi-mode PC connector",
                    "typical_core_diameter_um": 50.0,
                    "typical_cladding_diameter_um": 125.0,
                    "pass_fail_rules": {
                        "Core": {
                            "max_scratches": 1,
                            "max_scratch_length_um": 10,
                            "max_defects": 3,
                            "max_defect_size_um": 5,
                            "critical_zone": False
                        },
                        "Cladding": {
                            "max_scratches": "unlimited",
                            "max_defects": "unlimited",
                            "max_defect_size_um": 20,
                            "critical_zone": False
                        }
                    }
                }
            },
            "calibration": {
                "default_um_per_px": 0.5,
                "auto_calibration_enabled": True,
                "calibration_method": "cladding_diameter"
            },
            "reporting": {
                "generate_annotated_images": True,
                "generate_csv_reports": True,
                "generate_polar_histograms": True,
                "image_dpi": 150,
                "font_scale": 0.5,
                "line_thickness": 2
            }
        }
    
    def load_configuration(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Optional path to configuration file
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            logging.warning(f"Configuration file '{self.config_path}' not found. Creating default.")
            self.create_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            
            # Validate and merge with defaults
            self.config_data = self._merge_with_defaults(self.config_data, self.default_config)
            
            logging.info(f"Configuration loaded from '{self.config_path}'")
            
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to load configuration: {e}")
            logging.warning("Using default configuration")
            self.config_data = copy.deepcopy(self.default_config)
    
    def save_configuration(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Optional path to save configuration
        """
        save_path = Path(config_path) if config_path else self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Configuration saved to '{save_path}'")
            
        except IOError as e:
            logging.error(f"Failed to save configuration: {e}")
    
    def create_default_config(self) -> None:
        """Create default configuration file."""
        self.config_data = copy.deepcopy(self.default_config)
        self.save_configuration()
    
    def _merge_with_defaults(self, user_config: Dict[str, Any], 
                           default_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge user configuration with defaults.
        
        Args:
            user_config: User-provided configuration
            default_config: Default configuration
            
        Returns:
            Merged configuration
        """
        merged = copy.deepcopy(default_config)
        
        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_with_defaults(value, merged[key])
            else:
                merged[key] = value
        
        return merged
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return copy.deepcopy(self.config_data)
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            section_name: Name of the configuration section
            
        Returns:
            Configuration section dictionary
        """
        return copy.deepcopy(self.config_data.get(section_name, {}))
    
    def get_processing_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific processing profile.
        
        Args:
            profile_name: Name of the processing profile
            
        Returns:
            Profile configuration or None if not found
        """
        profiles = self.config_data.get("processing_profiles", {})
        return copy.deepcopy(profiles.get(profile_name))
    
    def get_fiber_type_definition(self, fiber_type: str) -> Optional[Dict[str, Any]]:
        """
        Get fiber type definition.
        
        Args:
            fiber_type: Fiber type identifier
            
        Returns:
            Fiber type definition or None if not found
        """
        fiber_types = self.config_data.get("fiber_type_definitions", {})
        return copy.deepcopy(fiber_types.get(fiber_type))
    
    def get_algorithm_parameters(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get algorithm-specific parameters.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Algorithm parameters dictionary
        """
        algo_params = self.config_data.get("algorithm_parameters", {})
        return copy.deepcopy(algo_params.get(algorithm_name, {}))
    
    def update_parameter(self, parameter_path: str, value: Any) -> None:
        """
        Update a specific parameter using dot notation.
        
        Args:
            parameter_path: Dot-separated path to parameter (e.g., "general_settings.log_level")
            value: New value for the parameter
        """
        keys = parameter_path.split('.')
        current = self.config_data
        
        # Navigate to the parent of the target parameter
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        logging.info(f"Updated parameter '{parameter_path}' to '{value}'")
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required sections
        required_sections = ["general_settings", "processing_profiles", "fiber_type_definitions"]
        for section in required_sections:
            if section not in self.config_data:
                errors.append(f"Missing required section: {section}")
        
        # Validate processing profiles
        profiles = self.config_data.get("processing_profiles", {})
        if not profiles:
            errors.append("No processing profiles defined")
        
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict):
                errors.append(f"Profile '{profile_name}' is not a dictionary")
                continue
            
            # Check required profile sections
            required_profile_sections = ["preprocessing", "localization", "defect_detection"]
            for section in required_profile_sections:
                if section not in profile:
                    errors.append(f"Profile '{profile_name}' missing section: {section}")
        
        # Validate fiber type definitions
        fiber_types = self.config_data.get("fiber_type_definitions", {})
        if not fiber_types:
            errors.append("No fiber type definitions found")
        
        for fiber_type, definition in fiber_types.items():
            if "pass_fail_rules" not in definition:
                errors.append(f"Fiber type '{fiber_type}' missing pass/fail rules")
        
        return errors
    
    def create_custom_profile(self, profile_name: str, 
                            base_profile: str = "deep_inspection",
                            modifications: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a custom processing profile based on an existing one.
        
        Args:
            profile_name: Name for the new profile
            base_profile: Base profile to copy from
            modifications: Dictionary of modifications to apply
        """
        profiles = self.config_data.get("processing_profiles", {})
        
        if base_profile not in profiles:
            raise ValueError(f"Base profile '{base_profile}' not found")
        
        # Copy base profile
        new_profile = copy.deepcopy(profiles[base_profile])
        
        # Apply modifications
        if modifications:
            new_profile = self._merge_with_defaults(modifications, new_profile)
        
        # Add description
        new_profile["description"] = f"Custom profile based on {base_profile}"
        
        # Save new profile
        self.config_data["processing_profiles"][profile_name] = new_profile
        logging.info(f"Created custom profile '{profile_name}' based on '{base_profile}'")
    
    def export_profile(self, profile_name: str, export_path: Union[str, Path]) -> None:
        """
        Export a processing profile to a separate file.
        
        Args:
            profile_name: Name of the profile to export
            export_path: Path to save the exported profile
        """
        profile = self.get_processing_profile(profile_name)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        export_data = {
            "profile_name": profile_name,
            "profile_data": profile,
            "exported_from": str(self.config_path)
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Profile '{profile_name}' exported to '{export_path}'")
    
    def import_profile(self, import_path: Union[str, Path], 
                      profile_name: Optional[str] = None) -> None:
        """
        Import a processing profile from a file.
        
        Args:
            import_path: Path to the profile file
            profile_name: Optional new name for the profile
        """
        with open(import_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        original_name = import_data.get("profile_name", "imported_profile")
        profile_data = import_data.get("profile_data", {})
        
        final_name = profile_name or original_name
        
        self.config_data["processing_profiles"][final_name] = profile_data
        logging.info(f"Profile imported as '{final_name}' from '{import_path}'")

# Convenience functions for standalone use
_global_config_manager = None

def get_global_config_manager(config_path: Optional[Union[str, Path]] = None) -> ConfigurationManager:
    """Get or create the global configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None or config_path is not None:
        _global_config_manager = ConfigurationManager(config_path)
    return _global_config_manager

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file (convenience function)."""
    manager = get_global_config_manager(config_path)
    return manager.get_config()

def get_processing_profile(profile_name: str) -> Optional[Dict[str, Any]]:
    """Get processing profile (convenience function)."""
    manager = get_global_config_manager()
    return manager.get_processing_profile(profile_name)

def get_fiber_type_definition(fiber_type: str) -> Optional[Dict[str, Any]]:
    """Get fiber type definition (convenience function)."""
    manager = get_global_config_manager()
    return manager.get_fiber_type_definition(fiber_type)

if __name__ == "__main__":
    """Test the configuration management system"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    print("Testing Configuration Management System...")
    
    # Test basic functionality
    config_manager = ConfigurationManager("test_config.json")
    
    # Test getting configuration
    config = config_manager.get_config()
    print(f"Loaded configuration with {len(config)} main sections")
    
    # Test getting specific sections
    general = config_manager.get_section("general_settings")
    print(f"General settings: {list(general.keys())}")
    
    # Test processing profiles
    profile = config_manager.get_processing_profile("deep_inspection")
    if profile:
        print(f"Deep inspection profile has {len(profile)} sections")
    
    # Test fiber type definitions
    fiber_def = config_manager.get_fiber_type_definition("single_mode_pc")
    if fiber_def:
        print(f"Single-mode PC definition: {list(fiber_def.keys())}")
    
    # Test parameter update
    config_manager.update_parameter("general_settings.log_level", "DEBUG")
    updated_level = config_manager.get_section("general_settings")["log_level"]
    print(f"Updated log level to: {updated_level}")
    
    # Test validation
    errors = config_manager.validate_configuration()
    print(f"Configuration validation: {len(errors)} errors found")
    
    # Test custom profile creation
    config_manager.create_custom_profile(
        "custom_fast", 
        "fast_scan", 
        {"defect_detection": {"confidence_threshold": 0.8}}
    )
    print("Created custom profile")
    
    # Test convenience functions
    global_config = load_config("test_config.json")
    print(f"Global config loaded: {len(global_config)} sections")
    
    # Clean up test file
    test_path = Path("test_config.json")
    if test_path.exists():
        test_path.unlink()
        print("Cleaned up test configuration file")
    
    print("Configuration management system tests completed!")
