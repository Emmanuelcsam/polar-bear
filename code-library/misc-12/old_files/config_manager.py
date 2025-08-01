#!/usr/bin/env python3
"""
Enhanced Configuration Manager with Basler Camera Support
Provides comprehensive configuration management for all system components
with specific optimizations for Basler a2A2590-22gmBAS camera.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Enhanced configuration manager with Basler camera support"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_default_config()
        self._load_user_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration with Basler camera optimizations"""
        return {
            "camera": {
                "basler": {
                    "target_model": "a2A2590-22gmBAS",
                    "target_serial": "40455566",
                    "pixel_format": "RGB8",
                    "exposure_auto": "Continuous",
                    "gain_auto": "Continuous",
                    "acquisition_mode": "Continuous",
                    "trigger_mode": "Off",
                    "auto_exposure_time_min": 1000,
                    "auto_exposure_time_max": 1000000,
                    "auto_gain_min": 0,
                    "auto_gain_max": 24,
                    "enable_auto_white_balance": True,
                    "enable_auto_gain": True,
                    "enable_auto_exposure": True
                },
                "pylon": {
                    "enable_pylon_detection": True,
                    "enable_auto_detection": True,
                    "preferred_backend": "Pylon",
                    "fallback_to_opencv": True,
                    "timeout_ms": 1000,
                    "grab_strategy": "LatestImageOnly"
                },
                "opencv": {
                    "enable_opencv_detection": True,
                    "backends": ["CAP_ANY", "CAP_DSHOW", "CAP_MSMF", "CAP_FFMPEG"],
                    "max_camera_index": 10,
                    "timeout_ms": 1000
                },
                "general": {
                    "auto_detect": True,
                    "demo_mode": False,
                    "camera_index": 0,
                    "use_pylon": True,
                    "enable_fallback": True
                }
            },
            "detection": {
                "process_interval": 0.2,  # Backward compatibility
                "auto_core_detection": {
                    "enable_geometric_detection": True,
                    "enable_improved_detection": True,
                    "enable_manual_learning": True,
                    "min_confidence": 0.3,
                    "max_confidence": 1.0,
                    "detection_timeout": 0.2,
                    "enable_parallel_detection": True,
                    "max_detection_workers": 4
                },
                "hough_circles": {
                    "dp": 2.0,
                    "min_dist": 150,
                    "param1": 50,
                    "param2": 25,
                    "min_radius_small": 5,
                    "max_radius_small": 50,
                    "min_radius_medium": 15,
                    "max_radius_medium": 150,
                    "min_radius_large": 50,
                    "max_radius_large": 500,
                    "enable_adaptive_parameters": True,
                    "adaptive_scale_factor": 0.1
                },
                "preprocessing": {
                    "enable_clahe": True,
                    "clahe_clip_limit": 2.0,
                    "clahe_tile_grid_size": 8,
                    "enable_gaussian_blur": True,
                    "gaussian_kernel_size": 7,
                    "gaussian_sigma": 1.5,
                    "enable_median_blur": False,
                    "median_kernel_size": 5,
                    "enable_bilateral_filter": False,
                    "bilateral_d": 9,
                    "bilateral_sigma_color": 75,
                    "bilateral_sigma_space": 75
                }
            },
            "circle_overlay": {
                "movement": {
                    "move_step": 8,
                    "resize_step": 5,
                    "enable_continuous_movement": False,
                    "enable_parallel_worker": False,
                    "key_repeat_rate": 0.008,
                },
                "circle": {
                    "initial_center_x": 320,
                    "initial_center_y": 240,
                    "initial_radius": 50,
                    "min_radius": 1,
                    "max_radius": 999999,
                    "color_red": 255,
                    "color_green": 0,
                    "color_blue": 0,
                    "thickness": 2,
                    "center_point_size": 3,
                },
                "keyboard": {
                    "enable_key_repeat": False,
                    "enable_simultaneous_keys": False,
                },
                "performance": {
                    "enable_performance_tracking": True,
                    "frame_time_history_size": 30,
                }
            },
            "pytorch": {
                "learning": {
                    "learning_rate": 0.001,
                    "save_interval": 10,
                    "enable_auto_save": True,
                    "model_path": "core_detection_model.pth",
                    "data_path": "detection_data.pkl"
                },
                "network": {
                    "hidden_layers": [128, 64, 32],
                    "dropout_rate": 0.3,
                    "activation_function": "relu"
                },
                "feature_extraction": {
                    "intensity_profile_size": 64,
                    "enable_texture_analysis": True,
                    "enable_gradient_analysis": True
                }
            },
            "performance": {
                "enable_gpu_acceleration": True,
                "enable_parallel_processing": True,
                "max_workers": 4,
                "frame_buffer_size": 10,
                "enable_performance_monitoring": True
            },
            "display": {
                "show_info_overlay": True,
                "show_detection_results": True,
                "show_performance_stats": True,
                "window_name": "Core Detection System",
                "enable_fullscreen": False
            },
            "logging": {
                "enable_logging": True,
                "log_level": "INFO",
                "log_file": "system.log",
                "enable_console_output": True
            }
        }
    
    def _load_user_config(self):
        """Load user configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Merge user config with default config
                self._merge_config(self.config, user_config)
                print(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
    
    def _merge_config(self, default_config: Dict, user_config: Dict):
        """Recursively merge user configuration with default configuration"""
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_config(default_config[key], value)
            else:
                default_config[key] = value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return self.config.get("camera", {})
    
    def get_basler_config(self) -> Dict[str, Any]:
        """Get Basler camera specific configuration"""
        return self.config.get("camera", {}).get("basler", {})
    
    def get_pylon_config(self) -> Dict[str, Any]:
        """Get Pylon camera configuration"""
        return self.config.get("camera", {}).get("pylon", {})
    
    def get_opencv_config(self) -> Dict[str, Any]:
        """Get OpenCV camera configuration"""
        return self.config.get("camera", {}).get("opencv", {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration"""
        return self.config.get("detection", {})
    
    def get_circle_overlay_config(self) -> Dict[str, Any]:
        """Get circle overlay configuration"""
        return self.config.get("circle_overlay", {})
    
    def get_pytorch_config(self) -> Dict[str, Any]:
        """Get PyTorch configuration"""
        return self.config.get("pytorch", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config.get("performance", {})
    
    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration"""
        return self.config.get("display", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get("logging", {})
    
    def get_live_feed_config(self) -> Dict[str, Any]:
        """Get live feed configuration"""
        return self.config.get("camera", {})
    
    def get_auto_core_detection_config(self) -> Dict[str, Any]:
        """Get auto core detection configuration"""
        return self.config.get("detection", {}).get("auto_core_detection", {})
    
    def update_config(self, section: str, key: str, value: Any):
        """Update a specific configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def get_basler_target_info(self) -> Dict[str, str]:
        """Get target Basler camera information"""
        basler_config = self.get_basler_config()
        return {
            "model": basler_config.get("target_model", "a2A2590-22gmBAS"),
            "serial": basler_config.get("target_serial", "40455566")
        }
    
    def is_basler_camera_enabled(self) -> bool:
        """Check if Basler camera detection is enabled"""
        camera_config = self.get_camera_config()
        return (camera_config.get("general", {}).get("use_pylon", True) and
                camera_config.get("pylon", {}).get("enable_pylon_detection", True))
    
    def get_camera_detection_methods(self) -> list:
        """Get list of camera detection methods in order of preference"""
        methods = []
        camera_config = self.get_camera_config()
        
        if camera_config.get("pylon", {}).get("enable_pylon_detection", True):
            methods.append("pylon")
        
        if camera_config.get("opencv", {}).get("enable_opencv_detection", True):
            methods.append("opencv")
        
        return methods
    
    def create_basler_optimized_config(self):
        """Create a configuration optimized for Basler camera"""
        basler_config = {
            "camera": {
                "basler": {
                    "target_model": "a2A2590-22gmBAS",
                    "target_serial": "40455566",
                    "pixel_format": "RGB8",
                    "exposure_auto": "Continuous",
                    "gain_auto": "Continuous",
                    "acquisition_mode": "Continuous",
                    "trigger_mode": "Off",
                    "auto_exposure_time_min": 1000,
                    "auto_exposure_time_max": 1000000,
                    "auto_gain_min": 0,
                    "auto_gain_max": 24,
                    "enable_auto_white_balance": True,
                    "enable_auto_gain": True,
                    "enable_auto_exposure": True
                },
                "pylon": {
                    "enable_pylon_detection": True,
                    "enable_auto_detection": True,
                    "preferred_backend": "Pylon",
                    "fallback_to_opencv": True,
                    "timeout_ms": 1000,
                    "grab_strategy": "LatestImageOnly"
                },
                "general": {
                    "auto_detect": True,
                    "demo_mode": False,
                    "camera_index": 0,
                    "use_pylon": True,
                    "enable_fallback": True
                }
            }
        }
        
        # Update current config with Basler optimizations
        self._merge_config(self.config, basler_config)
        self.save_config()
        print("Created Basler-optimized configuration")
    
    def validate_config(self) -> bool:
        """Validate configuration for errors"""
        try:
            # Check required sections
            required_sections = ["camera", "detection", "performance"]
            for section in required_sections:
                if section not in self.config:
                    print(f"Missing required configuration section: {section}")
                    return False
            
            # Check camera configuration
            camera_config = self.get_camera_config()
            if not camera_config:
                print("Camera configuration is empty")
                return False
            
            # Check Basler configuration
            basler_config = self.get_basler_config()
            if not basler_config:
                print("Basler camera configuration is missing")
                return False
            
            print("Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False


def main():
    """Test configuration manager"""
    print("Enhanced Configuration Manager Test")
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Test configuration sections
    print(f"Camera config: {len(config_manager.get_camera_config())} sections")
    print(f"Basler config: {len(config_manager.get_basler_config())} settings")
    print(f"Detection config: {len(config_manager.get_detection_config())} sections")
    
    # Test Basler target info
    target_info = config_manager.get_basler_target_info()
    print(f"Target Basler: {target_info['model']} (Serial: {target_info['serial']})")
    
    # Test detection methods
    methods = config_manager.get_camera_detection_methods()
    print(f"Detection methods: {methods}")
    
    # Validate configuration
    is_valid = config_manager.validate_config()
    print(f"Configuration valid: {is_valid}")
    
    # Create Basler-optimized config
    config_manager.create_basler_optimized_config()


if __name__ == "__main__":
    main() 