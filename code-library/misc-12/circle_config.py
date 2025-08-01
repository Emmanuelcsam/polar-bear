#!/usr/bin/env python3
"""
Circle Overlay Configuration Script
Allows manual control over every parameter and numerical value of the circle overlay.
"""

import json
import os
from typing import Dict, Any


class CircleConfig:
    """Comprehensive configuration for circle overlay with full manual control"""
    
    def __init__(self, config_file: str = "circle_config.json"):
        self.config_file = config_file
        self.default_config = {
            # Movement Control
            "movement": {
                "move_step": 8,                    # Pixels per movement
                "resize_step": 5,                  # Pixels per resize
                "enable_continuous_movement": False,  # Disable auto-movement
                "enable_parallel_worker": False,   # Disable parallel processing
                "key_repeat_rate": 0.008,          # 8ms for 120Hz response
                "movement_smoothing": 1.0,         # Movement smoothing factor
                "max_speed_multiplier": 3.0,       # Maximum speed multiplier
            },
            
            # Keyboard Control
            "keyboard": {
                "enable_key_repeat": False,        # Disable automatic key repeat
                "key_repeat_delay": 0.5,           # Delay before key repeat starts
                "key_repeat_interval": 0.1,        # Interval between key repeats
                "enable_simultaneous_keys": False, # Disable multiple key handling
                "key_debounce_time": 0.05,        # Key debounce time
            },
            
            # Circle Properties
            "circle": {
                "initial_center_x": 320,           # Initial X position
                "initial_center_y": 240,           # Initial Y position
                "initial_radius": 50,              # Initial radius
                "min_radius": 1,                   # Minimum radius
                "max_radius": 999999,              # Maximum radius
                "color_red": 255,                  # Red component (0-255)
                "color_green": 0,                  # Green component (0-255)
                "color_blue": 0,                   # Blue component (0-255)
                "thickness": 2,                    # Circle line thickness
                "center_point_size": 3,            # Center point size
            },
            
            # Performance Settings
            "performance": {
                "enable_performance_tracking": True,
                "frame_time_history_size": 30,     # Number of frame times to track
                "target_fps": 60,                  # Target FPS
                "enable_vsync": False,             # Enable vertical sync
                "enable_frame_limiting": False,    # Disable frame rate limiting
            },
            
            # Boundary Control
            "boundaries": {
                "enable_boundary_restrictions": False,  # Disable boundary limits
                "max_x": 10000,                   # Maximum X position
                "max_y": 10000,                   # Maximum Y position
                "min_x": -10000,                  # Minimum X position
                "min_y": -10000,                  # Minimum Y position
                "boundary_buffer": 100,           # Buffer beyond frame
            },
            
            # Display Settings
            "display": {
                "show_lock_indicator": True,       # Show lock status
                "show_performance_stats": True,    # Show FPS and stats
                "show_circle_info": True,          # Show circle position/radius
                "info_overlay_height": 60,         # Height of info overlay
                "font_scale": 0.4,                # Text font scale
                "text_color_red": 255,            # Text color red
                "text_color_green": 255,          # Text color green
                "text_color_blue": 255,           # Text color blue
            },
            
            # Advanced Control
            "advanced": {
                "enable_mouse_control": False,     # Enable mouse control
                "enable_touch_control": False,     # Enable touch control
                "enable_gesture_control": False,   # Enable gesture control
                "enable_voice_control": False,     # Enable voice control
                "enable_ai_assistance": False,     # Enable AI assistance
                "enable_macro_recording": False,   # Enable macro recording
                "enable_script_execution": False,  # Enable script execution
            },
            
            # Debug Settings
            "debug": {
                "enable_debug_logging": True,      # Enable debug output
                "log_level": "INFO",              # Log level (DEBUG, INFO, WARNING, ERROR)
                "enable_performance_monitoring": True,
                "enable_error_tracking": True,
                "enable_memory_monitoring": False,
            }
        }
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(self.default_config, loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.default_config.copy()
        else:
            # Create default config file
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Merge loaded config with defaults"""
        result = default.copy()
        for section, values in loaded.items():
            if section in result:
                result[section].update(values)
            else:
                result[section] = values
        return result
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_movement_config(self) -> Dict[str, Any]:
        """Get movement configuration"""
        return self.config["movement"]
    
    def get_keyboard_config(self) -> Dict[str, Any]:
        """Get keyboard configuration"""
        return self.config["keyboard"]
    
    def get_circle_config(self) -> Dict[str, Any]:
        """Get circle properties configuration"""
        return self.config["circle"]
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config["performance"]
    
    def get_boundary_config(self) -> Dict[str, Any]:
        """Get boundary configuration"""
        return self.config["boundaries"]
    
    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration"""
        return self.config["display"]
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Get advanced configuration"""
        return self.config["advanced"]
    
    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration"""
        return self.config["debug"]
    
    def update_config(self, section: str, key: str, value: Any):
        """Update a specific configuration value"""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            self.save_config()
            print(f"Updated {section}.{key} = {value}")
        else:
            print(f"Invalid configuration: {section}.{key}")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.default_config.copy()
        self.save_config()
        print("Configuration reset to defaults")
    
    def print_current_config(self):
        """Print current configuration"""
        print("\n=== Current Circle Overlay Configuration ===")
        for section, values in self.config.items():
            print(f"\n[{section.upper()}]")
            for key, value in values.items():
                print(f"  {key}: {value}")
    
    def create_custom_config(self):
        """Interactive configuration creation"""
        print("\n=== Circle Overlay Configuration Editor ===")
        print("This will guide you through setting up your custom configuration.")
        
        new_config = self.default_config.copy()
        
        # Movement settings
        print("\n--- Movement Settings ---")
        new_config["movement"]["move_step"] = int(input("Move step (pixels): ") or 8)
        new_config["movement"]["resize_step"] = int(input("Resize step (pixels): ") or 5)
        new_config["movement"]["enable_continuous_movement"] = input("Enable continuous movement (y/n): ").lower() == 'y'
        new_config["movement"]["enable_parallel_worker"] = input("Enable parallel worker (y/n): ").lower() == 'y'
        new_config["movement"]["key_repeat_rate"] = float(input("Key repeat rate (seconds): ") or 0.008)
        
        # Circle properties
        print("\n--- Circle Properties ---")
        new_config["circle"]["initial_center_x"] = int(input("Initial X position: ") or 320)
        new_config["circle"]["initial_center_y"] = int(input("Initial Y position: ") or 240)
        new_config["circle"]["initial_radius"] = int(input("Initial radius: ") or 50)
        new_config["circle"]["min_radius"] = int(input("Minimum radius: ") or 1)
        # Max radius is now unlimited (999999)
        new_config["circle"]["max_radius"] = 999999
        
        # Performance settings
        print("\n--- Performance Settings ---")
        new_config["performance"]["target_fps"] = int(input("Target FPS: ") or 60)
        new_config["performance"]["enable_frame_limiting"] = input("Enable frame limiting (y/n): ").lower() == 'y'
        
        # Save the custom configuration
        self.config = new_config
        self.save_config()
        print("\nCustom configuration saved!")
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        errors = []
        
        # Validate movement settings
        move_step = self.config["movement"]["move_step"]
        if move_step <= 0 or move_step > 100:
            errors.append("move_step must be between 1 and 100")
        
        # Validate circle settings
        radius = self.config["circle"]["initial_radius"]
        min_radius = self.config["circle"]["min_radius"]
        
        # Only check minimum radius to prevent negative values
        if radius < min_radius:
            errors.append("initial_radius must be at least min_radius")
        
        # Validate color values
        for color in ["color_red", "color_green", "color_blue"]:
            value = self.config["circle"][color]
            if value < 0 or value > 255:
                errors.append(f"{color} must be between 0 and 255")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation passed!")
        return True


def main():
    """Main configuration interface"""
    config = CircleConfig()
    
    while True:
        print("\n=== Circle Overlay Configuration ===")
        print("1. View current configuration")
        print("2. Edit specific setting")
        print("3. Create custom configuration")
        print("4. Reset to defaults")
        print("5. Validate configuration")
        print("6. Save and exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            config.print_current_config()
        
        elif choice == "2":
            print("\nAvailable sections:")
            sections = list(config.config.keys())
            for i, section in enumerate(sections, 1):
                print(f"{i}. {section}")
            
            try:
                section_idx = int(input("Select section (1-{}): ".format(len(sections)))) - 1
                if 0 <= section_idx < len(sections):
                    section = sections[section_idx]
                    print(f"\nSettings in {section}:")
                    for key, value in config.config[section].items():
                        print(f"  {key}: {value}")
                    
                    key = input("Enter setting name to edit: ").strip()
                    if key in config.config[section]:
                        new_value = input(f"Enter new value for {key}: ").strip()
                        # Try to convert to appropriate type
                        try:
                            if isinstance(config.config[section][key], bool):
                                new_value = new_value.lower() == 'true'
                            elif isinstance(config.config[section][key], int):
                                new_value = int(new_value)
                            elif isinstance(config.config[section][key], float):
                                new_value = float(new_value)
                            config.update_config(section, key, new_value)
                        except ValueError:
                            print("Invalid value type!")
                    else:
                        print("Invalid setting name!")
                else:
                    print("Invalid section number!")
            except ValueError:
                print("Invalid input!")
        
        elif choice == "3":
            config.create_custom_config()
        
        elif choice == "4":
            confirm = input("Are you sure you want to reset to defaults? (y/n): ")
            if confirm.lower() == 'y':
                config.reset_to_defaults()
        
        elif choice == "5":
            config.validate_config()
        
        elif choice == "6":
            if config.validate_config():
                print("Configuration saved successfully!")
                break
            else:
                print("Please fix configuration errors before exiting.")
        
        else:
            print("Invalid option!")


if __name__ == "__main__":
    main() 