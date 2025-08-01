#!/usr/bin/env python3
"""
Test script for the comprehensive configuration system
"""

from config_manager import ConfigManager


def test_config_manager():
    """Test the configuration manager"""
    print("Testing Comprehensive Configuration Manager")
    print("=" * 50)
    
    # Create config manager
    config_manager = ConfigManager("config.json")
    
    # Test getting different sections
    print("\n1. Testing configuration sections:")
    sections = [
        "circle_overlay",
        "auto_core_detection", 
        "live_feed",
        "main",
        "pytorch",
        "global"
    ]
    
    for section in sections:
        config = config_manager.get_section(section)
        print(f"  {section}: {len(config)} subsections")
    
    # Test getting specific values
    print("\n2. Testing specific configuration values:")
    
    # Circle overlay config
    circle_config = config_manager.get_circle_overlay_config()
    move_step = circle_config.get("movement", {}).get("move_step", "N/A")
    print(f"  Circle overlay move step: {move_step}")
    
    # Auto core detection config
    detection_config = config_manager.get_auto_core_detection_config()
    min_confidence = detection_config.get("detection", {}).get("min_confidence", "N/A")
    print(f"  Detection min confidence: {min_confidence}")
    
    # Live feed config
    live_config = config_manager.get_live_feed_config()
    camera_index = live_config.get("camera", {}).get("camera_index", "N/A")
    print(f"  Camera index: {camera_index}")
    
    # Test validation
    print("\n3. Testing configuration validation:")
    errors = config_manager.validate_config()
    if errors:
        print("  Validation errors found:")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  Configuration validation passed!")
    
    # Test setting values
    print("\n4. Testing configuration updates:")
    config_manager.set_value("test_section", "test_key", "test_value")
    test_value = config_manager.get_value("test_section", "test_key", "not_found")
    print(f"  Test value: {test_value}")
    
    # Print configuration summary
    print("\n5. Configuration Summary:")
    config_manager.print_config("circle_overlay")
    
    print("\nConfiguration system test completed successfully!")


if __name__ == "__main__":
    test_config_manager() 