#!/usr/bin/env python3
"""
Pylon SDK Setup Script for Fiber Optic End-Face CNN
This script helps install and configure Pylon SDK for realtime video processing.
"""

import os
import sys
import subprocess
import platform

def check_pylon_installation():
    """Check if Pylon SDK is properly installed."""
    try:
        import pypylon
        print("✓ Pylon SDK is installed")
        return True
    except ImportError:
        print("✗ Pylon SDK is not installed")
        return False

def install_pylon():
    """Install Pylon SDK using pip."""
    print("Installing Pylon SDK...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypylon"])
        print("✓ Pylon SDK installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install Pylon SDK: {e}")
        return False

def check_camera_devices():
    """Check for available camera devices."""
    try:
        from pypylon import pylon
        
        # Get the transport layer factory
        tl_factory = pylon.TlFactory.GetInstance()
        
        # Get all attached devices
        devices = tl_factory.EnumerateDevices()
        
        if len(devices) == 0:
            print("No Pylon cameras found")
            return False
        
        print(f"Found {len(devices)} Pylon camera(s):")
        for i, device in enumerate(devices):
            print(f"  {i+1}. {device.GetModelName()} (Serial: {device.GetSerialNumber()})")
        
        return True
        
    except Exception as e:
        print(f"Error checking camera devices: {e}")
        return False

def test_camera_connection():
    """Test camera connection and basic functionality."""
    try:
        from pypylon import pylon
        
        # Get the transport layer factory
        tl_factory = pylon.TlFactory.GetInstance()
        
        # Get all attached devices
        devices = tl_factory.EnumerateDevices()
        
        if len(devices) == 0:
            print("No cameras available for testing")
            return False
        
        # Use the first available camera
        camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
        
        # Open camera
        camera.Open()
        
        if camera.IsOpen():
            print(f"✓ Successfully connected to camera: {camera.GetDeviceInfo().GetModelName()}")
            
            # Test basic camera settings
            try:
                # Get current pixel format
                pixel_format = camera.PixelFormat.GetValue()
                print(f"  Current pixel format: {pixel_format}")
                
                # Get current exposure time
                exposure_time = camera.ExposureTime.GetValue()
                print(f"  Current exposure time: {exposure_time} μs")
                
                # Get current gain
                gain = camera.Gain.GetValue()
                print(f"  Current gain: {gain}")
                
                camera.Close()
                return True
                
            except Exception as e:
                print(f"  Warning: Could not read camera settings: {e}")
                camera.Close()
                return True
        else:
            print("✗ Failed to open camera")
            return False
            
    except Exception as e:
        print(f"Error testing camera connection: {e}")
        return False

def create_config_file():
    """Create a configuration file for camera settings."""
    config = {
        "camera_settings": {
            "pixel_format": "RGB8",
            "exposure_time": 10000,  # 10ms
            "gain": 0,
            "acquisition_mode": "Continuous",
            "trigger_mode": "Off"
        },
        "processing_settings": {
            "device": "cuda",
            "model_path": "checkpoints/best_model.pth",
            "num_classes": 40,
            "defect_threshold": 0.5
        },
        "visualization_settings": {
            "window_size": [15, 10],
            "update_rate": 30,  # FPS
            "save_frames": True
        }
    }
    
    import json
    with open("pylon_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created pylon_config.json configuration file")

def main():
    """Main setup function."""
    print("Pylon SDK Setup for Fiber Optic End-Face CNN")
    print("=" * 50)
    
    # Check if Pylon is installed
    if not check_pylon_installation():
        print("\nInstalling Pylon SDK...")
        if not install_pylon():
            print("\nManual installation required:")
            print("1. Download Pylon SDK from: https://www.baslerweb.com/en/sales-support/downloads/software-downloads/pylon-6-3-0/")
            print("2. Install the SDK for your platform")
            print("3. Run: pip install pypylon")
            return
    
    print("\nChecking camera devices...")
    check_camera_devices()
    
    print("\nTesting camera connection...")
    test_camera_connection()
    
    print("\nCreating configuration file...")
    create_config_file()
    
    print("\nSetup complete!")
    print("\nTo run the realtime visualizer:")
    print("python src/realtime_visualizer.py --weights checkpoints/best_model.pth")
    print("\nFor help:")
    print("python src/realtime_visualizer.py --help")

if __name__ == "__main__":
    main() 