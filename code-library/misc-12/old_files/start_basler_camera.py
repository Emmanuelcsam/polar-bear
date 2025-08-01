#!/usr/bin/env python3
"""
Basler Camera Startup Script
Specialized script to ensure Basler a2A2590-22gmBAS camera is detected and working.
"""

import sys
import os
import time
import subprocess
from typing import Optional, Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_pylon_installation():
    """Check if Pylon SDK is properly installed"""
    print("=== CHECKING PYLON INSTALLATION ===")
    
    try:
        import pypylon
        print("✓ Pylon SDK is installed")
        
        # Check Pylon version
        try:
            pylon_version = pypylon.pylon.GetPylonVersion()
            print(f"✓ Pylon version: {pylon_version}")
        except:
            print("⚠ Could not determine Pylon version")
        
        return True
    except ImportError:
        print("✗ Pylon SDK is not installed")
        print("  Install with: pip install pypylon")
        return False
    except Exception as e:
        print(f"✗ Error checking Pylon: {e}")
        return False


def check_basler_drivers():
    """Check if Basler drivers are installed"""
    print("\n=== CHECKING BASLER DRIVERS ===")
    
    # Check for common Basler driver locations
    driver_paths = [
        r"C:\Program Files\Basler\pylon 6.3\Runtime\x64\PylonRuntime_MD_VC141_v6_3_0.dll",
        r"C:\Program Files\Basler\pylon 6.2\Runtime\x64\PylonRuntime_MD_VC141_v6_2_0.dll",
        r"C:\Program Files\Basler\pylon 6.1\Runtime\x64\PylonRuntime_MD_VC141_v6_1_0.dll",
        r"C:\Program Files\Basler\pylon 6.0\Runtime\x64\PylonRuntime_MD_VC141_v6_0_0.dll"
    ]
    
    for path in driver_paths:
        if os.path.exists(path):
            print(f"✓ Basler driver found: {path}")
            return True
    
    print("⚠ Basler drivers not found in common locations")
    print("  Please ensure Basler Pylon SDK is installed")
    return False


def detect_basler_camera():
    """Detect Basler a2A2590-22gmBAS camera specifically"""
    print("\n=== DETECTING BASLER CAMERA ===")
    
    try:
        from pypylon import pylon
        
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        
        if len(devices) == 0:
            print("✗ No Pylon devices found")
            return None
        
        print(f"Found {len(devices)} Pylon device(s):")
        
        target_camera = None
        for i, device in enumerate(devices):
            model_name = device.GetModelName()
            serial_number = device.GetSerialNumber()
            vendor_name = device.GetVendorName()
            
            print(f"  Device {i}: {model_name} (Serial: {serial_number})")
            
            # Check if this is our target Basler camera
            if ('a2a2590' in model_name.lower() or 
                'basler' in model_name.lower() or
                '40455566' in serial_number):
                target_camera = {
                    'index': i,
                    'model': model_name,
                    'serial': serial_number,
                    'vendor': vendor_name,
                    'device': device
                }
                print(f"    *** TARGET BASLER CAMERA FOUND ***")
        
        if target_camera:
            print(f"\n✓ Target Basler camera found:")
            print(f"  Model: {target_camera['model']}")
            print(f"  Serial: {target_camera['serial']}")
            print(f"  Vendor: {target_camera['vendor']}")
            return target_camera
        else:
            print("\n✗ Target Basler camera not found")
            return None
            
    except Exception as e:
        print(f"✗ Error detecting Basler camera: {e}")
        return None


def test_basler_connection(camera_info: Dict):
    """Test connection to Basler camera"""
    print("\n=== TESTING BASLER CONNECTION ===")
    
    try:
        from pypylon import pylon
        
        tl_factory = pylon.TlFactory.GetInstance()
        
        # Create camera from device
        camera = pylon.InstantCamera(
            tl_factory.CreateDevice(camera_info['device'])
        )
        
        print("Attempting to open camera...")
        camera.Open()
        
        if camera.IsOpen():
            print("✓ Camera opened successfully")
            
            # Test camera settings
            try:
                camera.PixelFormat.SetValue("RGB8")
                print("✓ Set pixel format to RGB8")
            except Exception as e:
                print(f"⚠ Could not set RGB8 format: {e}")
            
            try:
                camera.ExposureAuto.SetValue("Continuous")
                print("✓ Set exposure to auto continuous")
            except Exception as e:
                print(f"⚠ Could not set auto exposure: {e}")
            
            try:
                camera.GainAuto.SetValue("Continuous")
                print("✓ Set gain to auto continuous")
            except Exception as e:
                print(f"⚠ Could not set auto gain: {e}")
            
            try:
                camera.AcquisitionMode.SetValue("Continuous")
                print("✓ Set acquisition mode to continuous")
            except Exception as e:
                print(f"⚠ Could not set acquisition mode: {e}")
            
            # Test frame capture
            print("Testing frame capture...")
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            time.sleep(0.1)  # Wait for camera to start
            
            grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab_result.GrabSucceeded():
                image = grab_result.Array
                print(f"✓ Frame captured successfully: {image.shape}")
                print(f"  Frame size: {image.shape[1]}x{image.shape[0]}")
                print(f"  Data type: {image.dtype}")
                
                # Test multiple frames
                frame_count = 0
                for i in range(5):
                    grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
                    if grab_result.GrabSucceeded():
                        frame_count += 1
                
                print(f"✓ Captured {frame_count}/5 test frames successfully")
                
            else:
                print("✗ Failed to capture frame")
            
            camera.StopGrabbing()
            camera.Close()
            print("✓ Camera test completed successfully")
            return True
        else:
            print("✗ Failed to open camera")
            return False
            
    except Exception as e:
        print(f"✗ Error testing camera connection: {e}")
        return False


def check_system_requirements():
    """Check system requirements for Basler camera"""
    print("\n=== CHECKING SYSTEM REQUIREMENTS ===")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major >= 3 and python_version.minor >= 7:
        print("✓ Python version is compatible")
    else:
        print("⚠ Python version may be too old for Pylon SDK")
    
    # Check for required packages
    required_packages = ['cv2', 'numpy', 'pypylon']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✓ OpenCV version: {cv2.__version__}")
            elif package == 'numpy':
                import numpy
                print(f"✓ NumPy version: {numpy.__version__}")
            elif package == 'pypylon':
                import pypylon
                print("✓ Pypylon is available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is not installed")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All required packages are installed")
    return True


def create_basler_config():
    """Create optimized configuration for Basler camera"""
    print("\n=== CREATING BASLER CONFIGURATION ===")
    
    try:
        from config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config_manager.create_basler_optimized_config()
        
        print("✓ Basler-optimized configuration created")
        return True
    except Exception as e:
        print(f"✗ Error creating Basler configuration: {e}")
        return False


def run_camera_test():
    """Run comprehensive camera test"""
    print("\n=== RUNNING CAMERA TEST ===")
    
    try:
        from live_feed import LiveFeed
        
        print("Initializing LiveFeed with Basler camera...")
        live_feed = LiveFeed(
            camera_index=0,
            use_pylon=True,
            auto_detect=True,
            demo_mode=False
        )
        
        if live_feed.camera is not None:
            print("✓ LiveFeed initialized successfully")
            
            # Test frame reading
            frame = live_feed.read_frame()
            if frame is not None:
                print(f"✓ Successfully read frame: {frame.shape}")
                
                # Test multiple frames
                frame_count = 0
                for i in range(10):
                    frame = live_feed.read_frame()
                    if frame is not None:
                        frame_count += 1
                
                print(f"✓ Successfully read {frame_count}/10 frames")
                live_feed.cleanup()
                return True
            else:
                print("✗ Could not read frame from LiveFeed")
                live_feed.cleanup()
                return False
        else:
            print("✗ LiveFeed initialization failed")
            return False
            
    except Exception as e:
        print(f"✗ Error running camera test: {e}")
        return False


def main():
    """Main Basler camera startup procedure"""
    print("BASLER CAMERA STARTUP PROCEDURE")
    print("Target: Basler a2A2590-22gmBAS (40455566)")
    print("="*60)
    
    # Step 1: Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met")
        return False
    
    # Step 2: Check Pylon installation
    if not check_pylon_installation():
        print("\n❌ Pylon SDK not properly installed")
        return False
    
    # Step 3: Check Basler drivers
    if not check_basler_drivers():
        print("\n⚠ Basler drivers may not be properly installed")
        print("  Continuing anyway...")
    
    # Step 4: Detect Basler camera
    camera_info = detect_basler_camera()
    if not camera_info:
        print("\n❌ Basler camera not detected")
        return False
    
    # Step 5: Test camera connection
    if not test_basler_connection(camera_info):
        print("\n❌ Camera connection test failed")
        return False
    
    # Step 6: Create optimized configuration
    if not create_basler_config():
        print("\n⚠ Could not create Basler configuration")
        print("  Continuing anyway...")
    
    # Step 7: Run comprehensive test
    if not run_camera_test():
        print("\n❌ Comprehensive camera test failed")
        return False
    
    print("\n" + "="*60)
    print("✓ BASLER CAMERA STARTUP SUCCESSFUL")
    print("="*60)
    print("Your Basler a2A2590-22gmBAS camera is ready to use!")
    print("You can now run the main application.")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Basler camera startup failed")
        print("Please check the error messages above and try again.")
        sys.exit(1)
    else:
        print("\n🎉 Basler camera is ready!")
        sys.exit(0) 