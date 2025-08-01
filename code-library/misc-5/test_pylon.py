#!/usr/bin/env python3
"""
Simple Pylon Camera Test Script
Tests basic camera functionality and video capture.
"""

import sys
import time
import cv2
import numpy as np

# Add the src directory to the path
sys.path.append('src')

try:
    from pypylon import pylon
    print("✓ Pylon SDK imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Pylon SDK: {e}")
    sys.exit(1)

def test_camera_detection():
    """Test camera detection."""
    print("\n=== Testing Camera Detection ===")
    
    try:
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
        print(f"Error detecting cameras: {e}")
        return False

def test_camera_connection():
    """Test camera connection and basic settings."""
    print("\n=== Testing Camera Connection ===")
    
    try:
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
                
                # Test setting pixel format to RGB8
                try:
                    camera.PixelFormat.SetValue("RGB8")
                    print("  ✓ Successfully set pixel format to RGB8")
                except Exception as e:
                    print(f"  ⚠ Could not set RGB8 format: {e}")
                    print(f"  Using current format: {camera.PixelFormat.GetValue()}")
                
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

def test_video_capture():
    """Test video capture functionality."""
    print("\n=== Testing Video Capture ===")
    
    try:
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
            print(f"Testing video capture from: {camera.GetDeviceInfo().GetModelName()}")
            
            # Configure camera for video capture
            try:
                # Set pixel format to RGB8 if possible
                try:
                    camera.PixelFormat.SetValue("RGB8")
                    print("  Set pixel format to RGB8")
                except Exception as e:
                    print(f"  Using current format: {camera.PixelFormat.GetValue()}")
                
                # Set exposure time
                camera.ExposureTime.SetValue(10000)  # 10ms
                
                # Set gain
                camera.Gain.SetValue(0)
                
                # Enable continuous acquisition
                camera.AcquisitionMode.SetValue("Continuous")
                
                # Set trigger mode to software
                camera.TriggerMode.SetValue("Off")
                
                print("  Camera configured for video capture")
                
                # Test capturing a few frames
                print("  Capturing 5 test frames...")
                
                # Start grabbing
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                
                for i in range(5):
                    # Grab one image
                    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    
                    if grab_result.GrabSucceeded():
                        # Convert to numpy array
                        image = grab_result.Array
                        print(f"    Frame {i+1}: {image.shape} - {image.dtype}")
                        
                        # Save first frame as test
                        if i == 0:
                            # Convert monochrome to BGR for saving
                            if len(image.shape) == 2:  # Monochrome
                                save_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                            else:  # RGB
                                save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite("test_frame.png", save_image)
                            print("    ✓ Saved test frame as 'test_frame.png'")
                        
                        grab_result.Release()
                    else:
                        print(f"    ✗ Failed to grab frame {i+1}")
                
                # Stop grabbing
                camera.StopGrabbing()
                
                camera.Close()
                print("  ✓ Video capture test completed successfully")
                return True
                
            except Exception as e:
                print(f"  Error during video capture: {e}")
                camera.Close()
                return False
        else:
            print("✗ Failed to open camera for video capture")
            return False
            
    except Exception as e:
        print(f"Error testing video capture: {e}")
        return False

def main():
    """Main test function."""
    print("Pylon Camera Test Script")
    print("=" * 30)
    
    # Test camera detection
    if not test_camera_detection():
        print("\n✗ Camera detection failed")
        return
    
    # Test camera connection
    if not test_camera_connection():
        print("\n✗ Camera connection failed")
        return
    
    # Test video capture
    if not test_video_capture():
        print("\n✗ Video capture failed")
        return
    
    print("\n" + "=" * 30)
    print("✓ All tests passed!")
    print("\nYour Pylon camera is ready for use with the realtime visualizer.")
    print("\nTo run the realtime visualizer:")
    print("python src/realtime_visualizer.py --weights checkpoints/best_model.pth")

if __name__ == "__main__":
    main() 