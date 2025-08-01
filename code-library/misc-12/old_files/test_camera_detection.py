#!/usr/bin/env python3
"""
Comprehensive Camera Detection Test for Basler a2A2590-22gmBAS
Tests all possible camera detection methods to ensure the Basler camera is found.
"""

import cv2
import numpy as np
import time
import sys
import os
from typing import List, Dict, Optional

# Try to import Pylon
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
    print("✓ Pylon SDK available")
except ImportError:
    print("✗ Pylon SDK not available - install with: pip install pypylon")

# Try to import GPU support
GPU_AVAILABLE = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        GPU_AVAILABLE = True
        print("✓ GPU acceleration available")
    else:
        print("✗ GPU acceleration not available")
except:
    print("✗ GPU acceleration not available")


def test_pylon_detection():
    """Test Pylon camera detection specifically for Basler"""
    print("\n" + "="*60)
    print("PYLON CAMERA DETECTION TEST")
    print("="*60)
    
    if not PYLON_AVAILABLE:
        print("Pylon SDK not available - skipping Pylon tests")
        return False
    
    try:
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        
        if len(devices) == 0:
            print("No Pylon devices found")
            return False
        
        print(f"Found {len(devices)} Pylon device(s):")
        
        basler_found = False
        target_camera = None
        
        for i, device in enumerate(devices):
            model_name = device.GetModelName()
            serial_number = device.GetSerialNumber()
            vendor_name = device.GetVendorName()
            
            print(f"  Device {i}:")
            print(f"    Model: {model_name}")
            print(f"    Serial: {serial_number}")
            print(f"    Vendor: {vendor_name}")
            
            # Check if this is our target Basler camera
            if ('a2a2590' in model_name.lower() or 
                'basler' in model_name.lower() or
                '40455566' in serial_number):
                basler_found = True
                target_camera = {
                    'index': i,
                    'model': model_name,
                    'serial': serial_number,
                    'vendor': vendor_name,
                    'device': device
                }
                print(f"    *** TARGET BASLER CAMERA FOUND ***")
        
        if basler_found and target_camera:
            print(f"\n✓ Target Basler camera found: {target_camera['model']}")
            print(f"  Serial: {target_camera['serial']}")
            
            # Try to open the camera
            print("\nTesting camera connection...")
            try:
                camera = pylon.InstantCamera(
                    tl_factory.CreateDevice(target_camera['device'])
                )
                camera.Open()
                
                if camera.IsOpen():
                    print("✓ Camera opened successfully")
                    
                    # Test camera settings
                    try:
                        camera.PixelFormat.SetValue("RGB8")
                        print("✓ Set pixel format to RGB8")
                    except Exception as e:
                        print(f"✗ Could not set RGB8 format: {e}")
                    
                    try:
                        camera.ExposureAuto.SetValue("Continuous")
                        print("✓ Set exposure to auto continuous")
                    except Exception as e:
                        print(f"✗ Could not set auto exposure: {e}")
                    
                    try:
                        camera.GainAuto.SetValue("Continuous")
                        print("✓ Set gain to auto continuous")
                    except Exception as e:
                        print(f"✗ Could not set auto gain: {e}")
                    
                    # Test frame capture
                    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    time.sleep(0.1)  # Wait for camera to start
                    
                    grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
                    if grab_result.GrabSucceeded():
                        image = grab_result.Array
                        print(f"✓ Frame captured successfully: {image.shape}")
                        print(f"  Frame size: {image.shape[1]}x{image.shape[0]}")
                        print(f"  Data type: {image.dtype}")
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
                print(f"✗ Error testing camera: {e}")
                return False
        else:
            print("\n✗ Target Basler camera not found")
            return False
            
    except Exception as e:
        print(f"✗ Error in Pylon detection: {e}")
        return False


def test_opencv_detection():
    """Test OpenCV camera detection with multiple backends"""
    print("\n" + "="*60)
    print("OPENCV CAMERA DETECTION TEST")
    print("="*60)
    
    backends = [
        (cv2.CAP_ANY, "CAP_ANY"),
        (cv2.CAP_DSHOW, "CAP_DSHOW"),
        (cv2.CAP_MSMF, "CAP_MSMF"),
        (cv2.CAP_FFMPEG, "CAP_FFMPEG"),
        (cv2.CAP_GSTREAMER, "CAP_GSTREAMER")
    ]
    
    cameras_found = []
    
    for backend, backend_name in backends:
        print(f"\nTesting backend: {backend_name}")
        
        for i in range(5):  # Test indices 0-4
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"  ✓ Camera {i}: {frame.shape[1]}x{frame.shape[0]}")
                        cameras_found.append({
                            'index': i,
                            'backend': backend_name,
                            'shape': frame.shape
                        })
                        cap.release()
                        break  # Found a working camera with this backend
                    else:
                        print(f"  ✗ Camera {i}: opened but no frame")
                        cap.release()
                else:
                    print(f"  - Camera {i}: not available")
            except Exception as e:
                print(f"  ✗ Camera {i}: error - {e}")
    
    if cameras_found:
        print(f"\n✓ Found {len(cameras_found)} working camera(s):")
        for cam in cameras_found:
            print(f"  - Index {cam['index']} with {cam['backend']}: {cam['shape'][1]}x{cam['shape'][0]}")
        return True
    else:
        print("\n✗ No cameras found with OpenCV")
        return False


def test_system_info():
    """Test system information and dependencies"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    if PYLON_AVAILABLE:
        try:
            # Get Pylon version
            pylon_version = pylon.GetPylonVersion()
            print(f"Pylon version: {pylon_version}")
        except:
            print("Pylon version: unknown")
    
    # Check for GPU
    if GPU_AVAILABLE:
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"CUDA devices: {device_count}")
            for i in range(device_count):
                device = cv2.cuda.getDevice(i)
                print(f"  Device {i}: {device.name()}")
        except Exception as e:
            print(f"GPU info error: {e}")
    
    # Check for common camera-related processes
    import psutil
    camera_processes = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if any(keyword in proc.info['name'].lower() for keyword in 
                   ['camera', 'pylon', 'basler', 'viewer']):
                camera_processes.append(proc.info)
        except:
            pass
    
    if camera_processes:
        print(f"\nCamera-related processes found: {len(camera_processes)}")
        for proc in camera_processes:
            print(f"  - {proc['name']} (PID: {proc['pid']})")
    else:
        print("\nNo camera-related processes found")


def test_camera_permissions():
    """Test camera access permissions"""
    print("\n" + "="*60)
    print("CAMERA PERMISSIONS TEST")
    print("="*60)
    
    # Test basic camera access
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Basic camera access works")
            else:
                print("✗ Camera opened but no frame captured")
            cap.release()
        else:
            print("✗ Cannot open camera - permission issue possible")
    except Exception as e:
        print(f"✗ Camera access error: {e}")
    
    # Check if running as administrator (Windows)
    if os.name == 'nt':
        try:
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            if is_admin:
                print("✓ Running as administrator")
            else:
                print("⚠ Not running as administrator - may cause camera issues")
        except:
            print("⚠ Could not check administrator status")


def main():
    """Run comprehensive camera detection tests"""
    print("BASLER CAMERA DETECTION TEST")
    print("Target: Basler a2A2590-22gmBAS (40455566)")
    print("="*60)
    
    # Test system information
    test_system_info()
    
    # Test camera permissions
    test_camera_permissions()
    
    # Test Pylon detection (primary method for Basler)
    pylon_success = test_pylon_detection()
    
    # Test OpenCV detection (fallback)
    opencv_success = test_opencv_detection()
    
    # Summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    
    if pylon_success:
        print("✓ Basler camera detected and working via Pylon")
        print("  The camera should work with the main application")
    elif opencv_success:
        print("⚠ Basler camera not found via Pylon, but other cameras available")
        print("  The application will use fallback cameras")
    else:
        print("✗ No cameras detected")
        print("  Please check camera connections and drivers")
    
    print("\nRECOMMENDATIONS:")
    if not PYLON_AVAILABLE:
        print("1. Install Pylon SDK: pip install pypylon")
    if not pylon_success and PYLON_AVAILABLE:
        print("2. Check Basler camera drivers and Pylon installation")
        print("3. Ensure camera is not being used by other applications")
    if not opencv_success:
        print("4. Check webcam drivers and connections")
    
    print("="*60)


if __name__ == "__main__":
    main() 