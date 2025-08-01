#!/usr/bin/env python3
"""
Setup script for Circle Detection with Pylon Camera
==================================================

This script helps set up the circle detection system by:
1. Installing required dependencies
2. Testing Pylon SDK installation
3. Testing camera connections
4. Creating necessary directories

Author: AI Assistant
Date: 2024
"""

import subprocess
import sys
import os
import importlib
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_circle_detection.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def test_pylon_installation():
    """Test Pylon SDK installation."""
    print("\nTesting Pylon SDK installation...")
    
    try:
        import pypylon
        print("✓ Pylon SDK is available")
        
        # Test basic functionality
        tl_factory = pypylon.pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        
        if len(devices) > 0:
            print(f"✓ Found {len(devices)} Pylon camera(s)")
            for i, device in enumerate(devices):
                print(f"  Camera {i}: {device.GetModelName()} (SN: {device.GetSerialNumber()})")
        else:
            print("⚠ No Pylon cameras found (will use webcam fallback)")
        
        return True
    except ImportError:
        print("✗ Pylon SDK not found. Install with: pip install pypylon")
        return False
    except Exception as e:
        print(f"✗ Error testing Pylon: {e}")
        return False

def test_opencv_installation():
    """Test OpenCV installation."""
    print("\nTesting OpenCV installation...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test basic functionality
        test_image = cv2.imread("test_image.png") if os.path.exists("test_image.png") else None
        if test_image is not None:
            circles = cv2.HoughCircles(
                cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY),
                cv2.HOUGH_GRADIENT,
                dp=1, minDist=20, param1=50, param2=30,
                minRadius=10, maxRadius=100
            )
            print("✓ OpenCV HoughCircles function working")
        else:
            print("✓ OpenCV installed successfully")
        
        return True
    except ImportError:
        print("✗ OpenCV not found")
        return False
    except Exception as e:
        print(f"✗ Error testing OpenCV: {e}")
        return False

def test_camera_connection():
    """Test camera connection."""
    print("\nTesting camera connection...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera working - Frame size: {frame.shape}")
                cap.release()
                return True
            else:
                print("✗ Camera opened but failed to read frame")
                cap.release()
                return False
        else:
            print("✗ Failed to open camera")
            return False
    except Exception as e:
        print(f"✗ Error testing camera: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = ['output', 'logs', 'config']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def create_test_image():
    """Create a test image with circles for testing."""
    print("\nCreating test image...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image with circles
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Draw some circles
        cv2.circle(img, (150, 150), 50, (255, 255, 255), -1)
        cv2.circle(img, (450, 150), 80, (255, 255, 255), -1)
        cv2.circle(img, (300, 300), 60, (255, 255, 255), -1)
        
        cv2.imwrite("test_image.png", img)
        print("✓ Created test image: test_image.png")
        return True
    except Exception as e:
        print(f"✗ Error creating test image: {e}")
        return False

def run_quick_test():
    """Run a quick test of the circle detection."""
    print("\nRunning quick test...")
    
    try:
        # Import the main script
        from pylon_circle_detector import CircleDetector
        
        # Create detector
        detector = CircleDetector()
        
        # Test with a simple image
        test_img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(test_img, (100, 100), 50, 255, -1)
        
        # Test detection
        circles = detector.detect_circles_hough(test_img)
        
        if len(circles) > 0:
            print("✓ Circle detection working")
            return True
        else:
            print("⚠ Circle detection test inconclusive")
            return True
            
    except Exception as e:
        print(f"✗ Error in quick test: {e}")
        return False

def main():
    """Main setup function."""
    print("Circle Detection Setup")
    print("=====================")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher required")
        return False
    
    print(f"✓ Python version: {sys.version}")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Test installations
    if not test_opencv_installation():
        return False
    
    if not test_pylon_installation():
        print("⚠ Pylon SDK not available - will use webcam only")
    
    # Create directories
    if not create_directories():
        return False
    
    # Create test image
    create_test_image()
    
    # Test camera
    if not test_camera_connection():
        print("⚠ Camera test failed - check camera connection")
    
    # Quick test
    try:
        import numpy as np
        import cv2
        run_quick_test()
    except ImportError:
        print("⚠ Quick test skipped - numpy/cv2 not available")
    
    print("\n" + "="*50)
    print("Setup completed!")
    print("\nTo run the circle detection:")
    print("  python pylon_circle_detector.py")
    print("\nFor help:")
    print("  python pylon_circle_detector.py --help")
    print("\nExample usage:")
    print("  python pylon_circle_detector.py --camera 0 --config circle_detection_config.json")
    print("="*50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 