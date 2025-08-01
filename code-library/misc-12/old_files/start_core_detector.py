#!/usr/bin/env python3
"""
Simple Core Detector Startup Script
Launches the core detection system with proper error handling
"""

import sys
import os
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are available"""
    print("=== CHECKING DEPENDENCIES ===")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    
    # Check NumPy
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    # Check Pylon (optional)
    try:
        from pypylon import pylon
        print("✓ Pylon SDK available")
    except ImportError:
        print("⚠ Pylon SDK not available - will use OpenCV fallback")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("=== INSTALLING DEPENDENCIES ===")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "opencv-python", "numpy"])
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main startup function"""
    print("=========================================")
    print("   Core Detection System Startup")
    print("=========================================")
    
    # Check dependencies
    if not check_dependencies():
        print("\nAttempting to install dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            print("Please install manually: pip install opencv-python numpy")
            return 1
    
    # Import and run core detector
    try:
        from core_detector import main as run_detector
        print("\n=== STARTING CORE DETECTOR ===")
        run_detector()
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1) 