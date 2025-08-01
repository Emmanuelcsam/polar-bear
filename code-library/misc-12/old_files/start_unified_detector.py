#!/usr/bin/env python3
"""
Unified Core Detector Startup Script
Launches the complete core detection system with manual overlay and automatic detection
"""

import sys
import subprocess
import webbrowser
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
    
    # Check Flask (for web interface)
    try:
        import flask
        print(f"✓ Flask {flask.__version__}")
    except ImportError:
        print("⚠ Flask not installed (web interface not available)")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n=== INSTALLING DEPENDENCIES ===")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "opencv-python", "numpy", "flask"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main startup function"""
    print("=========================================")
    print("   Unified Core Detection System")
    print("=========================================")
    
    # Check dependencies
    if not check_dependencies():
        print("\nAttempting to install dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            print("Please install manually: pip install opencv-python numpy flask")
            return 1
    
    # Import and run unified detector
    try:
        from unified_core_detector import main as run_detector
        print("\n=== STARTING UNIFIED CORE DETECTOR ===")
        run_detector()
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 