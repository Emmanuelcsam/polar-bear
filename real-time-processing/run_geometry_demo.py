#!/usr/bin/env python3
"""
Cross-platform launcher for the Geometry Detection Demo application.
Works on both Windows and Linux (including Wayland).
"""

import sys
import os
import subprocess
import platform

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['cv2', 'numpy', 'matplotlib']
    optional_packages = ['pypylon', 'GPUtil', 'psutil']
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'matplotlib':
                import matplotlib
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            if package == 'pypylon':
                from pypylon import pylon
            elif package == 'GPUtil':
                import GPUtil
            elif package == 'psutil':
                import psutil
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"Missing required packages: {', '.join(missing_required)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"Optional packages not installed: {', '.join(missing_optional)}")
        print("Some features may be unavailable.")
    
    return True

def main():
    """Main entry point."""
    if not check_dependencies():
        sys.exit(1)
    
    # Import and run the demo application
    try:
        from src.applications.example_application import main as run_demo
        
        print("Starting Geometry Detection Demo...")
        print("This will open a window showing real-time shape detection.")
        print("Press 'q' to quit")
        print("-" * 50)
        
        run_demo()
        
    except ImportError as e:
        print(f"Error importing module: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()