#!/usr/bin/env python3
"""
Cross-platform launcher for the Real-time Circle Detector application.
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
    required_packages = ['cv2', 'numpy', 'pypylon']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'pypylon':
                try:
                    from pypylon import pylon
                except ImportError:
                    print("Warning: pypylon not installed. Basler camera support will be unavailable.")
                    print("The application will work with webcam/USB cameras.")
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True

def main():
    """Main entry point."""
    if not check_dependencies():
        sys.exit(1)
    
    # Import and run the circle detector
    try:
        from src.applications.realtime_circle_detector import CircleDetector
        
        print("Starting Real-time Circle Detector...")
        print("Press 'q' to quit, 'p' to pause/resume, 's' to save frame")
        print("-" * 50)
        
        detector = CircleDetector()
        detector.run()
        
    except ImportError as e:
        print(f"Error importing module: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()