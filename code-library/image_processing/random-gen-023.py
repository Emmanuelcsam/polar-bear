#!/usr/bin/env python3
"""
Cross-platform launcher for the Calibration Tool.
Works on both Windows and Linux (including Wayland).
"""

import sys
import os
import platform

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['cv2', 'numpy', 'tkinter']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'tkinter':
                import tkinter
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        if 'tkinter' in missing:
            print("\nFor tkinter:")
            print("  Ubuntu/Debian: sudo apt-get install python3-tk")
            print("  Windows: tkinter should be included with Python")
        print("\nFor other packages: pip install -r requirements.txt")
        return False
    return True

def main():
    """Main entry point."""
    if not check_dependencies():
        sys.exit(1)
    
    # Special handling for Wayland
    if platform.system() == "Linux":
        # Check if running on Wayland
        wayland_display = os.environ.get('WAYLAND_DISPLAY')
        if wayland_display:
            print("Detected Wayland display system.")
            # Set backend to X11 for compatibility
            os.environ['GDK_BACKEND'] = 'x11'
    
    # Import and run the calibration tool
    try:
        from src.tools.realtime_calibration_tool import main as run_calibration
        
        print("Starting Real-time Calibration Tool...")
        print("This will open a GUI for calibrating the geometry detection system.")
        print("-" * 50)
        
        run_calibration()
        
    except ImportError as e:
        print(f"Error importing module: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()