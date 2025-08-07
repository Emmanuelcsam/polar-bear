#!/usr/bin/env python3
"""
Simple launcher for BMP Video Emulator GUI.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bmp_video_emulator import main
    
    print("Starting BMP Video Emulator...")
    print("Make sure 'good.bmp' is in the current directory.")
    print("Press Ctrl+C to exit.")
    
    main()
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required dependencies are installed:")
    print("  pip install opencv-python numpy")
    
except Exception as e:
    print(f"Error starting emulator: {e}")
    print("Please check that 'good.bmp' exists in the current directory.") 