#!/usr/bin/env python3
"""
Test script for EmulatedPylonGrabber.
"""

import cv2
import numpy as np
import time
import sys
import os

# Add the current directory to path
sys.path.append('.')

def test_emulated_grabber():
    """Test the EmulatedPylonGrabber functionality."""
    print("Testing EmulatedPylonGrabber...")
    
    try:
        from bmp_video_emulator import EmulatedPylonGrabber
        
        # Test image path
        image_path = "pictures/good.bmp"
        
        if not os.path.exists(image_path):
            print(f"Error: Test image not found: {image_path}")
            return
        
        print(f"Using test image: {image_path}")
        
        # Create grabber
        grabber = EmulatedPylonGrabber(
            use_emulation=True,
            image_path=image_path,
            frame_rate=10
        )
        
        print("Starting grabber...")
        grabber.start()
        
        # Wait a bit for grabber to start
        time.sleep(1)
        
        # Test reading frames
        print("Testing frame reading...")
        for i in range(5):
            frame = grabber.read()
            if frame is not None:
                print(f"Frame {i+1}: {frame.shape}")
            else:
                print(f"Frame {i+1}: None")
            time.sleep(0.5)
        
        print("Stopping grabber...")
        grabber.stop()
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_emulated_grabber() 