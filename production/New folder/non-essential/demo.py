#!/usr/bin/env python3
"""
Demo script for BMP Video Emulator.
Shows basic usage and integration with pylon_grabber.
"""

import time
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmp_video_emulator import BMPVideoEmulator, EmulatedPylonGrabber
from pylon_grabber import PYLON_AVAILABLE


def demo_basic_emulator():
    """Demonstrate basic BMP video emulator usage."""
    print("=== Basic BMP Video Emulator Demo ===")
    
    try:
        # Create emulator
        emulator = BMPVideoEmulator("good.bmp", frame_rate=15)
        print(f"✓ Emulator created with frame rate: {emulator.frame_rate} FPS")
        
        # Start emulation
        emulator.start()
        print("✓ Emulation started")
        
        # Read some frames
        print("Reading frames...")
        for i in range(5):
            frame = emulator.read()
            if frame is not None:
                print(f"  Frame {i+1}: Shape {frame.shape}")
            time.sleep(0.1)
        
        # Get statistics
        frame_count = emulator.get_frame_count()
        print(f"✓ Total frames processed: {frame_count}")
        
        # Stop emulation
        emulator.stop()
        print("✓ Emulation stopped")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_pylon_integration():
    """Demonstrate integration with pylon_grabber."""
    print("\n=== Pylon Integration Demo ===")
    
    try:
        # Create emulated grabber
        grabber = EmulatedPylonGrabber(
            use_emulation=True,
            image_path="good.bmp",
            frame_rate=20
        )
        print("✓ Emulated PylonGrabber created")
        
        # Start grabber
        grabber.start()
        print("✓ Grabber started")
        
        # Read frames using pylon interface
        print("Reading frames through pylon interface...")
        for i in range(5):
            frame = grabber.read()
            if frame is not None:
                print(f"  Frame {i+1}: Shape {frame.shape}")
            time.sleep(0.1)
        
        # Stop grabber
        grabber.stop()
        grabber.join(timeout=1.0)
        print("✓ Grabber stopped")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_pylon_availability():
    """Show Pylon SDK availability status."""
    print("\n=== Pylon SDK Status ===")
    
    if PYLON_AVAILABLE:
        print("✓ Pylon SDK is available")
        print("  - Real camera can be used")
        print("  - Emulation is optional")
    else:
        print("⚠ Pylon SDK is not available")
        print("  - Emulation will be used automatically")
        print("  - Install pypylon for real camera support")


def demo_frame_rate_comparison():
    """Compare different frame rates."""
    print("\n=== Frame Rate Comparison Demo ===")
    
    frame_rates = [10, 30, 60]
    
    for fps in frame_rates:
        try:
            print(f"\nTesting {fps} FPS:")
            
            emulator = BMPVideoEmulator("good.bmp", frame_rate=fps)
            emulator.start()
            
            # Measure actual frame rate
            start_time = time.time()
            frame_count = 0
            
            for _ in range(10):  # Read 10 frames
                frame = emulator.read()
                if frame is not None:
                    frame_count += 1
                time.sleep(0.1)
            
            elapsed_time = time.time() - start_time
            actual_fps = frame_count / elapsed_time
            
            print(f"  Target FPS: {fps}")
            print(f"  Actual FPS: {actual_fps:.1f}")
            print(f"  Accuracy: {((actual_fps/fps)*100):.1f}%")
            
            emulator.stop()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")


def main():
    """Run all demos."""
    print("BMP Video Emulator Demo")
    print("=" * 50)
    
    # Check if good.bmp exists
    if not os.path.exists("good.bmp"):
        print("✗ Error: good.bmp not found in current directory")
        print("Please ensure good.bmp is in the project root directory")
        return
    
    # Run demos
    demo_pylon_availability()
    demo_basic_emulator()
    demo_pylon_integration()
    demo_frame_rate_comparison()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nTo run the GUI application:")
    print("  python bmp_video_emulator.py")
    print("\nTo run the test suite:")
    print("  python non-essential/test_bmp_video_emulator.py")


if __name__ == "__main__":
    main() 