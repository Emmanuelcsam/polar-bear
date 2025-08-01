#!/usr/bin/env python3
"""
Test script to verify GUI fixes and Pylon Viewer integration
"""

import cv2
import numpy as np
import time
import sys
import os

def test_opencv_gui():
    """Test if OpenCV GUI functions work"""
    print("Testing OpenCV GUI functionality...")
    
    try:
        # Create a test image
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(test_image, "GUI Test", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Try to show the image
        cv2.imshow("OpenCV GUI Test", test_image)
        print("✓ OpenCV GUI is working")
        
        # Wait a bit and close
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"✗ OpenCV GUI error: {e}")
        print("  This is expected on systems without GUI support")
        return False

def test_headless_mode():
    """Test headless mode functionality"""
    print("\nTesting headless mode...")
    
    try:
        # Create a test image
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(test_image, "Headless Test", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Process image without displaying
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                                  dp=1, minDist=50, 
                                  param1=50, param2=30, 
                                  minRadius=10, maxRadius=100)
        
        if circles is not None:
            print("✓ Headless processing works")
        else:
            print("✓ Headless processing works (no circles found)")
        
        return True
        
    except Exception as e:
        print(f"✗ Headless mode error: {e}")
        return False

def test_pylon_integration():
    """Test Pylon integration"""
    print("\nTesting Pylon integration...")
    
    try:
        # Try to import pylon viewer integration
        from pylon_viewer_integration import PylonViewerManager
        
        manager = PylonViewerManager(auto_start=False)
        
        # Check if Pylon SDK is available
        if manager.is_pylon_available():
            print("✓ Pylon SDK is available")
        else:
            print("⚠ Pylon SDK not available (this is normal if not installed)")
        
        # Test finding Pylon Viewer
        viewer_path = manager.find_pylon_viewer()
        if viewer_path:
            print(f"✓ Pylon Viewer found at: {viewer_path}")
        else:
            print("⚠ Pylon Viewer not found (this is normal if not installed)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Pylon integration import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Pylon integration error: {e}")
        return False

def test_live_feed_headless():
    """Test live feed in headless mode"""
    print("\nTesting live feed headless mode...")
    
    try:
        # Import live feed module
        from live_feed import LiveFeed
        
        # Create live feed in headless mode
        live_feed = LiveFeed(
            camera_index=0,
            use_pylon=False,
            demo_mode=True,  # Use demo mode for testing
            config_file="config.json"
        )
        
        print("✓ Live feed headless mode initialized")
        
        # Test frame reading
        frame = live_feed.read_frame()
        if frame is not None:
            print("✓ Frame reading works in headless mode")
        else:
            print("⚠ No frame available (this is normal in demo mode)")
        
        return True
        
    except Exception as e:
        print(f"✗ Live feed headless mode error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("GUI Fixes and Pylon Integration Test")
    print("=" * 50)
    
    results = []
    
    # Test OpenCV GUI
    results.append(("OpenCV GUI", test_opencv_gui()))
    
    # Test headless mode
    results.append(("Headless Mode", test_headless_mode()))
    
    # Test Pylon integration
    results.append(("Pylon Integration", test_pylon_integration()))
    
    # Test live feed headless
    results.append(("Live Feed Headless", test_live_feed_headless()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The fixes are working correctly.")
    else:
        print("⚠ Some tests failed, but the system should still work in headless mode.")
    
    print("\nRecommendations:")
    print("- If OpenCV GUI failed: The system will run in headless mode")
    print("- If Pylon integration failed: The system will use webcam fallback")
    print("- The core detection functionality should work regardless")

if __name__ == "__main__":
    main() 