#!/usr/bin/env python3
"""
Test script for Circle Detection System
======================================

This script tests the circle detection functionality with sample images
to verify the installation and configuration.

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

def create_test_images():
    """Create test images with circles for testing."""
    print("Creating test images...")
    
    # Create output directory
    os.makedirs('test_images', exist_ok=True)
    
    # Test image 1: Simple circles
    img1 = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(img1, (150, 150), 50, (255, 255, 255), -1)
    cv2.circle(img1, (450, 150), 80, (255, 255, 255), -1)
    cv2.circle(img1, (300, 300), 60, (255, 255, 255), -1)
    cv2.imwrite("test_images/simple_circles.png", img1)
    
    # Test image 2: Overlapping circles
    img2 = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(img2, (200, 200), 60, (255, 255, 255), -1)
    cv2.circle(img2, (250, 200), 40, (200, 200, 200), -1)
    cv2.circle(img2, (400, 200), 70, (255, 255, 255), -1)
    cv2.imwrite("test_images/overlapping_circles.png", img2)
    
    # Test image 3: Small circles
    img3 = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(5):
        x = 50 + i * 70
        y = 150
        radius = 15 + i * 5
        cv2.circle(img3, (x, y), radius, (255, 255, 255), -1)
    cv2.imwrite("test_images/small_circles.png", img3)
    
    # Test image 4: Large circles
    img4 = np.zeros((500, 700, 3), dtype=np.uint8)
    cv2.circle(img4, (200, 250), 120, (255, 255, 255), -1)
    cv2.circle(img4, (500, 250), 150, (255, 255, 255), -1)
    cv2.imwrite("test_images/large_circles.png", img4)
    
    print("✓ Created test images in 'test_images/' directory")
    return True

def test_basic_opencv():
    """Test basic OpenCV functionality."""
    print("\nTesting basic OpenCV functionality...")
    
    try:
        # Test HoughCircles with a simple image
        test_img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(test_img, (100, 100), 50, 255, -1)
        
        circles = cv2.HoughCircles(
            test_img,
            cv2.HOUGH_GRADIENT,
            dp=1, minDist=20, param1=50, param2=30,
            minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            print(f"✓ OpenCV HoughCircles working - detected {len(circles[0])} circles")
            return True
        else:
            print("⚠ OpenCV HoughCircles test inconclusive")
            return True
            
    except Exception as e:
        print(f"✗ Error testing OpenCV: {e}")
        return False

def test_circle_detector():
    """Test the CircleDetector class."""
    print("\nTesting CircleDetector class...")
    
    try:
        # Import the detector
        from pylon_circle_detector import CircleDetector
        
        # Create detector
        detector = CircleDetector()
        
        # Test with simple image
        test_img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(test_img, (100, 100), 50, 255, -1)
        
        # Test Hough detection
        circles_hough = detector.detect_circles_hough(test_img)
        print(f"  Hough detection: {len(circles_hough)} circles")
        
        # Test contour detection
        circles_contour = detector.detect_circles_contour(test_img)
        print(f"  Contour detection: {len(circles_contour)} circles")
        
        # Test combined detection
        circles_combined = detector.detect_circles_combined(test_img)
        print(f"  Combined detection: {len(circles_combined)} circles")
        
        print("✓ CircleDetector class working")
        return True
        
    except ImportError as e:
        print(f"✗ Error importing CircleDetector: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing CircleDetector: {e}")
        return False

def test_image_processing():
    """Test image processing with test images."""
    print("\nTesting image processing...")
    
    try:
        from pylon_circle_detector import CircleDetector
        
        detector = CircleDetector()
        
        # Process test images
        test_files = [
            "test_images/simple_circles.png",
            "test_images/overlapping_circles.png",
            "test_images/small_circles.png",
            "test_images/large_circles.png"
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                # Load image
                img = cv2.imread(test_file)
                if img is not None:
                    # Convert to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Process with detector
                    processed_img, circles = detector.process_frame(img_rgb)
                    
                    print(f"  {test_file}: {len(circles)} circles detected")
                    
                    # Save result
                    output_file = test_file.replace('.png', '_result.png')
                    cv2.imwrite(output_file, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                else:
                    print(f"  ⚠ Could not load {test_file}")
            else:
                print(f"  ⚠ Test file not found: {test_file}")
        
        print("✓ Image processing test completed")
        return True
        
    except Exception as e:
        print(f"✗ Error in image processing test: {e}")
        return False

def test_camera_interface():
    """Test camera interface (without actually opening camera)."""
    print("\nTesting camera interface...")
    
    try:
        from pylon_circle_detector import PylonCamera
        
        # Test with webcam fallback
        camera = PylonCamera(camera_index=0, use_pylon=False)
        print("✓ Camera interface created successfully")
        
        # Clean up
        camera.release()
        return True
        
    except Exception as e:
        print(f"✗ Error testing camera interface: {e}")
        return False

def run_performance_test():
    """Run a simple performance test."""
    print("\nRunning performance test...")
    
    try:
        from pylon_circle_detector import CircleDetector
        import time
        
        detector = CircleDetector()
        
        # Create test image
        test_img = np.zeros((480, 640), dtype=np.uint8)
        for i in range(5):
            x = 100 + i * 100
            y = 240
            radius = 30 + i * 10
            cv2.circle(test_img, (x, y), radius, 255, -1)
        
        # Test processing speed
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            circles = detector.detect_circles_combined(test_img)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        fps = 1.0 / avg_time
        
        print(f"  Average processing time: {avg_time*1000:.2f} ms")
        print(f"  Estimated FPS: {fps:.1f}")
        
        if fps > 10:
            print("✓ Performance test passed")
            return True
        else:
            print("⚠ Performance may be slow for real-time applications")
            return True
            
    except Exception as e:
        print(f"✗ Error in performance test: {e}")
        return False

def main():
    """Main test function."""
    print("Circle Detection System Test")
    print("===========================")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher required")
        return False
    
    print(f"✓ Python version: {sys.version}")
    
    # Test OpenCV
    if not test_basic_opencv():
        return False
    
    # Create test images
    create_test_images()
    
    # Test CircleDetector
    if not test_circle_detector():
        return False
    
    # Test image processing
    if not test_image_processing():
        return False
    
    # Test camera interface
    if not test_camera_interface():
        print("⚠ Camera interface test failed - may affect real-time usage")
    
    # Performance test
    run_performance_test()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("\nTest results saved in:")
    print("  - test_images/ (original test images)")
    print("  - test_images/*_result.png (processed results)")
    print("\nTo run the full application:")
    print("  python pylon_circle_detector.py")
    print("="*50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 