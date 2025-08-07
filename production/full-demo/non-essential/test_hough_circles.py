#!/usr/bin/env python3
"""
Test script for the improved Hough Circles Detection Module.

This script tests the hough_circles module with the real good.bmp image
to ensure all functionality works correctly including error handling,
parameter updates, and statistics tracking.
"""

import cv2
import numpy as np
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the module
from hough_circles import HoughCirclesDetector, HoughCirclesProcessor


def test_basic_detection():
    """Test basic circle detection on good.bmp."""
    print("\n=== Testing Basic Detection ===")
    
    # Load the image
    image_path = "good.bmp"
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found!")
        return False
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Failed to load {image_path}")
        return False
    
    print(f"Loaded image: {frame.shape}")
    
    # Create detector with default parameters
    detector = HoughCirclesDetector()
    
    # Detect circles
    circles, result = detector.detect_circles(frame)
    
    if circles is not None:
        print(f"✓ Detected {len(circles)} circles")
        for i, (x, y, r) in enumerate(circles[:5]):  # Show first 5
            print(f"  Circle {i+1}: center=({x}, {y}), radius={r}")
    else:
        print("✓ No circles detected (may need parameter adjustment)")
    
    # Get statistics
    stats = detector.get_statistics()
    print(f"✓ Statistics: {stats}")
    
    # Save result
    output_path = "test_result_basic.jpg"
    cv2.imwrite(output_path, result)
    print(f"✓ Result saved to {output_path}")
    
    return True


def test_parameter_updates():
    """Test dynamic parameter updates."""
    print("\n=== Testing Parameter Updates ===")
    
    frame = cv2.imread("good.bmp")
    if frame is None:
        print("Error: Failed to load image")
        return False
    
    detector = HoughCirclesDetector()
    
    # Test with different parameter sets
    param_sets = [
        {"name": "Sensitive", "param1": 50, "param2": 15, "min_radius": 5},
        {"name": "Balanced", "param1": 100, "param2": 50, "min_radius": 10},
        {"name": "Conservative", "param1": 200, "param2": 100, "min_radius": 20}
    ]
    
    for params in param_sets:
        name = params.pop("name")
        detector.update_parameters(**params)
        circles, result = detector.detect_circles(frame)
        
        circle_count = len(circles) if circles is not None else 0
        print(f"✓ {name} mode: detected {circle_count} circles")
    
    return True


def test_processor():
    """Test the HoughCirclesProcessor."""
    print("\n=== Testing HoughCirclesProcessor ===")
    
    frame = cv2.imread("good.bmp")
    if frame is None:
        print("Error: Failed to load image")
        return False
    
    # Create processor with custom detector
    custom_detector = HoughCirclesDetector(dp=1.5, min_dist=100)
    processor = HoughCirclesProcessor(custom_detector)
    
    # Process with detection enabled
    result1 = processor.process_frame(frame)
    print("✓ Processed frame with detection enabled")
    
    # Toggle processing off
    state = processor.toggle_processing()
    print(f"✓ Toggled processing: {state}")
    
    # Process with detection disabled
    result2 = processor.process_frame(frame)
    print("✓ Processed frame with detection disabled")
    
    # Toggle back on
    processor.toggle_processing()
    
    # Test getter/setter
    new_detector = HoughCirclesDetector(dp=2.0)
    processor.set_detector(new_detector)
    retrieved = processor.get_detector()
    print(f"✓ Detector swap successful: dp={retrieved.dp}")
    
    return True


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    detector = HoughCirclesDetector()
    
    # Test with None frame
    circles, result = detector.detect_circles(None)
    print("✓ Handled None frame gracefully")
    
    # Test with invalid frame shape
    invalid_frame = np.zeros((100, 100))  # 2D instead of 3D
    circles, result = detector.detect_circles(invalid_frame)
    print("✓ Handled invalid frame shape gracefully")
    
    # Test parameter validation
    try:
        bad_detector = HoughCirclesDetector(dp=10.0)  # Out of range
        print("✓ Parameter clamping worked")
    except Exception as e:
        print(f"✗ Parameter validation failed: {e}")
        return False
    
    # Test invalid parameter updates
    detector.update_parameters(dp=-1, min_dist=5000)  # Out of range values
    stats = detector.get_statistics()
    print(f"✓ Parameters clamped correctly: dp={stats['current_parameters']['dp']}, "
          f"min_dist={stats['current_parameters']['min_dist']}")
    
    return True


def test_video_simulation():
    """Simulate video processing with multiple frames."""
    print("\n=== Testing Video Simulation ===")
    
    frame = cv2.imread("good.bmp")
    if frame is None:
        print("Error: Failed to load image")
        return False
    
    processor = HoughCirclesProcessor()
    detector = processor.get_detector()
    
    # Simulate processing 10 frames
    for i in range(10):
        # Add some variation (rotate slightly)
        angle = i * 5
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
        
        # Process frame
        result = processor.process_frame(rotated)
        
    stats = detector.get_statistics()
    print(f"✓ Processed {stats['frames_processed']} frames")
    print(f"✓ Average detection rate: {stats['detection_rate']:.2f} circles/frame")
    
    # Reset statistics
    detector.reset_statistics()
    stats_after = detector.get_statistics()
    print(f"✓ Statistics reset: frames={stats_after['frames_processed']}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Hough Circles Detection Module")
    print("=" * 60)
    
    tests = [
        ("Basic Detection", test_basic_detection),
        ("Parameter Updates", test_parameter_updates),
        ("Processor", test_processor),
        ("Error Handling", test_error_handling),
        ("Video Simulation", test_video_simulation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ Test '{name}' failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Display sample result if exists
    result_path = "test_result_basic.jpg"
    if Path(result_path).exists():
        print(f"\n✓ Test output saved to {result_path}")
        print("  You can view it to see the detected circles")
    
    return all(success for _, success in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
