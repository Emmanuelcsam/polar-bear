#!/usr/bin/env python3
"""
Test script for blob detection functionality.
Tests the blob detector on the created blob_test.bmp file.
"""

import cv2
import numpy as np
from blob_detector_module import BlobDetector
from pathlib import Path


def test_blob_detection():
    """Test blob detection on the test image."""
    print("Testing Blob Detection...")

    # Check if test image exists
    test_image_path = "blob_test.bmp"
    if not Path(test_image_path).exists():
        print(f"Error: Test image {test_image_path} not found!")
        print("Please run 'python create_blob_test_image.py' first.")
        return False

    # Load the test image
    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"Error: Could not load test image from {test_image_path}")
        return False

    print(f"Loaded test image: {frame.shape}")

    # Create blob detector with default parameters
    detector = BlobDetector()

    # Detect blobs
    print("Detecting blobs...")
    detections, processed_frame = detector.detect_blobs(frame)

    # Display results
    print(f"\nDetection Results:")
    print(f"Blobs found: {len(detections) if detections else 0}")

    if detections:
        print("\nDetected blobs:")
        for i, blob in enumerate(detections):
            print(f"  Blob {i+1}:")
            print(f"    Location: {blob['location']}")
            print(f"    Center: {blob['center']}")
            print(f"    Area: {blob['area']}")
            print(f"    Circularity: {blob['circularity']:.3f}")
            print(f"    Radius: {blob['radius']}")
            print(f"    Confidence: {blob['confidence']:.3f}")
            print()

    # Save the processed image
    output_path = "blob_detection_result.bmp"
    success = cv2.imwrite(output_path, processed_frame)
    if success:
        print(f"Saved detection result to: {output_path}")
    else:
        print("Error: Failed to save detection result")

    # Get statistics
    stats = detector.get_statistics()
    print(f"\nDetection Statistics:")
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Blobs detected: {stats['blobs_detected']}")
    print(f"Detection rate: {stats['detection_rate']:.3f}")

    return True


def test_parameter_adjustment():
    """Test parameter adjustment functionality."""
    print("\n" + "="*50)
    print("Testing Parameter Adjustment...")

    # Load test image
    frame = cv2.imread("blob_test.bmp")
    if frame is None:
        print("Error: Could not load test image")
        return False

    # Test different parameter sets
    parameter_sets = [
        {
            "name": "Default",
            "params": {}
        },
        {
            "name": "Small Blobs",
            "params": {
                "min_blob_area": 20,
                "max_blob_area": 500,
                "min_blob_circularity": 0.5,
                "threshold_value": 120
            }
        },
        {
            "name": "Large Blobs",
            "params": {
                "min_blob_area": 500,
                "max_blob_area": 10000,
                "min_blob_circularity": 0.2,
                "threshold_value": 100
            }
        }
    ]

    for param_set in parameter_sets:
        print(f"\nTesting {param_set['name']} parameters:")

        detector = BlobDetector()
        if param_set['params']:
            detector.update_parameters(**param_set['params'])

        detections, _ = detector.detect_blobs(frame)
        blob_count = len(detections) if detections else 0

        print(f"  Blobs detected: {blob_count}")

    return True


if __name__ == "__main__":
    print("Blob Detection Test Suite")
    print("=" * 50)

    # Test basic detection
    if test_blob_detection():
        # Test parameter adjustment
        test_parameter_adjustment()

        print("\n" + "="*50)
        print("All tests completed successfully!")
        print("You can now run 'python run_blob_detection.py' to start the GUI emulator.")
    else:
        print("Basic detection test failed.")
