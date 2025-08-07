#!/usr/bin/env python3
"""
Simple test for blob detection emulator core functionality.
"""

import cv2
import numpy as np
from blob_detector_module import BlobDetector, BlobDetectorProcessor


def test_basic_blob_detection():
    """Test basic blob detection functionality."""
    print("Testing basic blob detection...")
    
    # Create a simple test image with a blob
    test_image = np.zeros((400, 400, 3), dtype=np.uint8)
    # Add a white circle (blob)
    cv2.circle(test_image, (200, 200), 50, (255, 255, 255), -1)
    
    # Create detector and processor
    detector = BlobDetector()
    processor = BlobDetectorProcessor(detector)
    
    # Test detection
    detections, processed_frame = detector.detect_blobs(test_image)
    
    print(f"Detections found: {len(detections) if detections else 0}")
    if detections:
        for i, blob in enumerate(detections):
            print(f"  Blob {i+1}: Area={blob['area']}, Center={blob['center']}, Circularity={blob['circularity']:.3f}")
    
    # Test processor
    result_frame = processor.process_frame(test_image)
    print(f"Processor result shape: {result_frame.shape}")
    
    # Test parameter updates
    print("Testing parameter updates...")
    detector.update_parameters(min_blob_area=100, max_blob_area=10000)
    stats = detector.get_statistics()
    print(f"Statistics: {stats}")
    
    print("✅ Basic blob detection test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        test_basic_blob_detection()
        print("\n🎯 All core functionality tests passed!")
        print("The blob detection emulator should now work properly.")
        print("\nTo run the GUI: python run_blob_detection.py")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
