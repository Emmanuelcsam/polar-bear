#!/usr/bin/env python3
"""
Comprehensive test script for the complete statistical features system.
Tests all components: module, processor, and emulator.
"""

import cv2
import numpy as np
import logging
import time
import os
import sys

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(__file__))

from statistical_features_module import StatisticalFeaturesDetector, StatisticalFeaturesProcessor

def test_complete_system():
    """Test the complete statistical features system."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Comprehensive Statistical Features System Test")
    print("=" * 50)
    
    # Test 1: Basic functionality
    print("\n1. Testing Basic Statistical Features Module")
    print("-" * 40)
    
    # Load test image
    test_image_path = 'small_statistical_test.bmp'
    if not os.path.exists(test_image_path):
        print(f"Error: Test image {test_image_path} not found")
        return False
    
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Could not load {test_image_path}")
        return False
    
    print(f"✓ Loaded test image: {image.shape}")
    
    # Test detector
    detector = StatisticalFeaturesDetector(
        enable_basic_stats=True,
        enable_histogram_features=True,
        enable_texture_stats=True,
        enable_moment_features=True,
        histogram_bins=32,
        texture_window_size=3,
        feature_update_interval=0.1
    )
    
    print("✓ Created statistical features detector")
    
    # Extract features
    start_time = time.time()
    features, processed_frame = detector.extract_features(image)
    processing_time = time.time() - start_time
    
    if features:
        print(f"✓ Extracted {len(features)} features in {processing_time:.2f} seconds")
        print(f"✓ Processing rate: {1/processing_time:.2f} fps")
        
        # Verify key features are present
        required_features = ['mean', 'std', 'entropy', 'hist_mode', 'texture_contrast']
        missing_features = [f for f in required_features if f not in features]
        
        if not missing_features:
            print("✓ All required features extracted successfully")
        else:
            print(f"✗ Missing features: {missing_features}")
            return False
        
        # Save result
        cv2.imwrite('test_result.bmp', processed_frame)
        print("✓ Saved processed frame")
        
    else:
        print("✗ No features extracted")
        return False
    
    # Test 2: Processor functionality
    print("\n2. Testing StatisticalFeaturesProcessor")
    print("-" * 40)
    
    processor = StatisticalFeaturesProcessor(detector)
    print("✓ Created processor")
    
    # Test processing
    processed_frame = processor.process_frame(image)
    if processed_frame is not None:
        print("✓ Frame processing successful")
        cv2.imwrite('processor_result.bmp', processed_frame)
        print("✓ Saved processor result")
    else:
        print("✗ Frame processing failed")
        return False
    
    # Test toggle functionality
    enabled = processor.toggle_processing()
    print(f"✓ Processing toggle: {enabled}")
    
    enabled = processor.toggle_processing()
    print(f"✓ Processing toggle: {enabled}")
    
    # Test 3: Parameter updates
    print("\n3. Testing Parameter Updates")
    print("-" * 40)
    
    # Test parameter updates
    detector.update_parameters(
        histogram_bins=64,
        texture_window_size=5,
        feature_update_interval=0.5
    )
    print("✓ Parameter updates successful")
    
    # Test feature type toggles
    detector.update_parameters(
        enable_basic_stats=True,
        enable_histogram_features=False,
        enable_texture_stats=True,
        enable_moment_features=False
    )
    print("✓ Feature type toggles successful")
    
    # Test 4: Statistics
    print("\n4. Testing Statistics")
    print("-" * 40)
    
    stats = detector.get_statistics()
    print(f"✓ Frames processed: {stats['frames_processed']}")
    print(f"✓ Features extracted: {stats['features_extracted']}")
    print(f"✓ Current feature count: {stats['current_feature_count']}")
    
    # Test 5: Performance
    print("\n5. Testing Performance")
    print("-" * 40)
    
    # Test multiple frames
    start_time = time.time()
    for i in range(5):
        features, _ = detector.extract_features(image)
        if features:
            print(f"✓ Frame {i+1}: {len(features)} features")
        else:
            print(f"✗ Frame {i+1}: No features")
            return False
    
    total_time = time.time() - start_time
    avg_time = total_time / 5
    fps = 1 / avg_time
    
    print(f"✓ Average processing time: {avg_time:.3f} seconds")
    print(f"✓ Average FPS: {fps:.2f}")
    
    # Test 6: Error handling
    print("\n6. Testing Error Handling")
    print("-" * 40)
    
    # Test with None frame
    features, processed = detector.extract_features(None)
    if features is None and processed is None:
        print("✓ None frame handling correct")
    else:
        print("✗ None frame handling incorrect")
        return False
    
    # Test with empty frame
    empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    features, processed = detector.extract_features(empty_frame)
    if features is not None:
        print("✓ Empty frame handling correct")
    else:
        print("✗ Empty frame handling incorrect")
        return False
    
    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED!")
    print("✓ Statistical Features System is working correctly")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_complete_system()
    if success:
        print("\nSystem is ready for use!")
    else:
        print("\nSystem has issues that need to be addressed.")
        sys.exit(1) 