#!/usr/bin/env python3
"""
Test script for statistical features module.
"""

import cv2
import numpy as np
import logging
from statistical_features_module import StatisticalFeaturesDetector, StatisticalFeaturesProcessor

def test_statistical_features():
    """Test the statistical features module with the test image."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Statistical Features Module")
    print("=" * 40)
    
    # Load test image (use smaller version for faster testing)
    test_image_path = 'small_statistical_test.bmp'
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Error: Could not load {test_image_path}")
            return
        
        print(f"Loaded test image: {image.shape}")
        
        # Create detector with smaller texture window for faster processing
        detector = StatisticalFeaturesDetector(
            enable_basic_stats=True,
            enable_histogram_features=True,
            enable_texture_stats=True,
            enable_moment_features=True,
            histogram_bins=32,
            texture_window_size=3,  # Smaller window for faster processing
            feature_update_interval=0.1
        )
        
        print("Created statistical features detector")
        
        # Extract features
        features, processed_frame = detector.extract_features(image)
        
        if features:
            print(f"\nExtracted {len(features)} features:")
            
            # Display key features by category
            print("\nBasic Statistics:")
            basic_keys = ['mean', 'std', 'skewness', 'kurtosis', 'entropy', 'energy']
            for key in basic_keys:
                if key in features:
                    print(f"  {key}: {features[key]:.3f}")
            
            print("\nPercentiles:")
            percentile_keys = ['p10', 'p25', 'p50', 'p75', 'p90']
            for key in percentile_keys:
                if key in features:
                    print(f"  {key}: {features[key]:.3f}")
            
            print("\nHistogram Features:")
            hist_keys = [k for k in features.keys() if k.startswith('hist_')]
            for key in hist_keys[:5]:  # Show first 5
                print(f"  {key}: {features[key]:.3f}")
            
            print("\nTexture Features:")
            texture_keys = [k for k in features.keys() if k.startswith('texture_')]
            for key in texture_keys:
                print(f"  {key}: {features[key]:.3f}")
            
            print("\nMoment Features:")
            moment_keys = [k for k in features.keys() if k.startswith('hu_moment_') or k.startswith('centroid_')]
            for key in moment_keys:
                print(f"  {key}: {features[key]:.3f}")
            
            # Save processed frame
            output_path = 'statistical_features_result.bmp'
            cv2.imwrite(output_path, processed_frame)
            print(f"\nSaved processed frame to: {output_path}")
            
            # Get statistics
            stats = detector.get_statistics()
            print(f"\nProcessing Statistics:")
            print(f"  Frames processed: {stats['frames_processed']}")
            print(f"  Features extracted: {stats['features_extracted']}")
            print(f"  Current feature count: {stats['current_feature_count']}")
            print(f"  Processing rate: {stats['processing_rate']:.2f} fps")
            
        else:
            print("No features extracted")
        
        # Test processor
        print("\n" + "=" * 40)
        print("Testing StatisticalFeaturesProcessor")
        
        processor = StatisticalFeaturesProcessor(detector)
        
        # Process frame
        processed_frame = processor.process_frame(image)
        
        if processed_frame is not None:
            output_path = 'statistical_processor_result.bmp'
            cv2.imwrite(output_path, processed_frame)
            print(f"Saved processor result to: {output_path}")
        
        # Test toggle
        enabled = processor.toggle_processing()
        print(f"Processing enabled: {enabled}")
        
        enabled = processor.toggle_processing()
        print(f"Processing enabled: {enabled}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_statistical_features() 