#!/usr/bin/env python3
"""
Test script for Morphological Features Analysis.
Tests the morphological detector with the created test image.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from morphological_features_module import MorphologicalDetector, MorphologicalProcessor


def test_morphological_detector():
    """Test the morphological detector with test images."""
    print("Testing Morphological Features Detector")
    print("=" * 40)

    # Load test images
    test_images = [
        "pictures/morphological_test.bmp",
        "pictures/morphological_simple_test.bmp",
        "pictures/good.bmp"
    ]

    for image_path in test_images:
        path = Path(image_path)
        if not path.exists():
            print(f"Skipping {image_path} - file not found")
            continue

        print(f"\nTesting with: {image_path}")
        print("-" * 30)

        # Load image
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Could not load {image_path}")
            continue

        print(f"Image size: {frame.shape}")

        # Create detector with default parameters
        detector = MorphologicalDetector()

        # Analyze frame
        start_time = time.time()
        results, processed_frame = detector.analyze_frame(frame)
        analysis_time = time.time() - start_time

        print(f"Analysis time: {analysis_time:.3f} seconds")

        if results:
            print(f"Results keys: {list(results.keys())}")

            # Print features summary
            if 'features' in results:
                features = results['features']
                print(f"Morphological features: {len(features)}")
                # Print first few features
                for i, (key, value) in enumerate(list(features.items())[:5]):
                    print(f"  {key}: {value:.3f}")

            # Print complexity summary
            if 'complexity' in results:
                complexity = results['complexity']
                print(f"Shape complexity features: {len(complexity)}")
                for key, value in complexity.items():
                    print(f"  {key}: {value:.3f}")

            # Print skeleton summary
            if 'skeleton' in results:
                skeleton = results['skeleton']
                print(f"Skeleton features: {len(skeleton)}")
                for key, value in skeleton.items():
                    print(f"  {key}: {value:.3f}")

            # Print component summary
            if 'components' in results:
                components = results['components']
                print(f"Connected components: {len(components)}")
                for i, comp in enumerate(components[:3]):  # First 3
                    print(f"  Component {i+1}: area={comp['area']}, circularity={comp['circularity']:.2f}")

            # Print defect summary
            if 'defects' in results:
                defects = results['defects']
                print(f"Defect maps: {len(defects)}")
                for name, defect_map in defects.items():
                    defect_pixels = np.sum(defect_map > 30)
                    print(f"  {name}: {defect_pixels} defect pixels")
        else:
            print("No results returned")

        # Save processed frame
        output_path = str(path).replace('.bmp', '_morphological_result.bmp')
        cv2.imwrite(output_path, processed_frame)
        print(f"Processed frame saved to: {output_path}")

        # Get statistics
        stats = detector.get_statistics()
        print(f"Detector statistics: {stats}")


def test_morphological_processor():
    """Test the morphological processor."""
    print("\n\nTesting Morphological Features Processor")
    print("=" * 40)

    # Load test image
    test_image_path = "pictures/morphological_test.bmp"
    if not Path(test_image_path).exists():
        print(f"Test image not found: {test_image_path}")
        return

    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"Could not load test image: {test_image_path}")
        return

    print(f"Processing test image: {test_image_path}")

    # Create processor
    processor = MorphologicalProcessor()

    # Test processing
    processed_frame = processor.process_frame(frame)

    # Save result
    output_path = "pictures/morphological_processor_test_result.bmp"
    cv2.imwrite(output_path, processed_frame)
    print(f"Processor result saved to: {output_path}")

    # Test toggle
    print(f"Processing enabled: {processor.is_processing_enabled()}")
    processor.toggle_processing()
    print(f"After toggle: {processor.is_processing_enabled()}")

    # Process with disabled processing
    processed_disabled = processor.process_frame(frame)
    output_disabled = "pictures/morphological_processor_disabled_result.bmp"
    cv2.imwrite(output_disabled, processed_disabled)
    print(f"Disabled processing result saved to: {output_disabled}")


def test_parameter_updates():
    """Test parameter updating functionality."""
    print("\n\nTesting Parameter Updates")
    print("=" * 40)

    # Create detector
    detector = MorphologicalDetector()

    # Load test image
    frame = cv2.imread("pictures/morphological_test.bmp")
    if frame is None:
        print("Could not load test image for parameter testing")
        return

    print("Testing with default parameters...")
    results1, _ = detector.analyze_frame(frame)

    # Update parameters
    print("Updating parameters...")
    detector.update_parameters(
        analysis_types=['features', 'defects'],
        kernel_sizes=[5, 9, 13],
        min_component_area=100,
        defect_threshold=50,
        filter_operation='tophat',
        filter_kernel_size=7
    )

    print("Testing with updated parameters...")
    results2, processed = detector.analyze_frame(frame)

    if results1 and results2:
        print("Parameter update comparison:")
        if 'features' in results1 and 'features' in results2:
            print(f"  Features before: {len(results1['features'])}")
            print(f"  Features after: {len(results2['features'])}")
        if 'components' in results1 and 'components' in results2:
            print(f"  Components before: {len(results1.get('components', []))}")
            print(f"  Components after: {len(results2.get('components', []))}")

    # Save result with updated parameters
    output_path = "pictures/morphological_updated_params_result.bmp"
    cv2.imwrite(output_path, processed)
    print(f"Updated parameters result saved to: {output_path}")


def test_different_presets():
    """Test different parameter presets."""
    print("\n\nTesting Different Parameter Presets")
    print("=" * 40)

    # Load test image
    frame = cv2.imread("pictures/morphological_test.bmp")
    if frame is None:
        print("Could not load test image for preset testing")
        return

    # Define presets
    presets = {
        "fine_detail": {
            "analysis_types": ['features', 'complexity', 'defects'],
            "kernel_sizes": [3, 5, 7],
            "min_component_area": 25,
            "defect_threshold": 15,
            "filter_operation": "tophat",
            "filter_kernel_size": 3
        },
        "coarse_features": {
            "analysis_types": ['features', 'components'],
            "kernel_sizes": [7, 11, 15],
            "min_component_area": 100,
            "defect_threshold": 50,
            "filter_operation": "opening",
            "filter_kernel_size": 9
        },
        "defect_focused": {
            "analysis_types": ['defects', 'features'],
            "kernel_sizes": [5, 7, 9],
            "min_component_area": 30,
            "defect_threshold": 10,
            "filter_operation": "blackhat",
            "filter_kernel_size": 7
        }
    }

    for preset_name, params in presets.items():
        print(f"\nTesting preset: {preset_name}")
        detector = MorphologicalDetector(**params)

        start_time = time.time()
        results, processed = detector.analyze_frame(frame)
        analysis_time = time.time() - start_time

        print(f"  Analysis time: {analysis_time:.3f} seconds")

        if results:
            if 'features' in results:
                print(f"  Features extracted: {len(results['features'])}")
            if 'components' in results:
                print(f"  Components found: {len(results['components'])}")
            if 'defects' in results:
                defect_count = sum(np.sum(dm > params['defect_threshold'])
                                 for dm in results['defects'].values())
                print(f"  Defect pixels: {defect_count}")

        # Save result
        output_path = f"pictures/morphological_{preset_name}_result.bmp"
        cv2.imwrite(output_path, processed)
        print(f"  Result saved to: {output_path}")


def main():
    """Run all tests."""
    print("Morphological Features Analysis - Complete Test Suite")
    print("=" * 60)

    try:
        # Test basic detector functionality
        test_morphological_detector()

        # Test processor functionality
        test_morphological_processor()

        # Test parameter updates
        test_parameter_updates()

        # Test different presets
        test_different_presets()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("Check the 'pictures' directory for output images.")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
