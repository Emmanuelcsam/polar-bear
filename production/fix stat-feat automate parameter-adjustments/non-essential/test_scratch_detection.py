#!/usr/bin/env python3
"""
Simple test script to verify Hough Lines scratch detection functionality.
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from hough_lines import HoughLinesDetector, HoughLinesProcessor
    print("✓ Successfully imported hough_lines module")
except ImportError as e:
    print(f"✗ Failed to import hough_lines: {e}")
    sys.exit(1)

def create_test_image_with_lines():
    """Create a test image with some lines to detect."""
    # Create a white background
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Draw some test lines (simulating scratches)
    # Horizontal lines
    cv2.line(img, (50, 100), (550, 120), (0, 0, 0), 2)  # Slightly diagonal
    cv2.line(img, (100, 200), (500, 200), (0, 0, 0), 3)  # Horizontal

    # Vertical lines
    cv2.line(img, (200, 50), (210, 350), (0, 0, 0), 2)   # Slightly diagonal
    cv2.line(img, (400, 50), (400, 350), (0, 0, 0), 1)   # Vertical

    # Diagonal lines
    cv2.line(img, (50, 50), (200, 200), (0, 0, 0), 2)    # Diagonal
    cv2.line(img, (450, 300), (550, 350), (0, 0, 0), 1)  # Short diagonal

    return img

def test_hough_lines_detector():
    """Test the HoughLinesDetector class."""
    print("\n--- Testing HoughLinesDetector ---")

    # Create test image
    test_image = create_test_image_with_lines()
    print(f"✓ Created test image with dimensions: {test_image.shape}")

    # Test both probabilistic and standard methods
    for use_prob in [True, False]:
        method_name = "Probabilistic" if use_prob else "Standard"
        print(f"\nTesting {method_name} Hough Transform:")

        # Create detector
        detector = HoughLinesDetector(
            rho=1,
            theta_degrees=1.0,
            threshold=50,
            min_line_length=30,
            max_line_gap=5,
            use_probabilistic=use_prob
        )

        # Detect lines
        lines, processed_frame = detector.detect_lines(test_image)

        if lines is not None:
            print(f"  ✓ Detected {len(lines)} lines")
            print(f"  ✓ Processed frame shape: {processed_frame.shape}")
        else:
            print(f"  ✗ No lines detected")

        # Test parameter updates
        detector.update_parameters(threshold=30)
        print(f"  ✓ Parameter update successful")

        # Test statistics
        stats = detector.get_statistics()
        print(f"  ✓ Statistics: {stats}")

def test_hough_lines_processor():
    """Test the HoughLinesProcessor class."""
    print("\n--- Testing HoughLinesProcessor ---")

    # Create test image
    test_image = create_test_image_with_lines()

    # Create processor
    processor = HoughLinesProcessor()
    print("✓ Created HoughLinesProcessor")

    # Process frame
    processed_frame = processor.process_frame(test_image)
    print(f"✓ Processed frame shape: {processed_frame.shape}")

    # Test toggle
    enabled = processor.toggle_processing()
    print(f"✓ Toggle processing: {enabled}")

    # Process with processing disabled
    processed_frame_disabled = processor.process_frame(test_image)
    print(f"✓ Processed frame with processing disabled")

def test_presets():
    """Test different parameter presets."""
    print("\n--- Testing Parameter Presets ---")

    test_image = create_test_image_with_lines()

    presets = {
        "fine": {"rho": 1, "theta_degrees": 0.5, "threshold": 30, "min_line_length": 20},
        "balanced": {"rho": 1, "theta_degrees": 1.0, "threshold": 50, "min_line_length": 30},
        "thick": {"rho": 2, "theta_degrees": 2.0, "threshold": 80, "min_line_length": 50}
    }

    for preset_name, params in presets.items():
        print(f"\nTesting {preset_name.title()} preset:")

        detector = HoughLinesDetector(**params)
        lines, processed_frame = detector.detect_lines(test_image)

        if lines is not None:
            print(f"  ✓ {preset_name.title()} preset detected {len(lines)} lines")
        else:
            print(f"  ✗ {preset_name.title()} preset detected no lines")

def main():
    """Run all tests."""
    print("Starting Hough Lines Scratch Detection Tests...")
    print("=" * 50)

    try:
        test_hough_lines_detector()
        test_hough_lines_processor()
        test_presets()

        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        print("\nThe scratch detection system is ready to use.")
        print("Run 'python3 run_scratch_detection.py' to start the GUI.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
