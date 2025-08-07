#!/usr/bin/env python3
"""
Quick test to validate scratch detection on the test image.
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from hough_lines import HoughLinesDetector
    print("✓ Successfully imported hough_lines module")
except ImportError as e:
    print(f"✗ Failed to import hough_lines: {e}")
    sys.exit(1)

def test_scratch_detection_on_test_image():
    """Test scratch detection on our artificial test image."""
    print("\n--- Testing Scratch Detection on Test Image ---")

    # Load the test image
    test_image_path = "test_scratches.bmp"
    if not os.path.exists(test_image_path):
        print(f"✗ Test image not found: {test_image_path}")
        print("Please run 'python3 create_test_image.py' first")
        return False

    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"✗ Could not load test image: {test_image_path}")
        return False

    print(f"✓ Loaded test image: {test_image.shape}")

    # Test different detector configurations
    configs = [
        {
            "name": "Fine Detection",
            "params": {
                "rho": 1, "theta_degrees": 0.5, "threshold": 30,
                "min_line_length": 20, "max_line_gap": 5,
                "canny_low": 30, "canny_high": 100, "use_probabilistic": True
            }
        },
        {
            "name": "Balanced Detection",
            "params": {
                "rho": 1, "theta_degrees": 1.0, "threshold": 50,
                "min_line_length": 30, "max_line_gap": 5,
                "canny_low": 50, "canny_high": 150, "use_probabilistic": True
            }
        },
        {
            "name": "Thick Line Detection",
            "params": {
                "rho": 2, "theta_degrees": 2.0, "threshold": 80,
                "min_line_length": 50, "max_line_gap": 10,
                "canny_low": 80, "canny_high": 200, "use_probabilistic": True
            }
        }
    ]

    results = []

    for config in configs:
        print(f"\nTesting {config['name']}:")

        # Create detector with specific configuration
        detector = HoughLinesDetector(**config['params'])

        # Detect lines
        lines, processed_frame = detector.detect_lines(test_image)

        if lines is not None:
            line_count = len(lines)
            print(f"  ✓ Detected {line_count} lines")
            results.append((config['name'], line_count, processed_frame))

            # Save the result image
            output_name = f"result_{config['name'].lower().replace(' ', '_')}.bmp"
            cv2.imwrite(output_name, processed_frame)
            print(f"  ✓ Saved result: {output_name}")
        else:
            print(f"  ✗ No lines detected")
            results.append((config['name'], 0, None))

    return results

def main():
    """Main function to test scratch detection."""
    print("Testing Scratch Detection on Artificial Test Image")
    print("=" * 55)

    try:
        results = test_scratch_detection_on_test_image()

        if results:
            print("\n" + "=" * 55)
            print("DETECTION RESULTS SUMMARY:")
            print("-" * 30)

            for name, count, _ in results:
                status = "✓" if count > 0 else "✗"
                print(f"  {status} {name}: {count} lines detected")

            print("\n📁 Files created:")
            print("  - result_fine_detection.bmp")
            print("  - result_balanced_detection.bmp")
            print("  - result_thick_line_detection.bmp")

            print(f"\n🎯 Expected: ~15 artificial scratches were added to the test image")
            print("📊 Results show how different parameters affect detection sensitivity")

            print(f"\n🚀 Ready to run the interactive GUI!")
            print("   Run: python3 run_test_scratch_detection.py")

            return 0
        else:
            print("\n✗ Testing failed")
            return 1

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
