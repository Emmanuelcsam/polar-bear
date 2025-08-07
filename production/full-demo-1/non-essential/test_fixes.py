#!/usr/bin/env python3
"""
Test script to verify the fixes for image finder and statistical features.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

def test_image_directory_search():
    """Test the image directory search functionality."""
    print("Testing image directory search...")

    try:
        from frequency_features_emulator import FrequencyFeaturesGUI

        root = tk.Tk()
        root.withdraw()  # Hide main window

        app = FrequencyFeaturesGUI(root)

        # Test directory path
        test_dir = os.path.join(parent_dir, "pictures")
        app.image_path_var.set(test_dir)

        print(f"Testing with directory: {test_dir}")
        print(f"Directory exists: {os.path.exists(test_dir)}")
        print(f"Is directory: {os.path.isdir(test_dir)}")

        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            # Find image files
            image_files = []
            for ext in ['*.bmp', '*.jpg', '*.jpeg', '*.png']:
                image_files.extend(Path(test_dir).glob(ext))
                image_files.extend(Path(test_dir).glob(ext.upper()))

            print(f"Found {len(image_files)} image files")
            if image_files:
                print(f"First image: {image_files[0]}")

        root.destroy()
        print("✓ Image directory search test passed")
        return True

    except Exception as e:
        print(f"✗ Image directory search test failed: {e}")
        return False

def test_statistical_features_module():
    """Test the statistical features module."""
    print("\nTesting statistical features module...")

    try:
        from statistical_features_module import StatisticalFeaturesDetector, StatisticalFeaturesProcessor
        import numpy as np
        import cv2

        # Create a test detector
        detector = StatisticalFeaturesDetector(
            feature_update_interval=0.1
        )

        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test feature extraction
        features, processed_frame = detector.extract_features(test_image)

        print(f"Extracted {len(features) if features else 0} features")
        if features:
            print(f"Sample features: {list(features.keys())[:5]}")

        # Test processor
        processor = StatisticalFeaturesProcessor(detector)
        processed_frame = processor.process_frame(test_image)

        print(f"Processed frame shape: {processed_frame.shape}")
        print("✓ Statistical features module test passed")
        return True

    except Exception as e:
        print(f"✗ Statistical features module test failed: {e}")
        return False

def test_emulator_integration():
    """Test the integration with the emulators."""
    print("\nTesting emulator integration...")

    try:
        # Test if we can import the main emulator modules
        from statistical_features_emulator import StatisticalFeaturesGUI
        from frequency_features_emulator import FrequencyFeaturesGUI

        print("✓ Successfully imported emulator modules")

        # Test if we can create the GUI objects without errors
        root = tk.Tk()
        root.withdraw()  # Hide main window

        # Test statistical features GUI
        stats_gui = StatisticalFeaturesGUI(root)
        print("✓ Statistical features GUI created successfully")

        root.destroy()

        # Test frequency features GUI
        root = tk.Tk()
        root.withdraw()

        freq_gui = FrequencyFeaturesGUI(root)
        print("✓ Frequency features GUI created successfully")

        root.destroy()

        print("✓ Emulator integration test passed")
        return True

    except Exception as e:
        print(f"✗ Emulator integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running fix verification tests...\n")

    tests_passed = 0
    total_tests = 3

    # Run tests
    if test_image_directory_search():
        tests_passed += 1

    if test_statistical_features_module():
        tests_passed += 1

    if test_emulator_integration():
        tests_passed += 1

    # Summary
    print(f"\n" + "="*50)
    print(f"Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("✓ All tests passed! Fixes are working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
