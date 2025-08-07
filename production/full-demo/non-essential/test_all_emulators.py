#!/usr/bin/env python3
"""
Test all emulators to ensure they work correctly.
This script imports and tests all emulator modules without launching the GUI.
"""

import sys
import traceback
from pathlib import Path


def test_import(module_name, description):
    """Test importing a module."""
    print(f"Testing {description}...")
    try:
        __import__(module_name)
        print(f"  ✓ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ {module_name} failed to import: {e}")
        return False


def test_all_emulators():
    """Test all emulator modules."""
    print("Testing All Emulator Modules")
    print("=" * 40)

    # List of emulators to test
    emulators = [
        ("bmp_video_emulator", "BMP Video Emulator"),
        ("blob_detection_emulator", "Blob Detection Emulator"),
        ("scratch_detection_emulator", "Scratch Detection Emulator"),
        ("ssim_detection_emulator", "SSIM Detection Emulator"),
        ("statistical_features_emulator", "Statistical Features Emulator"),
        ("frequency_features_emulator", "Frequency Features Emulator"),
        ("morphological_features_emulator", "Morphological Features Emulator")
    ]

    # Test detector modules
    detectors = [
        ("blob_detector_module", "Blob Detector Module"),
        ("ssim_detector_module", "SSIM Detector Module"),
        ("statistical_features_module", "Statistical Features Module"),
        ("morphological_features_module", "Morphological Features Module")
    ]

    # Test other modules
    other_modules = [
        ("hough_circles", "Hough Circles Detection"),
        ("hough_lines", "Hough Lines Detection"),
        ("pylon_grabber", "Pylon Frame Grabber")
    ]

    total_tests = 0
    passed_tests = 0

    # Test emulator modules
    print("\n1. Testing Emulator Modules:")
    print("-" * 30)
    for module, description in emulators:
        if test_import(module, description):
            passed_tests += 1
        total_tests += 1

    # Test detector modules
    print("\n2. Testing Detector Modules:")
    print("-" * 30)
    for module, description in detectors:
        if test_import(module, description):
            passed_tests += 1
        total_tests += 1

    # Test other modules
    print("\n3. Testing Other Modules:")
    print("-" * 30)
    for module, description in other_modules:
        if test_import(module, description):
            passed_tests += 1
        total_tests += 1

    # Test dev modules
    print("\n4. Testing Dev Modules:")
    print("-" * 30)
    dev_modules = [
        ("dev.morphological_features", "Dev Morphological Features"),
        ("dev.statistical_features", "Dev Statistical Features"),
        ("dev.frequency_features", "Dev Frequency Features"),
        ("dev.ssim_detector", "Dev SSIM Detector"),
        ("dev.blob_detector", "Dev Blob Detector")
    ]

    for module, description in dev_modules:
        if test_import(module, description):
            passed_tests += 1
        total_tests += 1

    # Summary
    print("\n" + "=" * 40)
    print(f"Test Summary: {passed_tests}/{total_tests} modules passed")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print("🎉 All modules imported successfully!")
    else:
        print(f"⚠️  {total_tests - passed_tests} modules failed to import")

    return passed_tests == total_tests


def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\n\nTesting Basic Functionality")
    print("=" * 40)

    tests_passed = 0
    total_tests = 0

    # Test morphological features
    print("\nTesting Morphological Features...")
    try:
        from morphological_features_module import MorphologicalDetector
        import numpy as np

        # Create test detector
        detector = MorphologicalDetector()

        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test analysis
        results, processed = detector.analyze_frame(test_image)

        if results is not None and processed is not None:
            print("  ✓ Morphological analysis working")
            tests_passed += 1
        else:
            print("  ✗ Morphological analysis failed")

    except Exception as e:
        print(f"  ✗ Morphological features test failed: {e}")

    total_tests += 1

    # Test blob detector
    print("\nTesting Blob Detector...")
    try:
        from blob_detector_module import BlobDetector
        import numpy as np

        detector = BlobDetector()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        detections, processed = detector.detect_blobs(test_image)

        if processed is not None:
            print("  ✓ Blob detection working")
            tests_passed += 1
        else:
            print("  ✗ Blob detection failed")

    except Exception as e:
        print(f"  ✗ Blob detector test failed: {e}")

    total_tests += 1

    # Test hough circles
    print("\nTesting Hough Circles...")
    try:
        from hough_circles import HoughCirclesDetector
        import numpy as np

        detector = HoughCirclesDetector()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        circles, processed = detector.detect_circles(test_image)

        if processed is not None:
            print("  ✓ Hough circles detection working")
            tests_passed += 1
        else:
            print("  ✗ Hough circles detection failed")

    except Exception as e:
        print(f"  ✗ Hough circles test failed: {e}")

    total_tests += 1

    # Test statistical features
    print("\nTesting Statistical Features...")
    try:
        from statistical_features_module import StatisticalDetector
        import numpy as np

        detector = StatisticalDetector()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        results, processed = detector.analyze_frame(test_image)

        if results is not None and processed is not None:
            print("  ✓ Statistical analysis working")
            tests_passed += 1
        else:
            print("  ✗ Statistical analysis failed")

    except Exception as e:
        print(f"  ✗ Statistical features test failed: {e}")

    total_tests += 1

    print(f"\nFunctionality Tests: {tests_passed}/{total_tests} passed")

    return tests_passed == total_tests


def test_file_existence():
    """Test that all required files exist."""
    print("\n\nTesting File Existence")
    print("=" * 40)

    required_files = [
        "bmp_video_emulator.py",
        "blob_detection_emulator.py",
        "scratch_detection_emulator.py",
        "morphological_features_emulator.py",
        "blob_detector_module.py",
        "morphological_features_module.py",
        "hough_circles.py",
        "hough_lines.py",
        "pylon_grabber.py",
        "show_emulators.py",
        "test_morphological_features.py",
        "create_morphological_test_image.py"
    ]

    missing_files = []
    existing_files = []

    for file in required_files:
        if Path(file).exists():
            existing_files.append(file)
            print(f"  ✓ {file}")
        else:
            missing_files.append(file)
            print(f"  ✗ {file}")

    print(f"\nFile Check: {len(existing_files)}/{len(required_files)} files found")

    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")

    return len(missing_files) == 0


def main():
    """Run all tests."""
    print("Comprehensive Emulator Test Suite")
    print("=" * 50)

    try:
        # Test imports
        import_success = test_all_emulators()

        # Test file existence
        files_success = test_file_existence()

        # Test basic functionality
        func_success = test_basic_functionality()

        # Overall summary
        print("\n" + "=" * 50)
        print("OVERALL SUMMARY")
        print("=" * 50)
        print(f"Import Tests:        {'PASS' if import_success else 'FAIL'}")
        print(f"File Tests:          {'PASS' if files_success else 'FAIL'}")
        print(f"Functionality Tests: {'PASS' if func_success else 'FAIL'}")

        if import_success and files_success and func_success:
            print("\n🎉 ALL TESTS PASSED! 🎉")
            print("The emulator system is ready for use.")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("Check the output above for details.")

        return import_success and files_success and func_success

    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
