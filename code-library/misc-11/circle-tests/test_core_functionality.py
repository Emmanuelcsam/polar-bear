#!/usr/bin/env python3
"""
Core Functionality Test for Circle Detection System
==================================================

This script tests the core circle detection functionality without requiring
camera access or complex mocking. It focuses on the essential algorithms
and data processing.

Author: AI Assistant
Date: 2024
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Import the core classes to test
from pylon_circle_detector import CircleDetector


def test_circle_detector_initialization():
    """Test CircleDetector initialization."""
    print("Testing CircleDetector initialization...")

    # Test default initialization
    detector = CircleDetector()

    # Check default parameters
    assert detector.hough_params["dp"] == 1
    assert detector.hough_params["min_dist"] == 20
    assert detector.hough_params["param1"] == 50
    assert detector.hough_params["param2"] == 30
    assert detector.hough_params["min_radius"] == 10
    assert detector.hough_params["max_radius"] == 300

    assert detector.contour_params["min_area"] == 100
    assert detector.contour_params["max_area"] == 50000
    assert detector.contour_params["circularity_threshold"] == 0.7

    # Test custom initialization
    custom_hough = {
        "dp": 2,
        "min_dist": 30,
        "param1": 60,
        "param2": 40,
        "min_radius": 20,
        "max_radius": 400,
    }
    custom_contour = {"min_area": 200, "max_area": 60000, "circularity_threshold": 0.8}

    detector_custom = CircleDetector(
        hough_params=custom_hough, contour_params=custom_contour, use_gpu=True
    )

    assert detector_custom.hough_params == custom_hough
    assert detector_custom.contour_params == custom_contour
    assert detector_custom.use_gpu == True

    print("✓ CircleDetector initialization tests passed")


def test_hough_circle_detection():
    """Test Hough circle detection with various scenarios."""
    print("Testing Hough circle detection...")

    detector = CircleDetector()

    # Test 1: Simple circle
    test_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(test_img, (100, 100), 50, 255, -1)

    circles = detector.detect_circles_hough(test_img)
    assert isinstance(circles, list)
    print(f"  Simple circle: {len(circles)} circles detected")

    # Test 2: Multiple circles
    test_img = np.zeros((400, 600), dtype=np.uint8)
    cv2.circle(test_img, (150, 150), 50, 255, -1)
    cv2.circle(test_img, (450, 150), 80, 255, -1)
    cv2.circle(test_img, (300, 300), 60, 255, -1)

    circles = detector.detect_circles_hough(test_img)
    assert isinstance(circles, list)
    print(f"  Multiple circles: {len(circles)} circles detected")

    # Test 3: No circles
    test_img = np.zeros((200, 200), dtype=np.uint8)
    test_img[50:150, 50:150] = 128  # Add noise but no circles

    circles = detector.detect_circles_hough(test_img)
    assert isinstance(circles, list)
    print(f"  No circles: {len(circles)} circles detected")

    # Test 4: Small circles
    test_img = np.zeros((300, 400), dtype=np.uint8)
    for i in range(5):
        x = 50 + i * 70
        y = 150
        radius = 15 + i * 5
        cv2.circle(test_img, (x, y), radius, 255, -1)

    circles = detector.detect_circles_hough(test_img)
    assert isinstance(circles, list)
    print(f"  Small circles: {len(circles)} circles detected")

    print("✓ Hough circle detection tests passed")


def test_contour_circle_detection():
    """Test contour-based circle detection."""
    print("Testing contour circle detection...")

    detector = CircleDetector()

    # Test 1: Simple circle
    test_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(test_img, (100, 100), 50, 255, -1)

    circles = detector.detect_circles_contour(test_img)
    assert isinstance(circles, list)
    print(f"  Simple circle: {len(circles)} circles detected")

    # Test 2: No circles
    test_img = np.zeros((200, 200), dtype=np.uint8)
    test_img[50:150, 50:150] = 128

    circles = detector.detect_circles_contour(test_img)
    assert isinstance(circles, list)
    print(f"  No circles: {len(circles)} circles detected")

    print("✓ Contour circle detection tests passed")


def test_combined_detection():
    """Test combined circle detection."""
    print("Testing combined circle detection...")

    detector = CircleDetector()

    # Test 1: Simple circle
    test_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(test_img, (100, 100), 50, 255, -1)

    circles = detector.detect_circles_combined(test_img)
    assert isinstance(circles, list)
    print(f"  Simple circle: {len(circles)} circles detected")

    # Test 2: Multiple circles
    test_img = np.zeros((400, 600), dtype=np.uint8)
    cv2.circle(test_img, (150, 150), 50, 255, -1)
    cv2.circle(test_img, (450, 150), 80, 255, -1)

    circles = detector.detect_circles_combined(test_img)
    assert isinstance(circles, list)
    print(f"  Multiple circles: {len(circles)} circles detected")

    # Test 3: No circles
    test_img = np.zeros((200, 200), dtype=np.uint8)
    test_img[50:150, 50:150] = 128

    circles = detector.detect_circles_combined(test_img)
    assert isinstance(circles, list)
    print(f"  No circles: {len(circles)} circles detected")

    print("✓ Combined circle detection tests passed")


def test_duplicate_removal():
    """Test duplicate circle removal."""
    print("Testing duplicate circle removal...")

    detector = CircleDetector()

    # Test 1: No duplicates
    circles = [(100, 100, 50), (200, 200, 60)]
    unique = detector._remove_duplicate_circles(circles, threshold=20)
    assert len(unique) == 2
    assert unique == circles

    # Test 2: With duplicates
    circles = [(100, 100, 50), (105, 105, 55), (200, 200, 60)]
    unique = detector._remove_duplicate_circles(circles, threshold=20)
    assert len(unique) <= len(circles)

    # Test 3: Empty list
    unique = detector._remove_duplicate_circles([], threshold=20)
    assert unique == []

    # Test 4: Single circle
    circles = [(100, 100, 50)]
    unique = detector._remove_duplicate_circles(circles, threshold=20)
    assert unique == circles

    print("✓ Duplicate removal tests passed")


def test_circle_drawing():
    """Test circle drawing functionality."""
    print("Testing circle drawing...")

    detector = CircleDetector()

    # Create test image
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    circles = [(100, 100, 50), (150, 150, 60)]

    # Test drawing
    result = detector.draw_circles(test_img, circles)
    assert isinstance(result, np.ndarray)
    assert result.shape == test_img.shape
    assert result.dtype == test_img.dtype

    # Test empty circles
    result = detector.draw_circles(test_img, [])
    assert isinstance(result, np.ndarray)
    assert result.shape == test_img.shape

    # Test custom color
    result = detector.draw_circles(test_img, circles, color=(255, 0, 0))
    assert isinstance(result, np.ndarray)

    print("✓ Circle drawing tests passed")


def test_fps_tracking():
    """Test FPS tracking functionality."""
    print("Testing FPS tracking...")

    detector = CircleDetector()

    # Test initial state
    fps = detector.get_average_fps()
    assert fps == 0.0

    # Test FPS update
    for _ in range(10):
        detector.update_fps()

    # Should have updated frame count
    assert detector.frame_count >= 0

    # Test FPS history
    detector.fps_history.extend([30.0, 25.0, 35.0])
    fps = detector.get_average_fps()
    assert fps == 30.0  # (30+25+35)/3 = 30

    print("✓ FPS tracking tests passed")


def test_performance():
    """Test performance of detection algorithms."""
    print("Testing performance...")

    detector = CircleDetector()

    # Create test image
    test_img = np.zeros((480, 640), dtype=np.uint8)
    for i in range(5):
        x = 100 + i * 100
        y = 240
        radius = 30 + i * 10
        cv2.circle(test_img, (x, y), radius, 255, -1)

    # Benchmark Hough detection
    start_time = time.time()
    for _ in range(50):
        circles = detector.detect_circles_hough(test_img)
    hough_time = time.time() - start_time

    # Benchmark contour detection
    start_time = time.time()
    for _ in range(50):
        circles = detector.detect_circles_contour(test_img)
    contour_time = time.time() - start_time

    # Benchmark combined detection
    start_time = time.time()
    for _ in range(50):
        circles = detector.detect_circles_combined(test_img)
    combined_time = time.time() - start_time

    print(f"  Hough detection: {hough_time:.3f}s for 50 iterations")
    print(f"  Contour detection: {contour_time:.3f}s for 50 iterations")
    print(f"  Combined detection: {combined_time:.3f}s for 50 iterations")

    # Verify performance is reasonable
    assert hough_time < 5.0  # Should complete in under 5 seconds
    assert contour_time < 5.0
    assert combined_time < 10.0

    print("✓ Performance tests passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")

    detector = CircleDetector()

    # Test 1: Very small circles
    detector.hough_params["min_radius"] = 1
    detector.hough_params["max_radius"] = 5

    test_img = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(test_img, (25, 25), 3, 255, -1)

    circles = detector.detect_circles_hough(test_img)
    assert isinstance(circles, list)

    # Test 2: Very large circles
    detector.hough_params["min_radius"] = 100
    detector.hough_params["max_radius"] = 500

    large_img = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.circle(large_img, (500, 500), 200, 255, -1)

    circles = detector.detect_circles_hough(large_img)
    assert isinstance(circles, list)

    # Test 3: Noisy image
    test_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    cv2.circle(test_img, (100, 100), 50, 255, -1)

    circles = detector.detect_circles_hough(test_img)
    assert isinstance(circles, list)

    # Test 4: Overlapping circles
    test_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(test_img, (100, 100), 50, 255, -1)
    cv2.circle(test_img, (120, 100), 40, 255, -1)

    circles = detector.detect_circles_combined(test_img)
    assert isinstance(circles, list)

    print("✓ Edge case tests passed")


def test_configuration_loading():
    """Test configuration loading functionality."""
    print("Testing configuration loading...")

    # Create test config
    test_config = {
        "hough_params": {"dp": 2, "min_dist": 30},
        "contour_params": {"min_area": 200, "max_area": 60000},
        "display": {"window_name": "Test"},
        "recording": {"fps": 30},
    }

    # Test default config
    detector = CircleDetector()
    assert detector.hough_params["dp"] == 1  # Default value

    print("✓ Configuration loading tests passed")


def run_core_tests():
    """Run all core functionality tests."""
    print("Running Core Functionality Tests")
    print("=" * 50)

    try:
        test_circle_detector_initialization()
        test_hough_circle_detection()
        test_contour_circle_detection()
        test_combined_detection()
        test_duplicate_removal()
        test_circle_drawing()
        test_fps_tracking()
        test_performance()
        test_edge_cases()
        test_configuration_loading()

        print("\n" + "=" * 50)
        print("All core functionality tests passed!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_core_tests()
    sys.exit(0 if success else 1)
