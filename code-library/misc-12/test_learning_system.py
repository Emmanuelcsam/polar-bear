#!/usr/bin/env python3
"""
Test script for the Integrated Learning System
Verifies that all components work correctly together.
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path

# Import our modules
from geometric_core_detector import GeometricCoreDetector, DetectionResult
from circle_overlay import UltraFastCircleOverlay


def test_geometric_detector():
    """Test the geometric core detector"""
    print("Testing Geometric Core Detector...")
    
    # Create detector
    detector = GeometricCoreDetector()
    
    # Create a test image with a circle
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    center = (320, 240)
    radius = 50
    
    # Draw a circle
    cv2.circle(test_image, center, radius, (255, 255, 255), -1)
    
    # Test geometric detection
    result = detector.geometric_detection(test_image)
    
    print(f"Geometric detection result:")
    print(f"  Center: {result.center}")
    print(f"  Radius: {result.radius}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Method: {result.method}")
    
    # Test feature extraction
    features = detector.extract_features(test_image, result)
    print(f"Feature vector shape: {features.shape}")
    
    # Test improved detection
    improved_result = detector.improved_detection(test_image)
    print(f"Improved detection result:")
    print(f"  Center: {improved_result.center}")
    print(f"  Radius: {improved_result.radius}")
    print(f"  Confidence: {improved_result.confidence}")
    print(f"  Method: {improved_result.method}")
    
    print("✓ Geometric detector test passed")
    return True


def test_circle_overlay():
    """Test the circle overlay"""
    print("Testing Circle Overlay...")
    
    # Create circle overlay
    circle_overlay = UltraFastCircleOverlay()
    
    # Test initial state
    print(f"Initial center: {circle_overlay.center}")
    print(f"Initial radius: {circle_overlay.radius}")
    print(f"Initial locked state: {circle_overlay.is_locked}")
    
    # Test movement
    original_center = circle_overlay.center.copy()
    circle_overlay._apply_movement("up")
    circle_overlay._apply_movement("right")
    
    print(f"After movement: {circle_overlay.center}")
    assert circle_overlay.center != original_center, "Movement not working"
    
    # Test resize
    original_radius = circle_overlay.radius
    circle_overlay._apply_movement("larger")
    
    print(f"After resize: {circle_overlay.radius}")
    assert circle_overlay.radius > original_radius, "Resize not working"
    
    # Test lock/unlock
    circle_overlay.is_locked = True
    print(f"Locked state: {circle_overlay.is_locked}")
    
    # Test drawing
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    drawn_frame = circle_overlay.draw_circle(test_frame)
    
    print(f"Drawn frame shape: {drawn_frame.shape}")
    assert drawn_frame.shape == test_frame.shape, "Drawing not working"
    
    print("✓ Circle overlay test passed")
    return True


def test_learning_process():
    """Test the learning process"""
    print("Testing Learning Process...")
    
    # Create detector
    detector = GeometricCoreDetector()
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    center = (320, 240)
    radius = 50
    
    # Draw a circle
    cv2.circle(test_image, center, radius, (255, 255, 255), -1)
    
    # Test learning from manual detection
    manual_center = (300, 220)
    manual_radius = 45
    
    print(f"Training from manual detection: center={manual_center}, radius={manual_radius}")
    
    # Learn from manual detection
    detector.learn_from_manual_detection(test_image, manual_center, manual_radius)
    
    # Check training data
    print(f"Training samples: {len(detector.training_data)}")
    assert len(detector.training_data) > 0, "Training data not saved"
    
    # Test model saving
    detector.save_model()
    detector.save_training_data()
    
    # Check if files were created
    assert os.path.exists(detector.model_path), "Model not saved"
    assert os.path.exists(detector.data_path), "Training data not saved"
    
    print("✓ Learning process test passed")
    return True


def test_feature_extraction():
    """Test feature extraction"""
    print("Testing Feature Extraction...")
    
    # Create detector
    detector = GeometricCoreDetector()
    
    # Create test image
    test_image = np.zeros((480, 640), dtype=np.uint8)
    center = (320, 240)
    radius = 50
    
    # Draw a circle
    cv2.circle(test_image, center, radius, 255, -1)
    
    # Test intensity profile extraction
    profile = detector.feature_extractor.extract_intensity_profile(
        test_image, center, radius
    )
    print(f"Intensity profile shape: {profile.shape}")
    assert profile.shape[0] == 64, "Intensity profile wrong size"
    
    # Test image characteristics
    characteristics = detector.feature_extractor.extract_image_characteristics(
        test_image, center, radius
    )
    print(f"Image characteristics: {characteristics}")
    assert len(characteristics) > 0, "No characteristics extracted"
    
    # Test pixel analysis
    pixel_analysis = detector.feature_extractor.extract_pixel_analysis(
        test_image, center, radius
    )
    print(f"Pixel analysis keys: {list(pixel_analysis.keys())}")
    assert len(pixel_analysis) > 0, "No pixel analysis extracted"
    
    print("✓ Feature extraction test passed")
    return True


def test_integration():
    """Test integration of all components"""
    print("Testing Integration...")
    
    # Create components
    detector = GeometricCoreDetector()
    circle_overlay = UltraFastCircleOverlay()
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    center = (320, 240)
    radius = 50
    
    # Draw a circle
    cv2.circle(test_image, center, radius, (255, 255, 255), -1)
    
    # Test manual detection
    manual_result = DetectionResult(
        center=tuple(circle_overlay.center),
        radius=circle_overlay.radius,
        confidence=1.0 if circle_overlay.is_locked else 0.5,
        method="manual",
        timestamp=time.time()
    )
    
    # Test automatic detection
    auto_result = detector.geometric_detection(test_image)
    
    # Test improved detection
    improved_result = detector.improved_detection(test_image)
    
    # Verify all results have required fields
    for result in [manual_result, auto_result, improved_result]:
        assert hasattr(result, 'center'), "Result missing center"
        assert hasattr(result, 'radius'), "Result missing radius"
        assert hasattr(result, 'confidence'), "Result missing confidence"
        assert hasattr(result, 'method'), "Result missing method"
        assert hasattr(result, 'timestamp'), "Result missing timestamp"
    
    print("✓ Integration test passed")
    return True


def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        "core_detection_model.pth",
        "detection_data.pkl",
        "circle_config.json"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed test file: {file}")


def main():
    """Run all tests"""
    print("Running Integrated Learning System Tests")
    print("=" * 50)
    
    try:
        # Run tests
        test_geometric_detector()
        test_circle_overlay()
        test_feature_extraction()
        test_learning_process()
        test_integration()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("The integrated learning system is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test files
        cleanup_test_files()
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 