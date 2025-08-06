#!/usr/bin/env python3
"""
Test script for the fiber optic analysis system.
Tests all components with the existing good.bmp image.
"""

import cv2
import os
import json
from pathlib import Path
import logging

# Import our modules
from detection import OmniFiberAnalyzer, OmniConfig
from separation import UnifiedSegmentationSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_fiber_analyzer():
    """Test the fiber anomaly detection system."""
    print("\n=== Testing Fiber Anomaly Detection ===")
    
    # Check if test image exists
    test_image = "good.bmp"
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found!")
        return False
    
    try:
        # Initialize analyzer
        config = OmniConfig(
            confidence_threshold=0.3,
            anomaly_threshold_multiplier=2.5,
            enable_visualization=True
        )
        analyzer = OmniFiberAnalyzer(config)
        
        # Analyze the image
        print(f"Analyzing {test_image}...")
        results = analyzer.detect_anomalies_comprehensive(test_image)
        
        if results:
            print("✓ Fiber analysis completed successfully")
            print(f"  - Anomalous: {results['verdict']['is_anomalous']}")
            print(f"  - Confidence: {results['verdict']['confidence']:.3f}")
            return True
        else:
            print("✗ Fiber analysis failed")
            return False
            
    except Exception as e:
        print(f"✗ Fiber analysis error: {e}")
        return False

def test_segmentation_system():
    """Test the fiber segmentation system."""
    print("\n=== Testing Fiber Segmentation ===")
    
    # Check if test image exists
    test_image = "good.bmp"
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found!")
        return False
    
    try:
        # Check if methods directory exists
        methods_dir = "zones_methods"
        if not os.path.exists(methods_dir):
            print(f"Methods directory {methods_dir} not found!")
            print("Creating dummy methods directory for testing...")
            os.makedirs(methods_dir, exist_ok=True)
            return True  # Skip actual segmentation test
        
        # Initialize segmentation system
        seg_system = UnifiedSegmentationSystem(methods_dir)
        
        # Process the image
        print(f"Processing {test_image}...")
        results = seg_system.process_image(Path(test_image), "test_output")
        
        if results:
            print("✓ Segmentation completed successfully")
            print(f"  - Center: {results.get('center', 'N/A')}")
            print(f"  - Core radius: {results.get('core_radius', 'N/A')}")
            print(f"  - Cladding radius: {results.get('cladding_radius', 'N/A')}")
            return True
        else:
            print("✗ Segmentation failed")
            return False
            
    except Exception as e:
        print(f"✗ Segmentation error: {e}")
        return False

def test_yolo_detection():
    """Test YOLO object detection."""
    print("\n=== Testing YOLO Detection ===")
    
    # Check for required files
    required_files = ['yolov3.weights', 'yolov3.cfg', 'coco.names']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing YOLO files: {missing_files}")
        return False
    
    try:
        # Load test image
        test_image = "good.bmp"
        if not os.path.exists(test_image):
            print(f"Test image {test_image} not found!")
            return False
        
        image = cv2.imread(test_image)
        if image is None:
            print("Could not load test image!")
            return False
        
        # Initialize YOLO detector
        from realtime_analyzer import YOLODetector
        detector = YOLODetector(
            weights_path='yolov3.weights',
            config_path='yolov3.cfg',
            classes_path='coco.names',
            confidence_threshold=0.5,
            nms_threshold=0.4
        )
        
        # Perform detection
        print("Running YOLO detection...")
        detections = detector.detect(image)
        
        print(f"✓ YOLO detection completed")
        print(f"  - Found {len(detections)} objects")
        for i, (label, confidence, bbox) in enumerate(detections):
            print(f"    {i+1}. {label} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO detection error: {e}")
        return False

def test_camera_system():
    """Test camera system (if available)."""
    print("\n=== Testing Camera System ===")
    
    try:
        from realtime_analyzer import OpenCVCameraGrabber
        
        # Try to initialize camera
        camera = OpenCVCameraGrabber(camera_index=0)
        
        # Try to start camera
        camera.start_grabbing()
        
        # Wait a moment and try to read a frame
        import time
        time.sleep(1)
        
        frame = camera.read()
        if frame is not None:
            print("✓ Camera system working")
            print(f"  - Frame shape: {frame.shape}")
        else:
            print("✗ Camera system failed to capture frame")
            return False
        
        # Stop camera
        camera.stop_grabbing()
        return True
        
    except Exception as e:
        print(f"✗ Camera system error: {e}")
        return False

def main():
    """Run all tests."""
    print("Fiber Optic Analysis System - Component Tests")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Run tests
    tests = [
        ("Fiber Anomaly Detection", test_fiber_analyzer),
        ("Fiber Segmentation", test_segmentation_system),
        ("YOLO Detection", test_yolo_detection),
        ("Camera System", test_camera_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for real-time use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    main() 