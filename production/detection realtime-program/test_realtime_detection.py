#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Real-Time Defect Detection System

This test suite covers all components of the integrated system:
- EnhancedPylonGrabber
- RealTimeDetector  
- RealTimeController
- DetectionConfig
- DetectionResult

Run with: python test_realtime_detection.py
"""

import unittest
import tempfile
import shutil
import time
import threading
import queue
import numpy as np
import cv2
from pathlib import Path
import json
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main system components
try:
    from realtime_defect_detection import (
        DetectionConfig, DetectionResult, EnhancedPylonGrabber,
        RealTimeDetector, RealTimeController
    )
    from detection import OmniFiberAnalyzer, OmniConfig
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import system components: {e}")
    SYSTEM_AVAILABLE = False


class TestDetectionConfig(unittest.TestCase):
    """Test DetectionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DetectionConfig()
        
        self.assertIsNone(config.reference_image_path)
        self.assertEqual(config.anomaly_threshold, 2.0)
        self.assertEqual(config.ssim_threshold, 0.8)
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertTrue(config.enable_fast_mode)
        self.assertEqual(config.resize_factor, 1.0)
        self.assertEqual(config.max_processing_time, 0.1)
        self.assertEqual(config.min_defect_area, 25)
        self.assertEqual(config.max_defect_area, 5000)
        self.assertTrue(config.enable_visualization)
        self.assertTrue(config.save_results)
        self.assertEqual(config.output_dir, "realtime_output")
        self.assertEqual(config.exposure_time, 10000)
        self.assertEqual(config.gain, 0)
        self.assertEqual(config.buffer_size, 5)
        self.assertEqual(config.grab_strategy, "LatestImageOnly")
        self.assertEqual(config.processing_fps, 10.0)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DetectionConfig(
            reference_image_path="test.jpg",
            anomaly_threshold=1.5,
            ssim_threshold=0.9,
            confidence_threshold=0.7,
            enable_fast_mode=False,
            resize_factor=0.5,
            max_processing_time=0.05,
            min_defect_area=10,
            max_defect_area=1000,
            enable_visualization=False,
            save_results=False,
            output_dir="test_output",
            exposure_time=5000,
            gain=10,
            buffer_size=10,
            grab_strategy="OneByOne",
            processing_fps=20.0
        )
        
        self.assertEqual(config.reference_image_path, "test.jpg")
        self.assertEqual(config.anomaly_threshold, 1.5)
        self.assertEqual(config.ssim_threshold, 0.9)
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertFalse(config.enable_fast_mode)
        self.assertEqual(config.resize_factor, 0.5)
        self.assertEqual(config.max_processing_time, 0.05)
        self.assertEqual(config.min_defect_area, 10)
        self.assertEqual(config.max_defect_area, 1000)
        self.assertFalse(config.enable_visualization)
        self.assertFalse(config.save_results)
        self.assertEqual(config.output_dir, "test_output")
        self.assertEqual(config.exposure_time, 5000)
        self.assertEqual(config.gain, 10)
        self.assertEqual(config.buffer_size, 10)
        self.assertEqual(config.grab_strategy, "OneByOne")
        self.assertEqual(config.processing_fps, 20.0)


class TestDetectionResult(unittest.TestCase):
    """Test DetectionResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a detection result."""
        timestamp = time.time()
        defect_regions = [
            {'id': 0, 'bbox': [10, 20, 30, 40], 'area': 100.0, 'confidence': 0.8, 'type': 'anomaly'},
            {'id': 1, 'bbox': [50, 60, 20, 30], 'area': 50.0, 'confidence': 0.6, 'type': 'anomaly'}
        ]
        
        result = DetectionResult(
            timestamp=timestamp,
            is_anomalous=True,
            confidence=0.75,
            ssim_score=0.65,
            defect_count=2,
            defect_regions=defect_regions,
            processing_time=0.05,
            frame_id=123
        )
        
        self.assertEqual(result.timestamp, timestamp)
        self.assertTrue(result.is_anomalous)
        self.assertEqual(result.confidence, 0.75)
        self.assertEqual(result.ssim_score, 0.65)
        self.assertEqual(result.defect_count, 2)
        self.assertEqual(len(result.defect_regions), 2)
        self.assertEqual(result.processing_time, 0.05)
        self.assertEqual(result.frame_id, 123)
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = DetectionResult(
            timestamp=1234567890.123,
            is_anomalous=False,
            confidence=0.9,
            ssim_score=0.95,
            defect_count=0,
            defect_regions=[],
            processing_time=0.02,
            frame_id=456
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['timestamp'], 1234567890.123)
        self.assertFalse(result_dict['is_anomalous'])
        self.assertEqual(result_dict['confidence'], 0.9)
        self.assertEqual(result_dict['ssim_score'], 0.95)
        self.assertEqual(result_dict['defect_count'], 0)
        self.assertEqual(result_dict['defect_regions'], [])
        self.assertEqual(result_dict['processing_time'], 0.02)
        self.assertEqual(result_dict['frame_id'], 456)
    
    def test_result_json_serialization(self):
        """Test JSON serialization of result."""
        defect_regions = [
            {'id': 0, 'bbox': [10, 20, 30, 40], 'area': 100.0, 'confidence': 0.8, 'type': 'anomaly'}
        ]
        
        result = DetectionResult(
            timestamp=time.time(),
            is_anomalous=True,
            confidence=0.8,
            ssim_score=0.7,
            defect_count=1,
            defect_regions=defect_regions,
            processing_time=0.03,
            frame_id=789
        )
        
        # Convert to dict and back to JSON
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        parsed_dict = json.loads(json_str)
        
        # Verify all fields are preserved
        self.assertEqual(parsed_dict['is_anomalous'], True)
        self.assertEqual(parsed_dict['confidence'], 0.8)
        self.assertEqual(parsed_dict['ssim_score'], 0.7)
        self.assertEqual(parsed_dict['defect_count'], 1)
        self.assertEqual(parsed_dict['frame_id'], 789)
        self.assertEqual(len(parsed_dict['defect_regions']), 1)


class TestRealTimeDetector(unittest.TestCase):
    """Test RealTimeDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if not SYSTEM_AVAILABLE:
            raise unittest.SkipTest("System components not available")
        
        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
        cls.reference_image_path = os.path.join(cls.temp_dir, "reference.jpg")
        
        # Create a test reference image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(cls.reference_image_path, test_image)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up for each test."""
        self.config = DetectionConfig(
            reference_image_path=self.reference_image_path,
            anomaly_threshold=2.0,
            ssim_threshold=0.8,
            confidence_threshold=0.5,
            enable_fast_mode=True,
            resize_factor=1.0,
            min_defect_area=25,
            max_defect_area=5000,
            enable_visualization=False,
            save_results=False,
            output_dir=self.temp_dir
        )
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = RealTimeDetector(self.config)
        
        self.assertIsNotNone(detector.analyzer)
        self.assertIsNotNone(detector.reference_image)
        self.assertIsNotNone(detector.reference_gray)
        self.assertEqual(detector.frame_count, 0)
        self.assertEqual(detector.total_processing_time, 0)
        self.assertIsNone(detector.last_result)
    
    def test_load_reference_image(self):
        """Test reference image loading."""
        detector = RealTimeDetector(self.config)
        
        # Check that reference image was loaded
        self.assertIsNotNone(detector.reference_image)
        self.assertIsNotNone(detector.reference_gray)
        
        # Check image dimensions
        self.assertEqual(detector.reference_image.shape[2], 3)  # BGR
        self.assertEqual(len(detector.reference_gray.shape), 2)  # Grayscale
    
    def test_build_reference_model(self):
        """Test reference model building."""
        detector = RealTimeDetector(self.config)
        
        # Check that reference model was built
        self.assertIsNotNone(detector.analyzer.reference_model)
        self.assertIn('features', detector.analyzer.reference_model)
        self.assertIn('statistical_model', detector.analyzer.reference_model)
        self.assertIn('archetype_image', detector.analyzer.reference_model)
    
    def test_prepare_frame(self):
        """Test frame preparation."""
        detector = RealTimeDetector(self.config)
        
        # Test frame with resize
        test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        prepared_frame = detector._prepare_frame(test_frame)
        
        # Should be same size since resize_factor is 1.0
        self.assertEqual(prepared_frame.shape, test_frame.shape)
        
        # Test with different resize factor
        detector.config.resize_factor = 0.5
        prepared_frame = detector._prepare_frame(test_frame)
        expected_height = int(240 * 0.5)
        expected_width = int(320 * 0.5)
        self.assertEqual(prepared_frame.shape, (expected_height, expected_width, 3))
    
    def test_calculate_ssim(self):
        """Test SSIM calculation."""
        detector = RealTimeDetector(self.config)
        
        # Create identical images (should have SSIM = 1.0)
        img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = img1.copy()
        
        ssim_score = detector._calculate_ssim(img1, img2)
        self.assertGreater(ssim_score, 0.9)  # Should be very close to 1.0
        
        # Create different images (should have lower SSIM)
        img3 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        ssim_score = detector._calculate_ssim(img1, img3)
        self.assertLess(ssim_score, 0.9)  # Should be lower
    
    def test_fast_detection(self):
        """Test fast detection mode."""
        detector = RealTimeDetector(self.config)
        
        # Create test frame similar to reference
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = detector._fast_detection(test_frame, frame_id=1)
        
        # Check result structure
        self.assertIsInstance(result, DetectionResult)
        self.assertIsInstance(result.timestamp, float)
        self.assertIsInstance(result.is_anomalous, bool)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.ssim_score, float)
        self.assertIsInstance(result.defect_count, int)
        self.assertIsInstance(result.defect_regions, list)
        self.assertEqual(result.frame_id, 1)
        
        # Check value ranges
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreaterEqual(result.ssim_score, 0.0)
        self.assertLessEqual(result.ssim_score, 1.0)
        self.assertGreaterEqual(result.defect_count, 0)
    
    def test_detect_defects(self):
        """Test main detection method."""
        detector = RealTimeDetector(self.config)
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = detector.detect_defects(test_frame, frame_id=2)
        
        # Check result
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.frame_id, 2)
        self.assertGreater(result.processing_time, 0.0)
        
        # Check statistics update
        self.assertEqual(detector.frame_count, 1)
        self.assertGreater(detector.total_processing_time, 0.0)
        self.assertIsNotNone(detector.last_result)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        detector = RealTimeDetector(self.config)
        
        # Process a few frames
        for i in range(3):
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detector.detect_defects(test_frame, frame_id=i)
        
        stats = detector.get_statistics()
        
        self.assertEqual(stats['frames_processed'], 3)
        self.assertGreater(stats['avg_processing_time'], 0.0)
        self.assertGreater(stats['fps'], 0.0)
        self.assertTrue(stats['reference_loaded'])
        self.assertIsNotNone(stats['last_result'])
    
    def test_visualize_result(self):
        """Test result visualization."""
        detector = RealTimeDetector(self.config)
        
        # Create test frame and result
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        defect_regions = [
            {'id': 0, 'bbox': [10, 20, 30, 40], 'area': 100.0, 'confidence': 0.8, 'type': 'anomaly'}
        ]
        
        result = DetectionResult(
            timestamp=time.time(),
            is_anomalous=True,
            confidence=0.8,
            ssim_score=0.7,
            defect_count=1,
            defect_regions=defect_regions,
            processing_time=0.05,
            frame_id=1
        )
        
        vis_frame = detector.visualize_result(test_frame, result)
        
        # Check visualization
        self.assertEqual(vis_frame.shape, test_frame.shape)
        self.assertEqual(vis_frame.dtype, test_frame.dtype)


class TestEnhancedPylonGrabber(unittest.TestCase):
    """Test EnhancedPylonGrabber class."""
    
    def setUp(self):
        """Set up for each test."""
        self.config = DetectionConfig(
            buffer_size=3,
            exposure_time=10000,
            gain=0,
            grab_strategy="LatestImageOnly"
        )
    
    def test_grabber_initialization(self):
        """Test grabber initialization."""
        grabber = EnhancedPylonGrabber(self.config)
        
        self.assertEqual(grabber.config, self.config)
        self.assertIsNone(grabber.camera)
        self.assertIsNone(grabber.latest_frame)
        self.assertEqual(grabber.frame_count, 0)
        self.assertEqual(grabber.dropped_frames, 0)
        self.assertEqual(grabber.error_count, 0)
        self.assertFalse(grabber.is_running.is_set())
        self.assertFalse(grabber.is_initialized.is_set())
    
    def test_grabber_statistics(self):
        """Test grabber statistics."""
        grabber = EnhancedPylonGrabber(self.config)
        
        # Simulate some statistics
        grabber.frame_count = 100
        grabber.dropped_frames = 5
        grabber.error_count = 2
        grabber.current_fps = 25.5
        grabber.last_error = "Test error"
        
        stats = grabber.get_statistics()
        
        self.assertEqual(stats['fps'], 25.5)
        self.assertEqual(stats['total_frames'], 100)
        self.assertEqual(stats['dropped_frames'], 5)
        self.assertEqual(stats['error_count'], 2)
        self.assertEqual(stats['last_error'], "Test error")
        self.assertEqual(stats['buffer_size'], 0)  # Empty buffer
        self.assertFalse(stats['is_running'])
        self.assertFalse(stats['is_initialized'])
    
    def test_grabber_health_check(self):
        """Test grabber health check."""
        grabber = EnhancedPylonGrabber(self.config)
        
        # Should be unhealthy initially
        self.assertFalse(grabber.is_healthy())
        
        # Simulate healthy state
        grabber.is_running.set()
        grabber.frame_metadata = {'timestamp': time.time()}
        grabber.error_count = 0
        
        self.assertTrue(grabber.is_healthy())
    
    def test_grabber_stop(self):
        """Test grabber stop method."""
        grabber = EnhancedPylonGrabber(self.config)
        
        # Should not be running initially
        self.assertFalse(grabber.is_running.is_set())
        
        # Start and stop
        grabber.is_running.set()
        self.assertTrue(grabber.is_running.is_set())
        
        grabber.stop()
        self.assertFalse(grabber.is_running.is_set())


class TestRealTimeController(unittest.TestCase):
    """Test RealTimeController class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if not SYSTEM_AVAILABLE:
            raise unittest.SkipTest("System components not available")
        
        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
        cls.reference_image_path = os.path.join(cls.temp_dir, "reference.jpg")
        
        # Create a test reference image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(cls.reference_image_path, test_image)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up for each test."""
        self.config = DetectionConfig(
            reference_image_path=self.reference_image_path,
            enable_visualization=False,  # Disable for testing
            save_results=False,  # Disable for testing
            output_dir=self.temp_dir
        )
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        controller = RealTimeController(self.config)
        
        self.assertEqual(controller.config, self.config)
        self.assertIsNone(controller.frame_grabber)
        self.assertIsNone(controller.detector)
        self.assertEqual(controller.stats['frames_captured'], 0)
        self.assertEqual(controller.stats['frames_processed'], 0)
        self.assertEqual(controller.stats['defects_detected'], 0)
        self.assertEqual(controller.stats['average_processing_time'], 0)
        self.assertIsNone(controller.stats['start_time'])
        self.assertIsNone(controller.defect_alert_callback)
    
    def test_setup_logging(self):
        """Test logging setup."""
        controller = RealTimeController(self.config)
        
        # Check that output directory was created
        self.assertTrue(controller.output_dir.exists())
        
        # Check that logger was created
        self.assertIsNotNone(controller.logger)
    
    def test_register_defect_alert(self):
        """Test defect alert registration."""
        controller = RealTimeController(self.config)
        
        def test_alert(result, frame):
            pass
        
        controller.register_defect_alert(test_alert)
        self.assertEqual(controller.defect_alert_callback, test_alert)
    
    def test_add_stats_overlay(self):
        """Test statistics overlay."""
        controller = RealTimeController(self.config)
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        original_frame = test_frame.copy()
        
        # Add some statistics
        controller.stats['frames_captured'] = 100
        controller.stats['frames_processed'] = 95
        controller.stats['defects_detected'] = 5
        controller.stats['average_processing_time'] = 0.05
        
        # Add overlay
        controller._add_stats_overlay(test_frame)
        
        # Frame should be modified
        self.assertFalse(np.array_equal(test_frame, original_frame))
    
    def test_save_result(self):
        """Test result saving."""
        controller = RealTimeController(self.config)
        controller.config.save_results = True
        
        # Initialize detector properly
        controller.detector = RealTimeDetector(self.config)
        
        # Create test result
        defect_regions = [
            {'id': 0, 'bbox': [10, 20, 30, 40], 'area': 100.0, 'confidence': 0.8, 'type': 'anomaly'}
        ]
        
        result = DetectionResult(
            timestamp=time.time(),
            is_anomalous=True,
            confidence=0.8,
            ssim_score=0.7,
            defect_count=1,
            defect_regions=defect_regions,
            processing_time=0.05,
            frame_id=1
        )
        
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metadata = {'test': 'data'}
        
        # Save result
        controller._save_result(result, test_frame, metadata)
        
        # Check that files were created
        timestamp = int(result.timestamp)
        result_file = controller.output_dir / f"result_{timestamp}_{result.frame_id:06d}.json"
        frame_file = controller.output_dir / f"defect_{timestamp}_{result.frame_id:06d}.jpg"
        vis_file = controller.output_dir / f"defect_vis_{timestamp}_{result.frame_id:06d}.jpg"
        
        self.assertTrue(result_file.exists())
        self.assertTrue(frame_file.exists())
        self.assertTrue(vis_file.exists())
    
    def test_print_statistics(self):
        """Test statistics printing."""
        controller = RealTimeController(self.config)
        
        # Set some statistics
        controller.stats['start_time'] = time.time() - 10.0  # 10 seconds ago
        controller.stats['frames_captured'] = 100
        controller.stats['frames_processed'] = 95
        controller.stats['defects_detected'] = 5
        controller.stats['average_processing_time'] = 0.05
        
        # Mock frame grabber and detector
        controller.frame_grabber = type('MockGrabber', (), {
            'get_statistics': lambda self: {'fps': 25.0}
        })()
        
        controller.detector = type('MockDetector', (), {
            'get_statistics': lambda self: {'fps': 20.0}
        })()
        
        # This should not raise an exception
        controller._print_statistics()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if not SYSTEM_AVAILABLE:
            raise unittest.SkipTest("System components not available")
        
        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
        cls.reference_image_path = os.path.join(cls.temp_dir, "reference.jpg")
        
        # Create a test reference image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(cls.reference_image_path, test_image)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_detector_with_reference(self):
        """Test detector with actual reference image."""
        config = DetectionConfig(
            reference_image_path=self.reference_image_path,
            enable_visualization=False,
            save_results=False,
            output_dir=self.temp_dir
        )
        
        detector = RealTimeDetector(config)
        
        # Test with similar image
        similar_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_defects(similar_frame, frame_id=1)
        
        self.assertIsInstance(result, DetectionResult)
        self.assertGreater(result.ssim_score, 0.0)
        self.assertLess(result.ssim_score, 1.0)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = DetectionConfig(
            reference_image_path=self.reference_image_path,
            anomaly_threshold=1.5,
            ssim_threshold=0.9,
            confidence_threshold=0.7
        )
        
        self.assertEqual(config.anomaly_threshold, 1.5)
        self.assertEqual(config.ssim_threshold, 0.9)
        self.assertEqual(config.confidence_threshold, 0.7)
    
    def test_result_serialization(self):
        """Test result serialization and deserialization."""
        defect_regions = [
            {'id': 0, 'bbox': [10, 20, 30, 40], 'area': 100.0, 'confidence': 0.8, 'type': 'anomaly'},
            {'id': 1, 'bbox': [50, 60, 20, 30], 'area': 50.0, 'confidence': 0.6, 'type': 'anomaly'}
        ]
        
        result = DetectionResult(
            timestamp=time.time(),
            is_anomalous=True,
            confidence=0.75,
            ssim_score=0.65,
            defect_count=2,
            defect_regions=defect_regions,
            processing_time=0.05,
            frame_id=123
        )
        
        # Serialize to JSON
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        
        # Deserialize
        parsed_dict = json.loads(json_str)
        
        # Verify all fields
        self.assertEqual(parsed_dict['is_anomalous'], True)
        self.assertEqual(parsed_dict['confidence'], 0.75)
        self.assertEqual(parsed_dict['ssim_score'], 0.65)
        self.assertEqual(parsed_dict['defect_count'], 2)
        self.assertEqual(parsed_dict['frame_id'], 123)
        self.assertEqual(len(parsed_dict['defect_regions']), 2)


def run_performance_test():
    """Run a simple performance test."""
    print("\n" + "="*60)
    print("PERFORMANCE TEST")
    print("="*60)
    
    if not SYSTEM_AVAILABLE:
        print("System components not available - skipping performance test")
        return
    
    # Create test environment
    temp_dir = tempfile.mkdtemp()
    reference_image_path = os.path.join(temp_dir, "reference.jpg")
    
    # Create test reference image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(reference_image_path, test_image)
    
    try:
        # Test detector performance
        config = DetectionConfig(
            reference_image_path=reference_image_path,
            enable_visualization=False,
            save_results=False,
            output_dir=temp_dir
        )
        
        detector = RealTimeDetector(config)
        
        # Test processing speed
        test_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
        
        start_time = time.time()
        for i, frame in enumerate(test_frames):
            result = detector.detect_defects(frame, frame_id=i)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_frames)
        fps = 1.0 / avg_time
        
        print(f"Processed {len(test_frames)} frames in {total_time:.3f}s")
        print(f"Average processing time: {avg_time*1000:.2f}ms")
        print(f"Processing FPS: {fps:.1f}")
        
        # Test statistics
        stats = detector.get_statistics()
        print(f"Detector statistics: {stats}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("🧪 Real-Time Defect Detection System Tests")
    print("="*50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDetectionConfig,
        TestDetectionResult,
        TestRealTimeDetector,
        TestEnhancedPylonGrabber,
        TestRealTimeController,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Run performance test
    run_performance_test()
    
    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 