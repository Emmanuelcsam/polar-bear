#!/usr/bin/env python3
"""
Comprehensive test suite for realtime_calibration_tool.py
Tests all functions, classes, and methods with rigorous edge cases
"""

import unittest
import numpy as np
import cv2
import json
import tempfile
import os
import sys
import threading
import queue
import time
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import asdict
import tkinter as tk

# Import the module to test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_calibration_tool import (
    CalibrationConfig, InteractiveCalibrator, BatchCalibrator,
    create_test_pattern
)


class TestCalibrationConfig(unittest.TestCase):
    """Test the CalibrationConfig dataclass"""
    
    def test_calibration_config_default_values(self):
        """Test CalibrationConfig default values"""
        config = CalibrationConfig()
        
        # Detection parameters
        self.assertEqual(config.min_area, 100)
        self.assertEqual(config.max_area, 50000)
        self.assertEqual(config.min_perimeter, 50)
        self.assertEqual(config.max_perimeter, 2000)
        self.assertEqual(config.epsilon_factor, 0.02)
        self.assertEqual(config.min_circularity, 0.7)
        self.assertEqual(config.min_convexity, 0.7)
        self.assertEqual(config.min_inertia, 0.5)
        
        # Camera parameters
        self.assertEqual(config.exposure, -5)
        self.assertEqual(config.gain, 10)
        self.assertEqual(config.brightness, 50)
        self.assertEqual(config.contrast, 50)
        
        # Shape filters
        self.assertTrue(config.detect_circles)
        self.assertTrue(config.detect_rectangles)
        self.assertTrue(config.detect_triangles)
        self.assertTrue(config.detect_polygons)
        self.assertTrue(config.detect_lines)
        self.assertTrue(config.detect_ellipses)
        
        # Display options
        self.assertTrue(config.show_contours)
        self.assertTrue(config.show_centers)
        self.assertTrue(config.show_labels)
        self.assertTrue(config.show_bounding_boxes)
        self.assertTrue(config.show_statistics)
        
    def test_calibration_config_custom_values(self):
        """Test CalibrationConfig with custom values"""
        config = CalibrationConfig(
            min_area=200,
            max_area=10000,
            detect_circles=False,
            show_labels=False,
            exposure=-3
        )
        
        self.assertEqual(config.min_area, 200)
        self.assertEqual(config.max_area, 10000)
        self.assertFalse(config.detect_circles)
        self.assertFalse(config.show_labels)
        self.assertEqual(config.exposure, -3)
        
    def test_calibration_config_as_dict(self):
        """Test converting CalibrationConfig to dict"""
        config = CalibrationConfig(
            min_area=500,
            detect_triangles=False
        )
        
        config_dict = asdict(config)
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['min_area'], 500)
        self.assertFalse(config_dict['detect_triangles'])


class TestCreateTestPattern(unittest.TestCase):
    """Test the create_test_pattern function"""
    
    def test_create_test_pattern_default(self):
        """Test create_test_pattern with default parameters"""
        pattern = create_test_pattern()
        
        self.assertEqual(pattern.shape, (600, 800, 3))
        self.assertEqual(pattern.dtype, np.uint8)
        
        # Should have white background
        self.assertTrue(np.all(pattern[0, 0] == 255))
        
    def test_create_test_pattern_custom_size(self):
        """Test create_test_pattern with custom size"""
        pattern = create_test_pattern(width=1000, height=500)
        
        self.assertEqual(pattern.shape, (500, 1000, 3))
        
    def test_create_test_pattern_has_shapes(self):
        """Test that pattern contains shapes"""
        pattern = create_test_pattern()
        
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
        # Invert because shapes are black on white
        gray = 255 - gray
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Should have multiple shapes
        self.assertGreater(len(contours), 5)
        
    def test_create_test_pattern_shape_variety(self):
        """Test that pattern has variety of shapes"""
        pattern = create_test_pattern()
        
        # Convert and find contours
        gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for different vertex counts (indicating different shapes)
        vertex_counts = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            vertex_counts.append(len(approx))
            
        # Should have variety (circles will have many vertices)
        unique_counts = set(vertex_counts)
        self.assertGreater(len(unique_counts), 3)


class TestInteractiveCalibrator(unittest.TestCase):
    """Test the InteractiveCalibrator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('realtime_calibration_tool.GeometryDetectionSystem'):
            with patch('tkinter.Tk'):
                self.calibrator = InteractiveCalibrator()
                
    def test_interactive_calibrator_initialization(self):
        """Test InteractiveCalibrator initialization"""
        self.assertIsInstance(self.calibrator.config, CalibrationConfig)
        self.assertIsNotNone(self.calibrator.system)
        self.assertIsNotNone(self.calibrator.root)
        self.assertFalse(self.calibrator.running)
        self.assertFalse(self.calibrator.paused)
        self.assertIsNone(self.calibrator.current_frame)
        self.assertEqual(len(self.calibrator.detection_times), 0)
        
    @patch('tkinter.Frame')
    @patch('tkinter.Label')
    @patch('tkinter.Button')
    def test_create_gui(self, mock_button, mock_label, mock_frame):
        """Test create_gui method"""
        with patch.object(self.calibrator, '_create_detection_controls'):
            with patch.object(self.calibrator, '_create_camera_controls'):
                with patch.object(self.calibrator, '_create_filter_controls'):
                    with patch.object(self.calibrator, '_create_display_controls'):
                        self.calibrator.create_gui()
                        
        # Should create main window elements
        self.assertGreater(mock_label.call_count, 0)
        self.assertGreater(mock_button.call_count, 0)
        
    def test_create_slider(self):
        """Test _create_slider helper method"""
        parent = Mock()
        
        with patch('tkinter.Label') as mock_label:
            with patch('tkinter.Scale') as mock_scale:
                slider = self.calibrator._create_slider(
                    parent, "Test Slider", 0, 100, 50, self.calibrator.update_detection_params
                )
                
        # Should create label and scale
        mock_label.assert_called_once()
        mock_scale.assert_called_once()
        
        # Check scale configuration
        scale_kwargs = mock_scale.call_args[1]
        self.assertEqual(scale_kwargs['from_'], 0)
        self.assertEqual(scale_kwargs['to'], 100)
        self.assertEqual(scale_kwargs['orient'], tk.HORIZONTAL)
        
    def test_update_detection_params(self):
        """Test update_detection_params method"""
        # Set up mock sliders
        self.calibrator.min_area_slider = Mock(get=Mock(return_value=200))
        self.calibrator.max_area_slider = Mock(get=Mock(return_value=10000))
        self.calibrator.min_perimeter_slider = Mock(get=Mock(return_value=60))
        self.calibrator.max_perimeter_slider = Mock(get=Mock(return_value=1500))
        self.calibrator.epsilon_slider = Mock(get=Mock(return_value=0.03))
        self.calibrator.min_circularity_slider = Mock(get=Mock(return_value=0.8))
        self.calibrator.min_convexity_slider = Mock(get=Mock(return_value=0.75))
        self.calibrator.min_inertia_slider = Mock(get=Mock(return_value=0.6))
        
        self.calibrator.update_detection_params()
        
        # Check config was updated
        self.assertEqual(self.calibrator.config.min_area, 200)
        self.assertEqual(self.calibrator.config.max_area, 10000)
        self.assertEqual(self.calibrator.config.min_perimeter, 60)
        self.assertEqual(self.calibrator.config.max_perimeter, 1500)
        self.assertAlmostEqual(self.calibrator.config.epsilon_factor, 0.03)
        self.assertAlmostEqual(self.calibrator.config.min_circularity, 0.8)
        self.assertAlmostEqual(self.calibrator.config.min_convexity, 0.75)
        self.assertAlmostEqual(self.calibrator.config.min_inertia, 0.6)
        
        # Check detector config was updated
        self.assertEqual(self.calibrator.system.detector.config.min_area, 200)
        
    def test_update_camera_params(self):
        """Test update_camera_params method"""
        # Set up mocks
        self.calibrator.exposure_slider = Mock(get=Mock(return_value=-3))
        self.calibrator.gain_slider = Mock(get=Mock(return_value=15))
        self.calibrator.brightness_slider = Mock(get=Mock(return_value=60))
        self.calibrator.contrast_slider = Mock(get=Mock(return_value=70))
        
        self.calibrator.update_camera_params()
        
        # Check config updated
        self.assertEqual(self.calibrator.config.exposure, -3)
        self.assertEqual(self.calibrator.config.gain, 15)
        self.assertEqual(self.calibrator.config.brightness, 60)
        self.assertEqual(self.calibrator.config.contrast, 70)
        
        # Check camera properties were set
        expected_calls = [
            call('exposure', -3),
            call('gain', 15),
            call('brightness', 60),
            call('contrast', 70)
        ]
        self.calibrator.system.camera.set_property.assert_has_calls(expected_calls)
        
    def test_update_shape_filter(self):
        """Test update_shape_filter method"""
        # Set up shape filter variables
        self.calibrator.detect_circles_var = Mock(get=Mock(return_value=True))
        self.calibrator.detect_rectangles_var = Mock(get=Mock(return_value=False))
        self.calibrator.detect_triangles_var = Mock(get=Mock(return_value=True))
        self.calibrator.detect_polygons_var = Mock(get=Mock(return_value=True))
        self.calibrator.detect_lines_var = Mock(get=Mock(return_value=False))
        self.calibrator.detect_ellipses_var = Mock(get=Mock(return_value=True))
        
        self.calibrator.update_shape_filter()
        
        # Check config updated
        self.assertTrue(self.calibrator.config.detect_circles)
        self.assertFalse(self.calibrator.config.detect_rectangles)
        self.assertTrue(self.calibrator.config.detect_triangles)
        self.assertTrue(self.calibrator.config.detect_polygons)
        self.assertFalse(self.calibrator.config.detect_lines)
        self.assertTrue(self.calibrator.config.detect_ellipses)
        
    def test_update_display_options(self):
        """Test update_display_options method"""
        # Set up display option variables
        self.calibrator.show_contours_var = Mock(get=Mock(return_value=True))
        self.calibrator.show_centers_var = Mock(get=Mock(return_value=False))
        self.calibrator.show_labels_var = Mock(get=Mock(return_value=True))
        self.calibrator.show_bbox_var = Mock(get=Mock(return_value=False))
        self.calibrator.show_stats_var = Mock(get=Mock(return_value=True))
        
        self.calibrator.update_display_options()
        
        # Check config updated
        self.assertTrue(self.calibrator.config.show_contours)
        self.assertFalse(self.calibrator.config.show_centers)
        self.assertTrue(self.calibrator.config.show_labels)
        self.assertFalse(self.calibrator.config.show_bounding_boxes)
        self.assertTrue(self.calibrator.config.show_statistics)
        
    def test_toggle_detection(self):
        """Test toggle_detection method"""
        # Mock button
        self.calibrator.toggle_button = Mock()
        
        # Start detection
        with patch.object(self.calibrator, 'start_detection') as mock_start:
            self.calibrator.toggle_detection()
            mock_start.assert_called_once()
            
        self.calibrator.running = True
        
        # Stop detection
        with patch.object(self.calibrator, 'stop_detection') as mock_stop:
            self.calibrator.toggle_detection()
            mock_stop.assert_called_once()
            
    def test_start_detection(self):
        """Test start_detection method"""
        self.calibrator.toggle_button = Mock()
        
        with patch('threading.Thread') as mock_thread:
            self.calibrator.start_detection()
            
        self.assertTrue(self.calibrator.running)
        self.calibrator.toggle_button.config.assert_called_with(text="Stop Detection")
        mock_thread.assert_called_once()
        
    def test_stop_detection(self):
        """Test stop_detection method"""
        self.calibrator.running = True
        self.calibrator.toggle_button = Mock()
        
        self.calibrator.stop_detection()
        
        self.assertFalse(self.calibrator.running)
        self.calibrator.toggle_button.config.assert_called_with(text="Start Detection")
        
    def test_toggle_pause(self):
        """Test toggle_pause method"""
        self.calibrator.pause_button = Mock()
        
        # Initial state - not paused
        self.assertFalse(self.calibrator.paused)
        
        # Pause
        self.calibrator.toggle_pause()
        self.assertTrue(self.calibrator.paused)
        self.calibrator.pause_button.config.assert_called_with(text="Resume")
        
        # Resume
        self.calibrator.toggle_pause()
        self.assertFalse(self.calibrator.paused)
        self.calibrator.pause_button.config.assert_called_with(text="Pause")
        
    def test_filter_shapes(self):
        """Test filter_shapes method"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        # Create test shapes
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=1000,
                perimeter=100,
                bbox=(0, 0, 50, 50),
                confidence=0.9
            ),
            GeometricShape(
                type=ShapeType.RECTANGLE,
                contour=np.array([]),
                center=(200, 200),
                area=2000,
                perimeter=200,
                bbox=(0, 0, 100, 100),
                confidence=0.8
            ),
            GeometricShape(
                type=ShapeType.TRIANGLE,
                contour=np.array([]),
                center=(300, 300),
                area=500,
                perimeter=80,
                bbox=(0, 0, 30, 30),
                confidence=0.7
            )
        ]
        
        # Test with all shapes enabled
        self.calibrator.config.detect_circles = True
        self.calibrator.config.detect_rectangles = True
        self.calibrator.config.detect_triangles = True
        
        filtered = self.calibrator.filter_shapes(shapes)
        self.assertEqual(len(filtered), 3)
        
        # Test with only circles
        self.calibrator.config.detect_circles = True
        self.calibrator.config.detect_rectangles = False
        self.calibrator.config.detect_triangles = False
        
        filtered = self.calibrator.filter_shapes(shapes)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].type, ShapeType.CIRCLE)
        
    def test_draw_results(self):
        """Test draw_results method"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        shapes = []
        detection_time = 0.025
        
        # Mock visualizer
        self.calibrator.system.visualizer.draw_shapes = Mock(return_value=frame)
        
        # Test with show_statistics enabled
        self.calibrator.config.show_statistics = True
        
        with patch('cv2.putText') as mock_puttext:
            result = self.calibrator.draw_results(frame, shapes, detection_time)
            
            # Should draw statistics
            mock_puttext.assert_called()
            
    def test_mouse_callback(self):
        """Test mouse_callback method"""
        # Test left click - start ROI selection
        with patch.object(self.calibrator, 'start_roi_selection') as mock_roi:
            self.calibrator.mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 200, 0, None)
            mock_roi.assert_called_once_with(100, 200)
            
        # Test mouse move with selection
        self.calibrator.roi_selecting = True
        self.calibrator.roi_start = (50, 50)
        
        self.calibrator.mouse_callback(cv2.EVENT_MOUSEMOVE, 150, 150, 0, None)
        self.assertEqual(self.calibrator.roi_end, (150, 150))
        
        # Test left button up - end selection
        self.calibrator.mouse_callback(cv2.EVENT_LBUTTONUP, 200, 200, 0, None)
        self.assertFalse(self.calibrator.roi_selecting)
        self.assertIsNotNone(self.calibrator.roi)
        
        # Test right click - clear ROI
        self.calibrator.mouse_callback(cv2.EVENT_RBUTTONDOWN, 0, 0, 0, None)
        self.assertIsNone(self.calibrator.roi)
        
    def test_start_roi_selection(self):
        """Test start_roi_selection method"""
        self.calibrator.start_roi_selection(100, 200)
        
        self.assertTrue(self.calibrator.roi_selecting)
        self.assertEqual(self.calibrator.roi_start, (100, 200))
        self.assertIsNone(self.calibrator.roi_end)
        
    def test_reset_parameters(self):
        """Test reset_parameters method"""
        # Modify config
        self.calibrator.config.min_area = 500
        self.calibrator.config.detect_circles = False
        
        # Create mock sliders
        self.calibrator.min_area_slider = Mock()
        self.calibrator.max_area_slider = Mock()
        self.calibrator.detect_circles_var = Mock()
        
        # Add all required slider mocks
        slider_names = ['min_perimeter_slider', 'max_perimeter_slider', 
                       'epsilon_slider', 'min_circularity_slider',
                       'min_convexity_slider', 'min_inertia_slider',
                       'exposure_slider', 'gain_slider', 
                       'brightness_slider', 'contrast_slider']
        
        for name in slider_names:
            setattr(self.calibrator, name, Mock())
            
        # Add all checkbox mocks
        checkbox_names = ['detect_rectangles_var', 'detect_triangles_var',
                         'detect_polygons_var', 'detect_lines_var', 
                         'detect_ellipses_var', 'show_contours_var',
                         'show_centers_var', 'show_labels_var',
                         'show_bbox_var', 'show_stats_var']
        
        for name in checkbox_names:
            setattr(self.calibrator, name, Mock())
            
        self.calibrator.reset_parameters()
        
        # Check config reset
        self.assertEqual(self.calibrator.config.min_area, 100)  # Default
        self.assertTrue(self.calibrator.config.detect_circles)  # Default
        
        # Check sliders were reset
        self.calibrator.min_area_slider.set.assert_called_with(100)
        self.calibrator.detect_circles_var.set.assert_called_with(True)
        
    def test_save_config(self):
        """Test save_config method"""
        with patch('tkinter.filedialog.asksaveasfilename') as mock_dialog:
            mock_dialog.return_value = "/tmp/test_config.json"
            
            with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
                with patch('json.dump') as mock_json:
                    self.calibrator.save_config()
                    
                    # Check file was opened and data saved
                    mock_file.assert_called_once_with("/tmp/test_config.json", 'w')
                    mock_json.assert_called_once()
                    
                    # Check saved data structure
                    saved_data = mock_json.call_args[0][0]
                    self.assertIsInstance(saved_data, dict)
                    self.assertIn('min_area', saved_data)
                    
    def test_load_config(self):
        """Test load_config method"""
        test_config = {
            "min_area": 300,
            "max_area": 20000,
            "detect_circles": False,
            "show_labels": True
        }
        
        with patch('tkinter.filedialog.askopenfilename') as mock_dialog:
            mock_dialog.return_value = "/tmp/test_config.json"
            
            with patch('builtins.open', unittest.mock.mock_open(
                read_data=json.dumps(test_config))):
                
                # Add required UI element mocks
                self.calibrator.min_area_slider = Mock()
                self.calibrator.max_area_slider = Mock()
                self.calibrator.detect_circles_var = Mock()
                self.calibrator.show_labels_var = Mock()
                
                self.calibrator.load_config()
                
                # Check config was updated
                self.assertEqual(self.calibrator.config.min_area, 300)
                self.assertEqual(self.calibrator.config.max_area, 20000)
                self.assertFalse(self.calibrator.config.detect_circles)
                
                # Check UI was updated
                self.calibrator.min_area_slider.set.assert_called_with(300)
                self.calibrator.detect_circles_var.set.assert_called_with(False)
                
    @patch('cv2.imwrite')
    def test_save_screenshot(self, mock_imwrite):
        """Test save_screenshot method"""
        mock_imwrite.return_value = True
        
        # Set current frame
        self.calibrator.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        self.calibrator.save_screenshot()
        
        # Check imwrite was called
        mock_imwrite.assert_called_once()
        
        # Check filename format
        filename = mock_imwrite.call_args[0][0]
        self.assertTrue(filename.startswith('calibration_'))
        self.assertTrue(filename.endswith('.png'))
        
    def test_update_statistics(self):
        """Test update_statistics method"""
        # Add some detection times
        self.calibrator.detection_times = [0.02, 0.025, 0.022, 0.024, 0.023]
        self.calibrator.detected_shapes_count = [5, 7, 6, 8, 6]
        self.calibrator.frame_count = 100
        
        # Create mock labels
        self.calibrator.fps_label = Mock()
        self.calibrator.avg_time_label = Mock()
        self.calibrator.shape_count_label = Mock()
        self.calibrator.frame_count_label = Mock()
        
        self.calibrator.update_statistics()
        
        # Check labels were updated
        self.calibrator.fps_label.config.assert_called()
        self.calibrator.avg_time_label.config.assert_called()
        self.calibrator.shape_count_label.config.assert_called()
        self.calibrator.frame_count_label.config.assert_called()
        
        # Verify calculations
        fps_text = self.calibrator.fps_label.config.call_args[1]['text']
        self.assertIn("FPS:", fps_text)
        
    def test_on_closing(self):
        """Test on_closing method"""
        self.calibrator.running = True
        self.calibrator.root = Mock()
        
        self.calibrator.on_closing()
        
        self.assertFalse(self.calibrator.running)
        self.calibrator.system.cleanup.assert_called_once()
        self.calibrator.root.destroy.assert_called_once()
        
    def test_detection_loop(self):
        """Test detection_loop method"""
        # Set up mocks
        self.calibrator.system.camera.read = Mock(side_effect=[
            np.zeros((480, 640, 3), dtype=np.uint8),  # Frame 1
            np.zeros((480, 640, 3), dtype=np.uint8),  # Frame 2
            None  # End
        ])
        
        self.calibrator.system.detector.detect_shapes = Mock(return_value=[])
        self.calibrator.result_queue = queue.Queue()
        
        # Run detection loop
        self.calibrator.running = True
        self.calibrator.detection_loop()
        
        # Check frames were processed
        self.assertEqual(self.calibrator.system.camera.read.call_count, 3)
        self.assertEqual(self.calibrator.system.detector.detect_shapes.call_count, 2)
        
        # Check results queued
        self.assertFalse(self.calibrator.result_queue.empty())


class TestBatchCalibrator(unittest.TestCase):
    """Test the BatchCalibrator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('realtime_calibration_tool.GeometryDetectionSystem'):
            self.calibrator = BatchCalibrator()
            
    def test_batch_calibrator_initialization(self):
        """Test BatchCalibrator initialization"""
        self.assertIsNotNone(self.calibrator.system)
        
    def test_find_optimal_parameters_single_frame(self):
        """Test find_optimal_parameters with single test frame"""
        # Create test frame
        test_frame = create_test_pattern()
        
        # Mock detector
        self.calibrator.system.detector.detect_shapes = Mock(
            return_value=[Mock() for _ in range(5)]
        )
        
        # Define simple parameter ranges
        param_ranges = {
            'min_area': [50, 100, 150],
            'epsilon_factor': [0.01, 0.02, 0.03]
        }
        
        best_params, best_score = self.calibrator.find_optimal_parameters(
            test_frames=[test_frame],
            param_ranges=param_ranges
        )
        
        # Check results
        self.assertIsInstance(best_params, dict)
        self.assertIn('min_area', best_params)
        self.assertIn('epsilon_factor', best_params)
        self.assertIsInstance(best_score, float)
        self.assertGreater(best_score, 0)
        
    def test_find_optimal_parameters_multiple_frames(self):
        """Test find_optimal_parameters with multiple frames"""
        # Create test frames
        test_frames = [create_test_pattern() for _ in range(3)]
        
        # Mock detector with variable results
        detection_results = [
            [Mock() for _ in range(5)],
            [Mock() for _ in range(7)],
            [Mock() for _ in range(6)]
        ]
        self.calibrator.system.detector.detect_shapes = Mock(
            side_effect=detection_results * 10  # Repeat for all parameter combos
        )
        
        param_ranges = {
            'min_area': [100, 200],
            'max_area': [5000, 10000]
        }
        
        best_params, best_score = self.calibrator.find_optimal_parameters(
            test_frames=test_frames,
            param_ranges=param_ranges
        )
        
        # Verify optimization ran
        self.assertIsInstance(best_params, dict)
        self.assertGreater(best_score, 0)
        
        # Check all combinations were tested (2 * 2 = 4 combinations)
        expected_calls = len(test_frames) * 4
        self.assertEqual(
            self.calibrator.system.detector.detect_shapes.call_count, 
            expected_calls
        )
        
    def test_find_optimal_parameters_custom_weight(self):
        """Test find_optimal_parameters with custom weight function"""
        test_frame = create_test_pattern()
        
        # Custom weight function that prioritizes detection count
        def custom_weight(shapes):
            return len(shapes) * 2.0
            
        self.calibrator.system.detector.detect_shapes = Mock(
            return_value=[Mock() for _ in range(3)]
        )
        
        param_ranges = {'min_area': [50, 100]}
        
        best_params, best_score = self.calibrator.find_optimal_parameters(
            test_frames=[test_frame],
            param_ranges=param_ranges,
            weight_func=custom_weight
        )
        
        # Score should be based on custom weight
        self.assertEqual(best_score, 6.0)  # 3 shapes * 2.0
        
    def test_find_optimal_parameters_empty_results(self):
        """Test find_optimal_parameters when no shapes detected"""
        test_frame = create_test_pattern()
        
        # Mock detector returning no shapes
        self.calibrator.system.detector.detect_shapes = Mock(return_value=[])
        
        param_ranges = {'min_area': [1000, 2000]}  # Too high
        
        best_params, best_score = self.calibrator.find_optimal_parameters(
            test_frames=[test_frame],
            param_ranges=param_ranges
        )
        
        # Should still return valid results
        self.assertIsInstance(best_params, dict)
        self.assertEqual(best_score, 0.0)
        
    def test_find_optimal_parameters_progress_callback(self):
        """Test find_optimal_parameters with progress callback"""
        test_frame = create_test_pattern()
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
            
        self.calibrator.system.detector.detect_shapes = Mock(return_value=[])
        
        param_ranges = {
            'min_area': [50, 100],
            'epsilon_factor': [0.01, 0.02]
        }
        
        self.calibrator.find_optimal_parameters(
            test_frames=[test_frame],
            param_ranges=param_ranges,
            progress_callback=progress_callback
        )
        
        # Check progress was reported
        self.assertGreater(len(progress_calls), 0)
        
        # Check final progress
        final_current, final_total = progress_calls[-1]
        self.assertEqual(final_current, final_total)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for calibration tool scenarios"""
    
    def test_test_pattern_detection(self):
        """Test that test pattern produces detectable shapes"""
        pattern = create_test_pattern()
        
        # Basic shape detection
        gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray  # Invert
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        min_area = 100
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Should detect multiple shapes
        self.assertGreater(len(valid_contours), 5)
        
        # Check shape properties
        for contour in valid_contours[:5]:  # Check first 5
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            self.assertGreater(area, min_area)
            self.assertGreater(perimeter, 0)
            
    def test_calibration_config_completeness(self):
        """Test CalibrationConfig covers all necessary parameters"""
        config = CalibrationConfig()
        config_dict = asdict(config)
        
        # Check all parameter categories exist
        param_categories = {
            'detection': ['min_area', 'max_area', 'min_perimeter', 'max_perimeter'],
            'camera': ['exposure', 'gain', 'brightness', 'contrast'],
            'filters': ['detect_circles', 'detect_rectangles', 'detect_triangles'],
            'display': ['show_contours', 'show_centers', 'show_labels']
        }
        
        for category, params in param_categories.items():
            for param in params:
                self.assertIn(param, config_dict, 
                             f"Missing {param} in {category} parameters")
                             
    @patch('realtime_calibration_tool.GeometryDetectionSystem')
    def test_parameter_update_flow(self, mock_system_class):
        """Test complete parameter update flow"""
        mock_system = Mock()
        mock_system_class.return_value = mock_system
        
        with patch('tkinter.Tk'):
            calibrator = InteractiveCalibrator()
            
        # Set up mock UI elements
        calibrator.min_area_slider = Mock(get=Mock(return_value=300))
        calibrator.detect_circles_var = Mock(get=Mock(return_value=False))
        
        # Update parameters
        calibrator.update_detection_params()
        calibrator.update_shape_filter()
        
        # Verify system configuration updated
        self.assertEqual(calibrator.system.detector.config.min_area, 300)
        self.assertEqual(calibrator.config.detect_circles, False)


if __name__ == '__main__':
    unittest.main(verbosity=2)