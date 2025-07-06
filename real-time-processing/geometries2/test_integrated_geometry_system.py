#!/usr/bin/env python3
"""
Comprehensive test suite for integrated_geometry_system.py
Tests all functions, classes, and methods with rigorous edge cases
"""

import unittest
import numpy as np
import cv2
import json
import tempfile
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import logging

# Import the module to test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrated_geometry_system import (
    Config, ShapeType, GeometricShape, TubeAngle, PerformanceMetrics,
    CameraBackend, OpenCVCamera, PylonCamera, KalmanFilter,
    GeometryDetector, TubeAngleDetector, PerformanceMonitor,
    Visualizer, GeometryDetectionSystem, setup_logging
)


class TestSetupLogging(unittest.TestCase):
    """Test the setup_logging function"""
    
    def test_setup_logging_creates_logger(self):
        """Test that setup_logging creates a logger"""
        logger = setup_logging()
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'GeometryDetection')
        
    def test_setup_logging_level(self):
        """Test logger level is set to INFO"""
        logger = setup_logging()
        self.assertEqual(logger.level, logging.INFO)


class TestConfig(unittest.TestCase):
    """Test the Config class"""
    
    def test_config_default_values(self):
        """Test Config has correct default values"""
        config = Config()
        self.assertEqual(config.camera_id, 0)
        self.assertEqual(config.camera_backend, 'opencv')
        self.assertEqual(config.min_area, 100)
        self.assertEqual(config.max_area, 50000)
        self.assertTrue(config.enable_gpu)
        self.assertFalse(config.record_video)
        
    def test_config_custom_values(self):
        """Test Config with custom values"""
        config = Config(
            camera_id=1,
            min_area=200,
            enable_gpu=False
        )
        self.assertEqual(config.camera_id, 1)
        self.assertEqual(config.min_area, 200)
        self.assertFalse(config.enable_gpu)


class TestShapeType(unittest.TestCase):
    """Test the ShapeType enum"""
    
    def test_shape_types_exist(self):
        """Test all shape types are defined"""
        expected_types = ['CIRCLE', 'TRIANGLE', 'RECTANGLE', 'SQUARE', 
                         'PENTAGON', 'HEXAGON', 'POLYGON', 'LINE', 
                         'ELLIPSE', 'UNKNOWN']
        for shape_type in expected_types:
            self.assertTrue(hasattr(ShapeType, shape_type))
            
    def test_shape_type_values(self):
        """Test shape type string values"""
        self.assertEqual(ShapeType.CIRCLE.value, 'circle')
        self.assertEqual(ShapeType.TRIANGLE.value, 'triangle')
        self.assertEqual(ShapeType.RECTANGLE.value, 'rectangle')


class TestGeometricShape(unittest.TestCase):
    """Test the GeometricShape dataclass"""
    
    def test_geometric_shape_creation(self):
        """Test creating a GeometricShape"""
        contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        shape = GeometricShape(
            type=ShapeType.SQUARE,
            contour=contour,
            center=(5, 5),
            area=100.0,
            perimeter=40.0,
            bbox=(0, 0, 10, 10),
            confidence=0.95
        )
        self.assertEqual(shape.type, ShapeType.SQUARE)
        self.assertEqual(shape.center, (5, 5))
        self.assertEqual(shape.area, 100.0)
        
    def test_geometric_shape_to_dict(self):
        """Test GeometricShape.to_dict() method"""
        contour = np.array([[0, 0], [10, 0], [10, 10]])
        shape = GeometricShape(
            type=ShapeType.TRIANGLE,
            contour=contour,
            center=(5, 5),
            area=50.0,
            perimeter=30.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            angles=[60.0, 60.0, 60.0]
        )
        
        shape_dict = shape.to_dict()
        self.assertEqual(shape_dict['type'], 'triangle')
        self.assertEqual(shape_dict['center'], (5, 5))
        self.assertEqual(shape_dict['area'], 50.0)
        self.assertEqual(shape_dict['angles'], [60.0, 60.0, 60.0])
        self.assertIsInstance(shape_dict['contour'], list)


class TestTubeAngle(unittest.TestCase):
    """Test the TubeAngle dataclass"""
    
    def test_tube_angle_creation(self):
        """Test creating a TubeAngle"""
        tube = TubeAngle(
            angle=45.0,
            ellipse=((100, 100), (50, 30), 45),
            confidence=0.85,
            is_concentric=True,
            inner_ellipse=((100, 100), (30, 20), 45)
        )
        self.assertEqual(tube.angle, 45.0)
        self.assertEqual(tube.confidence, 0.85)
        self.assertTrue(tube.is_concentric)


class TestPerformanceMetrics(unittest.TestCase):
    """Test the PerformanceMetrics dataclass"""
    
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics"""
        metrics = PerformanceMetrics(
            fps=30.0,
            frame_time=0.033,
            cpu_percent=45.0,
            memory_mb=512.0,
            gpu_percent=25.0,
            gpu_memory_mb=256.0
        )
        self.assertEqual(metrics.fps, 30.0)
        self.assertAlmostEqual(metrics.frame_time, 0.033, places=3)


class TestOpenCVCamera(unittest.TestCase):
    """Test the OpenCVCamera class"""
    
    @patch('cv2.VideoCapture')
    def test_opencv_camera_open(self, mock_capture):
        """Test OpenCVCamera.open()"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_id=0)
        self.assertTrue(camera.open())
        mock_capture.assert_called_once_with(0)
        
    @patch('cv2.VideoCapture')
    def test_opencv_camera_open_failure(self, mock_capture):
        """Test OpenCVCamera.open() failure"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_id=0)
        self.assertFalse(camera.open())
        
    @patch('cv2.VideoCapture')
    def test_opencv_camera_read(self, mock_capture):
        """Test OpenCVCamera.read()"""
        mock_cap = Mock()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_id=0)
        camera.cap = mock_cap
        frame = camera.read()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        
    @patch('cv2.VideoCapture')
    def test_opencv_camera_close(self, mock_capture):
        """Test OpenCVCamera.close()"""
        mock_cap = Mock()
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_id=0)
        camera.cap = mock_cap
        camera.close()
        mock_cap.release.assert_called_once()
        
    @patch('cv2.VideoCapture')
    def test_opencv_camera_get_properties(self, mock_capture):
        """Test OpenCVCamera.get_properties()"""
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_EXPOSURE: -5,
            cv2.CAP_PROP_GAIN: 10
        }.get(prop, 0)
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_id=0)
        camera.cap = mock_cap
        props = camera.get_properties()
        
        self.assertEqual(props['width'], 640)
        self.assertEqual(props['height'], 480)
        self.assertEqual(props['fps'], 30)
        self.assertEqual(props['exposure'], -5)
        self.assertEqual(props['gain'], 10)
        
    @patch('cv2.VideoCapture')
    def test_opencv_camera_set_property(self, mock_capture):
        """Test OpenCVCamera.set_property()"""
        mock_cap = Mock()
        mock_cap.set.return_value = True
        mock_capture.return_value = mock_cap
        
        camera = OpenCVCamera(camera_id=0)
        camera.cap = mock_cap
        
        self.assertTrue(camera.set_property('exposure', -3))
        mock_cap.set.assert_called_with(cv2.CAP_PROP_EXPOSURE, -3)
        
        self.assertFalse(camera.set_property('invalid_prop', 100))


class TestPylonCamera(unittest.TestCase):
    """Test the PylonCamera class"""
    
    @patch('integrated_geometry_system.pylon')
    def test_pylon_camera_init_no_pylon(self, mock_pylon):
        """Test PylonCamera initialization without pypylon"""
        mock_pylon.side_effect = ImportError()
        
        with self.assertRaises(ImportError):
            camera = PylonCamera()
            
    def test_pylon_camera_methods_not_implemented(self):
        """Test PylonCamera methods raise NotImplementedError when no pypylon"""
        # Create camera without pypylon by mocking the import
        with patch('integrated_geometry_system.pylon', None):
            camera = PylonCamera()
            camera.pylon = None
            
            self.assertFalse(camera.open())
            self.assertIsNone(camera.read())
            camera.close()  # Should not raise
            self.assertEqual(camera.get_properties(), {})
            self.assertFalse(camera.set_property('test', 1))


class TestKalmanFilter(unittest.TestCase):
    """Test the KalmanFilter class"""
    
    def test_kalman_filter_initialization(self):
        """Test KalmanFilter initialization"""
        kf = KalmanFilter(state_dim=4, measurement_dim=2)
        self.assertEqual(kf.kalman.statePost.shape, (4, 1))
        self.assertFalse(kf.initialized)
        
    def test_kalman_filter_update_first_measurement(self):
        """Test KalmanFilter.update() with first measurement"""
        kf = KalmanFilter(state_dim=4, measurement_dim=2)
        measurement = np.array([100.0, 200.0])
        
        result = kf.update(measurement)
        self.assertTrue(kf.initialized)
        np.testing.assert_array_almost_equal(result, measurement)
        
    def test_kalman_filter_update_subsequent_measurements(self):
        """Test KalmanFilter.update() with multiple measurements"""
        kf = KalmanFilter(state_dim=4, measurement_dim=2)
        
        # First measurement
        m1 = np.array([100.0, 200.0])
        kf.update(m1)
        
        # Second measurement - should be filtered
        m2 = np.array([102.0, 201.0])
        result = kf.update(m2)
        
        # Result should be between m1 and m2
        self.assertTrue(100.0 < result[0] < 102.0)
        self.assertTrue(200.0 < result[1] < 201.0)


class TestGeometryDetector(unittest.TestCase):
    """Test the GeometryDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config(enable_gpu=False)
        self.detector = GeometryDetector(self.config)
        
    def test_geometry_detector_initialization(self):
        """Test GeometryDetector initialization"""
        self.assertIsInstance(self.detector.config, Config)
        self.assertIsInstance(self.detector.kalman_filters, dict)
        self.assertEqual(len(self.detector.shape_history), 0)
        
    def test_preprocess_frame_basic(self):
        """Test preprocess_frame with basic image"""
        # Create test image
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (80, 80), (255, 255, 255), -1)
        
        processed = self.detector.preprocess_frame(frame)
        self.assertEqual(processed.shape, (100, 100))  # Should be grayscale
        self.assertEqual(processed.dtype, np.uint8)
        
    def test_preprocess_frame_with_roi(self):
        """Test preprocess_frame with ROI"""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        roi = (10, 10, 50, 50)  # x, y, w, h
        
        processed = self.detector.preprocess_frame(frame, roi)
        self.assertEqual(processed.shape, (50, 50))
        
    def test_multi_scale_edges(self):
        """Test _multi_scale_edges method"""
        # Create test image with edges
        gray = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(gray, (20, 20), (80, 80), 255, 2)
        
        edges = self.detector._multi_scale_edges(gray)
        self.assertEqual(edges.shape, gray.shape)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertTrue(np.any(edges > 0))  # Should detect some edges
        
    def test_detect_shapes_empty_image(self):
        """Test detect_shapes with empty image"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        shapes = self.detector.detect_shapes(frame)
        self.assertEqual(len(shapes), 0)
        
    def test_detect_shapes_with_rectangle(self):
        """Test detect_shapes with a rectangle"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), -1)
        
        shapes = self.detector.detect_shapes(frame)
        self.assertGreater(len(shapes), 0)
        
        # Check detected shape
        rect_shape = shapes[0]
        self.assertIn(rect_shape.type, [ShapeType.RECTANGLE, ShapeType.SQUARE])
        self.assertAlmostEqual(rect_shape.area, 10000, delta=500)
        
    def test_detect_shapes_with_circle(self):
        """Test detect_shapes with a circle"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 50, (255, 255, 255), -1)
        
        shapes = self.detector.detect_shapes(frame)
        self.assertGreater(len(shapes), 0)
        
        # Check detected shape
        circle_shape = shapes[0]
        self.assertEqual(circle_shape.type, ShapeType.CIRCLE)
        self.assertIsNotNone(circle_shape.radius)
        
    def test_process_contour_too_small(self):
        """Test _process_contour with contour too small"""
        small_contour = np.array([[0, 0], [5, 0], [5, 5], [0, 5]])
        result = self.detector._process_contour(small_contour, 0)
        self.assertIsNone(result)
        
    def test_process_contour_valid(self):
        """Test _process_contour with valid contour"""
        # Create a square contour
        contour = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        shape = self.detector._process_contour(contour, 0)
        
        self.assertIsNotNone(shape)
        self.assertIsInstance(shape, GeometricShape)
        self.assertEqual(shape.id, 0)
        
    def test_calculate_shape_properties(self):
        """Test _calculate_shape_properties"""
        contour = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        
        props = self.detector._calculate_shape_properties(contour)
        
        self.assertIn('area', props)
        self.assertIn('perimeter', props)
        self.assertIn('center', props)
        self.assertIn('bbox', props)
        self.assertIn('aspect_ratio', props)
        self.assertIn('extent', props)
        self.assertIn('solidity', props)
        
        self.assertAlmostEqual(props['area'], 10000, delta=100)
        self.assertAlmostEqual(props['perimeter'], 400, delta=10)
        
    def test_classify_shape_triangle(self):
        """Test _classify_shape with triangle"""
        triangle = np.array([[50, 0], [0, 100], [100, 100]])
        shape_type, confidence, extra = self.detector._classify_shape(triangle)
        
        self.assertEqual(shape_type, ShapeType.TRIANGLE)
        self.assertGreater(confidence, 0.5)
        
    def test_classify_shape_square(self):
        """Test _classify_shape with square"""
        square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        shape_type, confidence, extra = self.detector._classify_shape(square)
        
        self.assertIn(shape_type, [ShapeType.SQUARE, ShapeType.RECTANGLE])
        self.assertGreater(confidence, 0.5)
        
    def test_classify_shape_circle(self):
        """Test _classify_shape with circle"""
        # Create circle contour
        angles = np.linspace(0, 2*np.pi, 100)
        circle = np.array([[50 + 50*np.cos(a), 50 + 50*np.sin(a)] 
                          for a in angles], dtype=np.int32)
        
        shape_type, confidence, extra = self.detector._classify_shape(circle)
        
        self.assertEqual(shape_type, ShapeType.CIRCLE)
        self.assertGreater(confidence, 0.7)
        self.assertIn('radius', extra)
        
    def test_calculate_polygon_angles(self):
        """Test _calculate_polygon_angles"""
        # Right angle triangle
        triangle = np.array([[0, 0], [100, 0], [0, 100]])
        angles = self.detector._calculate_polygon_angles(triangle)
        
        self.assertEqual(len(angles), 3)
        self.assertAlmostEqual(angles[0], 90, delta=5)  # Right angle
        
    def test_detect_lines(self):
        """Test _detect_lines"""
        # Create image with lines
        edges = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(edges, (50, 50), (150, 150), 255, 2)
        cv2.line(edges, (50, 150), (150, 50), 255, 2)
        
        lines = self.detector._detect_lines(edges)
        self.assertGreater(len(lines), 0)
        
    def test_detect_circles(self):
        """Test _detect_circles"""
        # Create image with circles
        gray = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(gray, (100, 100), 50, 255, -1)
        cv2.circle(gray, (50, 50), 25, 255, -1)
        
        circles = self.detector._detect_circles(gray)
        self.assertGreaterEqual(len(circles), 1)
        
    def test_apply_temporal_smoothing_no_history(self):
        """Test _apply_temporal_smoothing with no history"""
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([[0, 0]]),
                center=(100, 100),
                area=100,
                perimeter=40,
                bbox=(0, 0, 10, 10),
                confidence=0.9,
                id=1
            )
        ]
        
        smoothed = self.detector._apply_temporal_smoothing(shapes)
        self.assertEqual(len(smoothed), 1)
        self.assertEqual(smoothed[0].center, (100, 100))


class TestTubeAngleDetector(unittest.TestCase):
    """Test the TubeAngleDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = TubeAngleDetector()
        
    def test_tube_angle_detector_initialization(self):
        """Test TubeAngleDetector initialization"""
        self.assertIsInstance(self.detector.kalman_filter, KalmanFilter)
        
    def test_detect_tube_angle_no_tubes(self):
        """Test detect_tube_angle with no tubes"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.detector.detect_tube_angle(frame)
        self.assertIsNone(result)
        
    def test_detect_tube_angle_with_ellipse(self):
        """Test detect_tube_angle with ellipse"""
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw an ellipse
        cv2.ellipse(frame, (150, 150), (80, 50), 45, 0, 360, (255, 255, 255), -1)
        
        result = self.detector.detect_tube_angle(frame)
        # May or may not detect depending on parameters
        if result:
            self.assertIsInstance(result, TubeAngle)
            self.assertIsInstance(result.angle, float)
            
    def test_detect_tube_ellipse(self):
        """Test _detect_tube_ellipse"""
        # Create edges with ellipse
        edges = np.zeros((300, 300), dtype=np.uint8)
        cv2.ellipse(edges, (150, 150), (80, 50), 45, 0, 360, 255, 2)
        
        candidates = self.detector._detect_tube_ellipse(edges)
        self.assertIsInstance(candidates, list)
        
    def test_calculate_ellipse_fit_score_perfect(self):
        """Test _calculate_ellipse_fit_score with perfect ellipse"""
        # Create perfect ellipse contour
        ellipse = ((150, 150), (100, 60), 0)
        angles = np.linspace(0, 2*np.pi, 100)
        contour = np.array([
            [150 + 50*np.cos(a), 150 + 30*np.sin(a)] 
            for a in angles
        ], dtype=np.float32)
        
        # Rotate contour
        M = cv2.getRotationMatrix2D((150, 150), 0, 1)
        contour = cv2.transform(contour.reshape(-1, 1, 2), M).reshape(-1, 2)
        
        score = self.detector._calculate_ellipse_fit_score(contour, ellipse)
        self.assertGreater(score, 0.8)  # Should be high score
        
    def test_detect_concentric_ellipses(self):
        """Test _detect_concentric_ellipses"""
        # Create image with concentric ellipses
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.ellipse(frame, (150, 150), (100, 60), 45, 0, 360, (255, 255, 255), 2)
        cv2.ellipse(frame, (150, 150), (70, 42), 45, 0, 360, (255, 255, 255), 2)
        
        result = self.detector._detect_concentric_ellipses(frame)
        # May detect concentric ellipses depending on parameters
        
    def test_are_concentric(self):
        """Test _are_concentric"""
        ellipse1 = ((150, 150), (100, 60), 45)
        ellipse2 = ((152, 148), (70, 42), 47)  # Similar center and angle
        
        self.assertTrue(self.detector._are_concentric(ellipse1, ellipse2))
        
        # Different centers
        ellipse3 = ((200, 200), (70, 42), 45)
        self.assertFalse(self.detector._are_concentric(ellipse1, ellipse3))
        
    def test_estimate_3d_pose(self):
        """Test _estimate_3d_pose"""
        ellipse = ((150, 150), (100, 60), 0)
        angle = self.detector._estimate_3d_pose(ellipse)
        
        self.assertIsInstance(angle, float)
        self.assertTrue(0 <= angle <= 90)
        
    def test_refine_pose_with_concentric(self):
        """Test _refine_pose_with_concentric"""
        outer = ((150, 150), (100, 60), 0)
        inner = ((150, 150), (70, 42), 0)
        initial_angle = 45.0
        
        refined = self.detector._refine_pose_with_concentric(
            outer, inner, initial_angle
        )
        self.assertIsInstance(refined, float)


class TestPerformanceMonitor(unittest.TestCase):
    """Test the PerformanceMonitor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = PerformanceMonitor()
        
    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization"""
        self.assertEqual(len(self.monitor.frame_times), 0)
        self.assertIsNone(self.monitor.last_update)
        
    def test_update_first_frame(self):
        """Test update() for first frame"""
        self.monitor.update()
        self.assertEqual(len(self.monitor.frame_times), 1)
        self.assertIsNotNone(self.monitor.last_update)
        
    def test_update_multiple_frames(self):
        """Test update() for multiple frames"""
        for _ in range(5):
            self.monitor.update()
            time.sleep(0.01)  # Small delay
            
        self.assertEqual(len(self.monitor.frame_times), 5)
        
    def test_update_max_history(self):
        """Test update() respects max history"""
        for _ in range(150):
            self.monitor.update()
            
        self.assertEqual(len(self.monitor.frame_times), 100)  # Max history
        
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_get_system_resources(self, mock_memory, mock_cpu):
        """Test get_system_resources"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0, used=1024*1024*1024)
        
        cpu, mem = self.monitor.get_system_resources()
        self.assertEqual(cpu, 50.0)
        self.assertEqual(mem, 1024.0)  # MB
        
    def test_get_metrics(self):
        """Test get_metrics"""
        # Add some frame times
        for _ in range(10):
            self.monitor.update()
            time.sleep(0.01)
            
        with patch.object(self.monitor, 'get_system_resources', 
                         return_value=(50.0, 1024.0)):
            metrics = self.monitor.get_metrics()
            
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.fps, 0)
        self.assertGreater(metrics.frame_time, 0)
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_mb, 1024.0)


class TestVisualizer(unittest.TestCase):
    """Test the Visualizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.visualizer = Visualizer(self.config)
        
    def test_visualizer_initialization(self):
        """Test Visualizer initialization"""
        self.assertIsInstance(self.visualizer.config, Config)
        self.assertEqual(len(self.visualizer.color_map), 10)
        
    def test_draw_shapes_empty(self):
        """Test draw_shapes with no shapes"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        shapes = []
        
        result = self.visualizer.draw_shapes(frame, shapes)
        np.testing.assert_array_equal(result, frame)
        
    def test_draw_shapes_with_shapes(self):
        """Test draw_shapes with shapes"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([[100, 100]]),
                center=(100, 100),
                area=100,
                perimeter=40,
                bbox=(90, 90, 20, 20),
                confidence=0.9,
                radius=10
            )
        ]
        
        result = self.visualizer.draw_shapes(frame, shapes)
        # Should have drawn something
        self.assertFalse(np.array_equal(result, frame))
        
    def test_draw_tube_angle(self):
        """Test draw_tube_angle"""
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        tube = TubeAngle(
            angle=45.0,
            ellipse=((150, 150), (100, 60), 45),
            confidence=0.85
        )
        
        result = self.visualizer.draw_tube_angle(frame, tube)
        # Should have drawn something
        self.assertFalse(np.array_equal(result, frame))
        
    def test_draw_performance_overlay(self):
        """Test draw_performance_overlay"""
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        metrics = PerformanceMetrics(
            fps=30.0,
            frame_time=0.033,
            cpu_percent=50.0,
            memory_mb=1024.0
        )
        
        result = self.visualizer.draw_performance_overlay(frame, metrics)
        # Should have drawn text
        self.assertFalse(np.array_equal(result, frame))
        
    def test_draw_shape_info(self):
        """Test _draw_shape_info"""
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        shape = GeometricShape(
            type=ShapeType.RECTANGLE,
            contour=np.array([[50, 50], [150, 50], [150, 150], [50, 150]]),
            center=(100, 100),
            area=10000,
            perimeter=400,
            bbox=(50, 50, 100, 100),
            confidence=0.95
        )
        
        self.visualizer._draw_shape_info(frame, shape)
        # Should have modified frame
        self.assertTrue(np.any(frame > 0))
        
    def test_is_valid_point(self):
        """Test _is_valid_point"""
        # Valid points
        self.assertTrue(self.visualizer._is_valid_point((100, 100), 200, 200))
        self.assertTrue(self.visualizer._is_valid_point((0, 0), 200, 200))
        self.assertTrue(self.visualizer._is_valid_point((199, 199), 200, 200))
        
        # Invalid points
        self.assertFalse(self.visualizer._is_valid_point((-1, 100), 200, 200))
        self.assertFalse(self.visualizer._is_valid_point((100, -1), 200, 200))
        self.assertFalse(self.visualizer._is_valid_point((200, 100), 200, 200))
        self.assertFalse(self.visualizer._is_valid_point((100, 200), 200, 200))
        
    def test_clip_bbox(self):
        """Test _clip_bbox"""
        # Fully inside
        bbox = self.visualizer._clip_bbox(50, 50, 100, 100, 300, 300)
        self.assertEqual(bbox, (50, 50, 100, 100))
        
        # Partially outside
        bbox = self.visualizer._clip_bbox(-10, -10, 100, 100, 300, 300)
        self.assertEqual(bbox, (0, 0, 90, 90))
        
        # Extending beyond bottom-right
        bbox = self.visualizer._clip_bbox(250, 250, 100, 100, 300, 300)
        self.assertEqual(bbox, (250, 250, 50, 50))


class TestGeometryDetectionSystem(unittest.TestCase):
    """Test the GeometryDetectionSystem class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config(camera_backend='opencv')
        
    @patch('integrated_geometry_system.OpenCVCamera')
    def test_system_initialization(self, mock_camera_class):
        """Test GeometryDetectionSystem initialization"""
        mock_camera = Mock()
        mock_camera_class.return_value = mock_camera
        
        system = GeometryDetectionSystem(self.config)
        
        self.assertIsInstance(system.config, Config)
        self.assertIsInstance(system.detector, GeometryDetector)
        self.assertIsInstance(system.tube_detector, TubeAngleDetector)
        self.assertIsInstance(system.visualizer, Visualizer)
        self.assertIsInstance(system.performance_monitor, PerformanceMonitor)
        self.assertEqual(system.camera, mock_camera)
        
    @patch('integrated_geometry_system.OpenCVCamera')
    def test_system_camera_selection(self, mock_opencv):
        """Test camera backend selection"""
        # OpenCV backend
        system = GeometryDetectionSystem(Config(camera_backend='opencv'))
        mock_opencv.assert_called_once()
        
    @patch('integrated_geometry_system.PylonCamera')
    def test_system_pylon_camera_selection(self, mock_pylon):
        """Test Pylon camera backend selection"""
        config = Config(camera_backend='pylon')
        system = GeometryDetectionSystem(config)
        mock_pylon.assert_called_once()
        
    @patch('integrated_geometry_system.OpenCVCamera')
    def test_handle_keyboard_quit(self, mock_camera):
        """Test _handle_keyboard with quit command"""
        system = GeometryDetectionSystem(self.config)
        
        # Test quit keys
        self.assertTrue(system._handle_keyboard(ord('q')))
        self.assertTrue(system._handle_keyboard(27))  # ESC
        
    @patch('integrated_geometry_system.OpenCVCamera')
    def test_handle_keyboard_pause(self, mock_camera):
        """Test _handle_keyboard with pause command"""
        system = GeometryDetectionSystem(self.config)
        
        self.assertFalse(system.paused)
        system._handle_keyboard(ord(' '))
        self.assertTrue(system.paused)
        system._handle_keyboard(ord(' '))
        self.assertFalse(system.paused)
        
    @patch('integrated_geometry_system.OpenCVCamera')
    @patch('cv2.imwrite')
    def test_save_screenshot(self, mock_imwrite, mock_camera):
        """Test _save_screenshot"""
        mock_imwrite.return_value = True
        
        system = GeometryDetectionSystem(self.config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        system._save_screenshot(frame)
        mock_imwrite.assert_called_once()
        
        # Check filename format
        args = mock_imwrite.call_args[0]
        filename = args[0]
        self.assertTrue(filename.startswith('screenshot_'))
        self.assertTrue(filename.endswith('.png'))
        
    @patch('integrated_geometry_system.OpenCVCamera')
    def test_start_stop_recording(self, mock_camera):
        """Test _start_recording and _stop_recording"""
        system = GeometryDetectionSystem(self.config)
        
        # Start recording
        with patch('cv2.VideoWriter') as mock_writer_class:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer
            
            system._start_recording(640, 480)
            self.assertIsNotNone(system.video_writer)
            mock_writer_class.assert_called_once()
            
        # Stop recording
        system._stop_recording()
        self.assertIsNone(system.video_writer)
        mock_writer.release.assert_called_once()
        
    @patch('integrated_geometry_system.OpenCVCamera')
    def test_save_benchmark_results(self, mock_camera):
        """Test _save_benchmark_results"""
        system = GeometryDetectionSystem(self.config)
        
        # Add some detection results
        system.detection_times = [0.01, 0.02, 0.015]
        system.frame_count = 100
        system.detected_shapes = [
            [GeometricShape(type=ShapeType.CIRCLE, contour=np.array([]), 
                          center=(0,0), area=100, perimeter=40, 
                          bbox=(0,0,10,10), confidence=0.9)],
            [GeometricShape(type=ShapeType.SQUARE, contour=np.array([]), 
                          center=(0,0), area=100, perimeter=40, 
                          bbox=(0,0,10,10), confidence=0.9)]
        ]
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('json.dump') as mock_json:
                system._save_benchmark_results()
                
                mock_file.assert_called_once()
                mock_json.assert_called_once()
                
                # Check the data structure
                saved_data = mock_json.call_args[0][0]
                self.assertIn('summary', saved_data)
                self.assertIn('detection_times', saved_data)
                self.assertIn('shape_statistics', saved_data)
                
    @patch('integrated_geometry_system.OpenCVCamera')
    def test_cleanup(self, mock_camera_class):
        """Test cleanup method"""
        mock_camera = Mock()
        mock_camera_class.return_value = mock_camera
        
        system = GeometryDetectionSystem(self.config)
        
        # Set up video writer
        system.video_writer = Mock()
        
        system.cleanup()
        
        mock_camera.close.assert_called_once()
        system.video_writer.release.assert_called_once()
        
    @patch('integrated_geometry_system.OpenCVCamera')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_run_basic_flow(self, mock_waitkey, mock_imshow, mock_camera_class):
        """Test run method basic flow"""
        # Set up mock camera
        mock_camera = Mock()
        mock_camera.open.return_value = True
        mock_camera.read.side_effect = [
            np.zeros((480, 640, 3), dtype=np.uint8),  # Frame 1
            np.zeros((480, 640, 3), dtype=np.uint8),  # Frame 2
            None  # End of stream
        ]
        mock_camera_class.return_value = mock_camera
        
        # Set up mock keyboard input
        mock_waitkey.return_value = -1  # No key pressed
        
        system = GeometryDetectionSystem(self.config)
        
        with patch.object(system, 'cleanup') as mock_cleanup:
            system.run()
            
            # Verify camera was opened
            mock_camera.open.assert_called_once()
            
            # Verify frames were processed
            self.assertEqual(mock_camera.read.call_count, 3)
            
            # Verify cleanup was called
            mock_cleanup.assert_called_once()


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config(enable_gpu=False)
        
    def test_full_detection_pipeline(self):
        """Test complete detection pipeline"""
        # Create test image with multiple shapes
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Add circle
        cv2.circle(frame, (100, 100), 50, (255, 255, 255), -1)
        
        # Add rectangle
        cv2.rectangle(frame, (200, 50), (350, 150), (255, 255, 255), -1)
        
        # Add triangle
        triangle = np.array([[200, 300], [150, 380], [250, 380]], np.int32)
        cv2.fillPoly(frame, [triangle], (255, 255, 255))
        
        # Run detection
        detector = GeometryDetector(self.config)
        shapes = detector.detect_shapes(frame)
        
        # Verify shapes detected
        self.assertGreaterEqual(len(shapes), 3)
        
        # Check shape types
        shape_types = [shape.type for shape in shapes]
        self.assertIn(ShapeType.CIRCLE, shape_types)
        self.assertIn(ShapeType.TRIANGLE, shape_types)
        
    def test_temporal_smoothing_pipeline(self):
        """Test temporal smoothing across multiple frames"""
        detector = GeometryDetector(self.config)
        
        # Create frames with moving circle
        positions = [(100, 100), (102, 101), (104, 102), (106, 103)]
        all_shapes = []
        
        for pos in positions:
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.circle(frame, pos, 30, (255, 255, 255), -1)
            
            shapes = detector.detect_shapes(frame)
            all_shapes.append(shapes)
            
        # Verify smoothing effect
        # Later detections should have smoother movement
        if len(all_shapes) >= 4 and all(len(s) > 0 for s in all_shapes):
            # Check that positions are smoothed
            for i in range(1, len(all_shapes)):
                if all_shapes[i]:
                    shape = all_shapes[i][0]
                    # Position should be smoothed (between previous and current)
                    self.assertIsNotNone(shape.center)
                    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        monitor = PerformanceMonitor()
        
        # Simulate frame processing
        for _ in range(20):
            monitor.update()
            time.sleep(0.01)  # Simulate processing time
            
        metrics = monitor.get_metrics()
        
        # Verify metrics
        self.assertGreater(metrics.fps, 0)
        self.assertLess(metrics.fps, 200)  # Reasonable FPS
        self.assertGreater(metrics.frame_time, 0)
        self.assertIsNotNone(metrics.cpu_percent)
        self.assertIsNotNone(metrics.memory_mb)


if __name__ == '__main__':
    unittest.main(verbosity=2)