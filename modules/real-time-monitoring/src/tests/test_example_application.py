#!/usr/bin/env python3
"""
Comprehensive test suite for example_application.py
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
import threading
import queue
from collections import deque, defaultdict
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call

# Import the module to test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.applications.example_application import ShapeAnalysisDashboard


class TestShapeAnalysisDashboard(unittest.TestCase):
    """Test the ShapeAnalysisDashboard class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Patch the parent class
        with patch('example_application.GeometryDetectionSystem'):
            self.dashboard = ShapeAnalysisDashboard()
            
    def test_dashboard_initialization(self):
        """Test ShapeAnalysisDashboard initialization"""
        # Check custom attributes
        self.assertIsInstance(self.dashboard.shape_history, deque)
        self.assertEqual(self.dashboard.shape_history.maxlen, 1000)
        self.assertIsInstance(self.dashboard.shape_stats, defaultdict)
        self.assertIsInstance(self.dashboard.frame_buffer, deque)
        self.assertEqual(self.dashboard.frame_buffer.maxlen, 30)
        self.assertIsInstance(self.dashboard.alerts, list)
        self.assertIsInstance(self.dashboard.export_queue, queue.Queue)
        self.assertFalse(self.dashboard.recording_stats)
        self.assertIsNone(self.dashboard.stats_file)
        
        # Check filter settings
        self.assertIsNone(self.dashboard.filter_shape_type)
        self.assertIsNone(self.dashboard.filter_min_area)
        self.assertIsNone(self.dashboard.filter_max_area)
        self.assertIsNone(self.dashboard.filter_min_confidence)
        
        # Check alert thresholds
        self.assertEqual(self.dashboard.alert_max_shapes, 50)
        self.assertEqual(self.dashboard.alert_min_fps, 15)
        self.assertEqual(self.dashboard.alert_shape_rate_change, 0.5)
        
    def test_process_frame_basic(self):
        """Test process_frame with basic input"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        # Create test frame and shapes
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=500,
                perimeter=80,
                bbox=(80, 80, 40, 40),
                confidence=0.9
            ),
            GeometricShape(
                type=ShapeType.RECTANGLE,
                contour=np.array([]),
                center=(200, 200),
                area=1000,
                perimeter=120,
                bbox=(150, 150, 100, 100),
                confidence=0.85
            )
        ]
        
        # Mock parent process_frame
        with patch.object(self.dashboard, '_draw_visualization') as mock_draw:
            mock_draw.return_value = frame
            
            result = self.dashboard.process_frame(frame, shapes)
            
        # Verify shape history updated
        self.assertEqual(len(self.dashboard.shape_history), 2)
        
        # Verify statistics updated
        self.assertEqual(self.dashboard.shape_stats[ShapeType.CIRCLE]['count'], 1)
        self.assertEqual(self.dashboard.shape_stats[ShapeType.RECTANGLE]['count'], 1)
        
        # Verify frame buffer updated
        self.assertEqual(len(self.dashboard.frame_buffer), 1)
        
    def test_filter_shapes_no_filters(self):
        """Test _filter_shapes with no filters set"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=500,
                perimeter=80,
                bbox=(0, 0, 50, 50),
                confidence=0.9
            )
        ]
        
        filtered = self.dashboard._filter_shapes(shapes)
        self.assertEqual(len(filtered), 1)
        
    def test_filter_shapes_by_type(self):
        """Test _filter_shapes filtering by shape type"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=500,
                perimeter=80,
                bbox=(0, 0, 50, 50),
                confidence=0.9
            ),
            GeometricShape(
                type=ShapeType.RECTANGLE,
                contour=np.array([]),
                center=(200, 200),
                area=1000,
                perimeter=120,
                bbox=(0, 0, 100, 100),
                confidence=0.85
            )
        ]
        
        # Filter only circles
        self.dashboard.filter_shape_type = ShapeType.CIRCLE
        filtered = self.dashboard._filter_shapes(shapes)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].type, ShapeType.CIRCLE)
        
    def test_filter_shapes_by_area(self):
        """Test _filter_shapes filtering by area"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=200,
                perimeter=50,
                bbox=(0, 0, 20, 20),
                confidence=0.9
            ),
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(200, 200),
                area=800,
                perimeter=100,
                bbox=(0, 0, 40, 40),
                confidence=0.9
            ),
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(300, 300),
                area=1500,
                perimeter=140,
                bbox=(0, 0, 50, 50),
                confidence=0.9
            )
        ]
        
        # Filter by area range
        self.dashboard.filter_min_area = 500
        self.dashboard.filter_max_area = 1200
        
        filtered = self.dashboard._filter_shapes(shapes)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].area, 800)
        
    def test_filter_shapes_by_confidence(self):
        """Test _filter_shapes filtering by confidence"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=500,
                perimeter=80,
                bbox=(0, 0, 50, 50),
                confidence=0.7
            ),
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(200, 200),
                area=500,
                perimeter=80,
                bbox=(0, 0, 50, 50),
                confidence=0.9
            )
        ]
        
        # Filter by minimum confidence
        self.dashboard.filter_min_confidence = 0.8
        
        filtered = self.dashboard._filter_shapes(shapes)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].confidence, 0.9)
        
    def test_filter_shapes_combined_filters(self):
        """Test _filter_shapes with multiple filters"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=300,
                perimeter=60,
                bbox=(0, 0, 30, 30),
                confidence=0.95
            ),
            GeometricShape(
                type=ShapeType.RECTANGLE,
                contour=np.array([]),
                center=(200, 200),
                area=800,
                perimeter=120,
                bbox=(0, 0, 40, 40),
                confidence=0.9
            ),
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(300, 300),
                area=600,
                perimeter=90,
                bbox=(0, 0, 35, 35),
                confidence=0.85
            )
        ]
        
        # Set multiple filters
        self.dashboard.filter_shape_type = ShapeType.CIRCLE
        self.dashboard.filter_min_area = 400
        self.dashboard.filter_min_confidence = 0.8
        
        filtered = self.dashboard._filter_shapes(shapes)
        
        # Only the third circle should pass all filters
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].area, 600)
        
    def test_update_statistics_basic(self):
        """Test _update_statistics with basic shapes"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=500,
                perimeter=80,
                bbox=(0, 0, 50, 50),
                confidence=0.9
            ),
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(200, 200),
                area=700,
                perimeter=95,
                bbox=(0, 0, 55, 55),
                confidence=0.85
            ),
            GeometricShape(
                type=ShapeType.RECTANGLE,
                contour=np.array([]),
                center=(300, 300),
                area=1000,
                perimeter=130,
                bbox=(0, 0, 50, 100),
                confidence=0.95
            )
        ]
        
        self.dashboard._update_statistics(shapes)
        
        # Check circle statistics
        circle_stats = self.dashboard.shape_stats[ShapeType.CIRCLE]
        self.assertEqual(circle_stats['count'], 2)
        self.assertEqual(circle_stats['total_area'], 1200)
        self.assertEqual(circle_stats['areas'], [500, 700])
        self.assertAlmostEqual(circle_stats['avg_area'], 600)
        self.assertAlmostEqual(circle_stats['avg_confidence'], 0.875)
        
        # Check rectangle statistics
        rect_stats = self.dashboard.shape_stats[ShapeType.RECTANGLE]
        self.assertEqual(rect_stats['count'], 1)
        self.assertEqual(rect_stats['total_area'], 1000)
        
    def test_update_statistics_empty_shapes(self):
        """Test _update_statistics with no shapes"""
        self.dashboard._update_statistics([])
        
        # Stats should be empty or have zero counts
        self.assertEqual(len(self.dashboard.shape_stats), 0)
        
    def test_check_alerts_too_many_shapes(self):
        """Test _check_alerts for too many shapes alert"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        # Create many shapes
        shapes = []
        for i in range(60):  # More than alert_max_shapes
            shapes.append(GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(i*10, i*10),
                area=100,
                perimeter=40,
                bbox=(0, 0, 20, 20),
                confidence=0.9
            ))
            
        self.dashboard._check_alerts(shapes, 30.0)
        
        # Should have alert for too many shapes
        self.assertGreater(len(self.dashboard.alerts), 0)
        self.assertIn("Too many shapes", self.dashboard.alerts[-1])
        
    def test_check_alerts_low_fps(self):
        """Test _check_alerts for low FPS alert"""
        self.dashboard._check_alerts([], 10.0)  # FPS below threshold
        
        # Should have low FPS alert
        self.assertGreater(len(self.dashboard.alerts), 0)
        self.assertIn("Low FPS", self.dashboard.alerts[-1])
        
    def test_check_alerts_shape_rate_change(self):
        """Test _check_alerts for shape rate change"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        # Add historical data with consistent shape count
        for _ in range(50):
            shape = GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=100,
                perimeter=40,
                bbox=(0, 0, 20, 20),
                confidence=0.9
            )
            self.dashboard.shape_history.append({
                'timestamp': time.time(),
                'shape': shape
            })
            
        # Now add many more shapes (sudden increase)
        current_shapes = []
        for i in range(30):
            current_shapes.append(GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(i*20, i*20),
                area=100,
                perimeter=40,
                bbox=(0, 0, 20, 20),
                confidence=0.9
            ))
            
        self.dashboard._check_alerts(current_shapes, 30.0)
        
        # Should detect significant rate change
        alerts_text = ' '.join(self.dashboard.alerts)
        self.assertIn("rate change", alerts_text.lower())
        
    def test_check_alerts_max_alerts(self):
        """Test that alerts list doesn't grow unbounded"""
        # Generate many alerts
        for i in range(15):
            self.dashboard._check_alerts([], 10.0)  # Low FPS alert
            
        # Should maintain max of 10 alerts
        self.assertLessEqual(len(self.dashboard.alerts), 10)
        
    def test_draw_visualization(self):
        """Test _draw_visualization method"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=500,
                perimeter=80,
                bbox=(0, 0, 50, 50),
                confidence=0.9
            )
        ]
        
        # Mock parent visualizer
        self.dashboard.visualizer = Mock()
        self.dashboard.visualizer.draw_shapes = Mock(return_value=frame)
        
        with patch.object(self.dashboard, '_draw_stats_panel') as mock_stats:
            with patch.object(self.dashboard, '_draw_alerts') as mock_alerts:
                with patch.object(self.dashboard, '_update_dashboard') as mock_update:
                    result = self.dashboard._draw_visualization(frame, shapes)
                    
        # Verify all components called
        self.dashboard.visualizer.draw_shapes.assert_called_once_with(frame, shapes)
        mock_stats.assert_called_once()
        mock_alerts.assert_called_once()
        mock_update.assert_called_once()
        
    def test_draw_stats_panel(self):
        """Test _draw_stats_panel method"""
        from integrated_geometry_system import ShapeType
        
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Add some statistics
        self.dashboard.shape_stats[ShapeType.CIRCLE] = {
            'count': 5,
            'avg_area': 600,
            'avg_confidence': 0.9
        }
        self.dashboard.shape_stats[ShapeType.RECTANGLE] = {
            'count': 3,
            'avg_area': 1200,
            'avg_confidence': 0.85
        }
        
        with patch('cv2.rectangle') as mock_rect:
            with patch('cv2.putText') as mock_text:
                self.dashboard._draw_stats_panel(frame)
                
        # Should draw panel background
        mock_rect.assert_called()
        
        # Should draw text for statistics
        self.assertGreater(mock_text.call_count, 5)  # Multiple stat lines
        
    def test_draw_alerts(self):
        """Test _draw_alerts method"""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Add test alerts
        self.dashboard.alerts = [
            "Alert: Too many shapes detected",
            "Warning: Low FPS"
        ]
        
        with patch('cv2.putText') as mock_text:
            self.dashboard._draw_alerts(frame)
            
        # Should draw each alert
        self.assertGreaterEqual(mock_text.call_count, 2)
        
    def test_update_dashboard(self):
        """Test _update_dashboard method"""
        # Set up matplotlib figure mock
        self.dashboard.dashboard_fig = Mock()
        self.dashboard.shape_count_ax = Mock()
        self.dashboard.area_dist_ax = Mock()
        self.dashboard.confidence_ax = Mock()
        self.dashboard.timeline_ax = Mock()
        
        # Add some data
        from integrated_geometry_system import ShapeType
        
        self.dashboard.shape_stats[ShapeType.CIRCLE] = {
            'count': 10,
            'areas': [100, 200, 300, 400, 500],
            'confidences': [0.8, 0.85, 0.9, 0.95, 0.92]
        }
        
        # Add shape history
        for i in range(10):
            self.dashboard.shape_history.append({
                'timestamp': time.time() - i,
                'shape': Mock(type=ShapeType.CIRCLE)
            })
            
        with patch('matplotlib.pyplot.draw') as mock_draw:
            with patch('matplotlib.pyplot.pause') as mock_pause:
                self.dashboard._update_dashboard()
                
        # Should clear and update all axes
        self.dashboard.shape_count_ax.clear.assert_called()
        self.dashboard.area_dist_ax.clear.assert_called()
        self.dashboard.confidence_ax.clear.assert_called()
        self.dashboard.timeline_ax.clear.assert_called()
        
        # Should draw and pause
        mock_draw.assert_called()
        mock_pause.assert_called()
        
    def test_export_worker(self):
        """Test _export_worker thread function"""
        # Add test data to export queue
        test_data = {
            'timestamp': '2024-01-01 12:00:00',
            'shapes': [
                {'type': 'circle', 'area': 500},
                {'type': 'rectangle', 'area': 1000}
            ]
        }
        
        self.dashboard.export_queue.put(test_data)
        self.dashboard.export_queue.put(None)  # Sentinel to stop
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            self.dashboard.stats_file = tmp.name
            
        try:
            # Run export worker
            self.dashboard._export_worker()
            
            # Check file contents
            with open(self.dashboard.stats_file, 'r') as f:
                lines = f.readlines()
                
            self.assertEqual(len(lines), 1)
            
            # Parse exported data
            exported = json.loads(lines[0])
            self.assertEqual(exported['timestamp'], test_data['timestamp'])
            self.assertEqual(len(exported['shapes']), 2)
            
        finally:
            if os.path.exists(self.dashboard.stats_file):
                os.unlink(self.dashboard.stats_file)
                
    def test_handle_keyboard_record_stats(self):
        """Test handle_keyboard for recording statistics"""
        # Test start recording (press 'r')
        with patch('threading.Thread') as mock_thread:
            with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
                self.dashboard.handle_keyboard(ord('r'))
                
        self.assertTrue(self.dashboard.recording_stats)
        mock_thread.assert_called_once()
        
        # Test stop recording (press 'r' again)
        self.dashboard.handle_keyboard(ord('r'))
        self.assertFalse(self.dashboard.recording_stats)
        
    def test_handle_keyboard_save_dashboard(self):
        """Test handle_keyboard for saving dashboard"""
        self.dashboard.dashboard_fig = Mock()
        
        with patch('matplotlib.pyplot.savefig') as mock_save:
            self.dashboard.handle_keyboard(ord('d'))
            
        mock_save.assert_called_once()
        filename = mock_save.call_args[0][0]
        self.assertTrue(filename.startswith('dashboard_'))
        self.assertTrue(filename.endswith('.png'))
        
    def test_handle_keyboard_export_current(self):
        """Test handle_keyboard for exporting current statistics"""
        # Add test statistics
        from integrated_geometry_system import ShapeType
        
        self.dashboard.shape_stats[ShapeType.CIRCLE] = {
            'count': 5,
            'avg_area': 600
        }
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('json.dump') as mock_json:
                self.dashboard.handle_keyboard(ord('e'))
                
        # Should save current stats
        mock_file.assert_called()
        mock_json.assert_called()
        
        # Check saved data structure
        saved_data = mock_json.call_args[0][0]
        self.assertIn('timestamp', saved_data)
        self.assertIn('statistics', saved_data)
        
    def test_handle_keyboard_cycle_filters(self):
        """Test handle_keyboard for cycling through shape filters"""
        from integrated_geometry_system import ShapeType
        
        # Initially no filter
        self.assertIsNone(self.dashboard.filter_shape_type)
        
        # Press 'f' to cycle through filters
        self.dashboard.handle_keyboard(ord('f'))
        self.assertEqual(self.dashboard.filter_shape_type, ShapeType.CIRCLE)
        
        self.dashboard.handle_keyboard(ord('f'))
        self.assertEqual(self.dashboard.filter_shape_type, ShapeType.TRIANGLE)
        
        # Continue cycling...
        for _ in range(10):  # Cycle through all types
            self.dashboard.handle_keyboard(ord('f'))
            
        # Should eventually return to None
        self.assertIsNone(self.dashboard.filter_shape_type)
        
    def test_handle_keyboard_parent_keys(self):
        """Test that parent keyboard handling works"""
        # Mock parent handle_keyboard
        parent_handle = Mock(return_value=False)
        
        with patch('example_application.GeometryDetectionSystem.handle_keyboard', parent_handle):
            # Test unhandled key
            result = self.dashboard.handle_keyboard(ord('x'))
            
        parent_handle.assert_called_once_with(ord('x'))
        
    def test_run_method(self):
        """Test run method override"""
        # Mock parent run and matplotlib
        with patch('example_application.GeometryDetectionSystem.run') as mock_parent_run:
            with patch('matplotlib.pyplot.figure') as mock_figure:
                with patch('matplotlib.pyplot.ion') as mock_ion:
                    self.dashboard.run()
                    
        # Should set up matplotlib
        mock_ion.assert_called_once()
        mock_figure.assert_called_once()
        
        # Should call parent run
        mock_parent_run.assert_called_once()
        
    def test_statistics_calculation_accuracy(self):
        """Test accuracy of statistical calculations"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        # Create shapes with known properties
        shapes = []
        areas = [100, 200, 300, 400, 500]
        confidences = [0.8, 0.85, 0.9, 0.95, 1.0]
        
        for area, conf in zip(areas, confidences):
            shapes.append(GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(100, 100),
                area=area,
                perimeter=2 * np.pi * np.sqrt(area/np.pi),
                bbox=(0, 0, 50, 50),
                confidence=conf
            ))
            
        self.dashboard._update_statistics(shapes)
        
        # Check calculations
        stats = self.dashboard.shape_stats[ShapeType.CIRCLE]
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['total_area'], sum(areas))
        self.assertAlmostEqual(stats['avg_area'], np.mean(areas))
        self.assertAlmostEqual(stats['avg_confidence'], np.mean(confidences))
        self.assertAlmostEqual(stats['min_area'], min(areas))
        self.assertAlmostEqual(stats['max_area'], max(areas))


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for example application scenarios"""
    
    def test_complete_analysis_workflow(self):
        """Test complete shape analysis workflow"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        with patch('example_application.GeometryDetectionSystem'):
            dashboard = ShapeAnalysisDashboard()
            
        # Simulate processing multiple frames
        for frame_idx in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Generate varying shapes
            shapes = []
            num_shapes = 5 + (frame_idx % 3)  # Vary shape count
            
            for i in range(num_shapes):
                shape_type = [ShapeType.CIRCLE, ShapeType.RECTANGLE][i % 2]
                shapes.append(GeometricShape(
                    type=shape_type,
                    contour=np.array([]),
                    center=(100 + i*50, 100 + i*50),
                    area=500 + i*100,
                    perimeter=80 + i*10,
                    bbox=(0, 0, 50, 50),
                    confidence=0.8 + (i * 0.02)
                ))
                
            # Process frame
            with patch.object(dashboard, '_draw_visualization', return_value=frame):
                dashboard.process_frame(frame, shapes)
                
        # Verify statistics accumulated
        self.assertGreater(len(dashboard.shape_history), 0)
        self.assertIn(ShapeType.CIRCLE, dashboard.shape_stats)
        self.assertIn(ShapeType.RECTANGLE, dashboard.shape_stats)
        
        # Check frame buffer
        self.assertGreater(len(dashboard.frame_buffer), 0)
        
    def test_alert_generation_scenarios(self):
        """Test various alert generation scenarios"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        with patch('example_application.GeometryDetectionSystem'):
            dashboard = ShapeAnalysisDashboard()
            
        # Scenario 1: Too many shapes
        many_shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([]),
                center=(i*10, i*10),
                area=100,
                perimeter=40,
                bbox=(0, 0, 20, 20),
                confidence=0.9
            ) for i in range(60)
        ]
        
        dashboard._check_alerts(many_shapes, 30.0)
        alert_found = any("Too many shapes" in alert for alert in dashboard.alerts)
        self.assertTrue(alert_found)
        
        # Clear alerts
        dashboard.alerts.clear()
        
        # Scenario 2: Low FPS
        dashboard._check_alerts([], 5.0)
        alert_found = any("Low FPS" in alert for alert in dashboard.alerts)
        self.assertTrue(alert_found)
        
    def test_data_export_format(self):
        """Test that exported data has correct format"""
        from integrated_geometry_system import GeometricShape, ShapeType
        
        with patch('example_application.GeometryDetectionSystem'):
            dashboard = ShapeAnalysisDashboard()
            
        # Add test data
        shapes = [
            GeometricShape(
                type=ShapeType.CIRCLE,
                contour=np.array([[0, 0], [10, 0], [10, 10]]),
                center=(100, 100),
                area=500,
                perimeter=80,
                bbox=(80, 80, 40, 40),
                confidence=0.9,
                radius=20
            )
        ]
        
        # Process to update statistics
        with patch.object(dashboard, '_draw_visualization', return_value=np.zeros((480, 640, 3))):
            dashboard.process_frame(np.zeros((480, 640, 3)), shapes)
            
        # Export current statistics
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
                # Capture the JSON dump
                saved_data = None
                def capture_dump(data, file, **kwargs):
                    nonlocal saved_data
                    saved_data = data
                    
                with patch('json.dump', side_effect=capture_dump):
                    # Trigger export
                    dashboard.handle_keyboard(ord('e'))
                    
            # Verify export format
            self.assertIsNotNone(saved_data)
            self.assertIn('timestamp', saved_data)
            self.assertIn('statistics', saved_data)
            self.assertIn('total_shapes', saved_data)
            
            # Check statistics structure
            stats = saved_data['statistics']
            self.assertIn('circle', stats)
            self.assertIn('count', stats['circle'])
            self.assertIn('avg_area', stats['circle'])
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)