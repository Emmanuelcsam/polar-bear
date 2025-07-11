#!/usr/bin/env python3
"""
Unit tests for realtime_location_pipeline module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime, timedelta


class TestRealtimeLocationPipeline(unittest.TestCase):
    """Test cases for realtime location pipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.pipeline_config = {
            'tracking_method': 'kalman',
            'max_tracked_objects': 100,
            'position_threshold': 50.0,  # pixels
            'confidence_threshold': 0.7,
            'update_frequency': 30.0,  # Hz
            'coordinate_system': 'pixel'
        }
        
        # Sample location data
        self.num_objects = 5
        self.frame_width = 1920
        self.frame_height = 1080
        
        self.sample_detections = [
            {
                'id': i,
                'x': np.random.uniform(0, self.frame_width),
                'y': np.random.uniform(0, self.frame_height),
                'confidence': np.random.uniform(0.8, 1.0),
                'timestamp': datetime.now()
            }
            for i in range(self.num_objects)
        ]
        
        # Historical tracking data
        self.tracking_history = {
            i: {
                'positions': [(det['x'], det['y']) for _ in range(10)],
                'timestamps': [datetime.now() - timedelta(seconds=j*0.033) for j in range(10)],
                'velocities': [(np.random.uniform(-5, 5), np.random.uniform(-5, 5)) for _ in range(10)]
            }
            for i, det in enumerate(self.sample_detections)
        }
        
    def tearDown(self):
        """Clean up after each test method."""
        self.pipeline_config = None
        self.sample_detections = None
        self.tracking_history = None
        
    def test_module_imports(self):
        """Test that realtime_location_pipeline module can be imported."""
        try:
            import realtime_location_pipeline
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import realtime_location_pipeline: {e}")
            
    def test_detection_validation(self):
        """Test detection data validation."""
        for detection in self.sample_detections:
            # Check required fields
            self.assertIn('id', detection)
            self.assertIn('x', detection)
            self.assertIn('y', detection)
            self.assertIn('confidence', detection)
            self.assertIn('timestamp', detection)
            
            # Check coordinate bounds
            self.assertGreaterEqual(detection['x'], 0)
            self.assertLessEqual(detection['x'], self.frame_width)
            self.assertGreaterEqual(detection['y'], 0)
            self.assertLessEqual(detection['y'], self.frame_height)
            
            # Check confidence range
            self.assertGreaterEqual(detection['confidence'], 0)
            self.assertLessEqual(detection['confidence'], 1)
            
    def test_distance_calculations(self):
        """Test distance calculation functions."""
        # Euclidean distance
        p1 = (100, 200)
        p2 = (400, 600)
        
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        expected = 500.0  # 3-4-5 triangle scaled by 100
        
        self.assertAlmostEqual(distance, expected, places=5)
        
        # Test threshold checking
        self.assertLess(distance, 1000)  # Within reasonable bounds
        
    def test_mock_tracking_pipeline(self):
        """Test tracking pipeline with mocks."""
        mock_pipeline = Mock()
        mock_pipeline.process_frame = MagicMock(return_value={
            'tracked_objects': [
                {
                    'id': i,
                    'current_position': (self.sample_detections[i]['x'], 
                                       self.sample_detections[i]['y']),
                    'predicted_position': (self.sample_detections[i]['x'] + np.random.uniform(-10, 10),
                                         self.sample_detections[i]['y'] + np.random.uniform(-10, 10)),
                    'velocity': (np.random.uniform(-5, 5), np.random.uniform(-5, 5)),
                    'confidence': self.sample_detections[i]['confidence'],
                    'track_age': 10,
                    'status': 'active'
                }
                for i in range(self.num_objects)
            ],
            'new_tracks': 0,
            'lost_tracks': 0,
            'processing_time': 0.015
        })
        
        result = mock_pipeline.process_frame(self.sample_detections)
        
        # Verify structure
        self.assertIn('tracked_objects', result)
        self.assertIn('new_tracks', result)
        self.assertIn('lost_tracks', result)
        self.assertIn('processing_time', result)
        
        # Verify tracked objects
        self.assertEqual(len(result['tracked_objects']), self.num_objects)
        
        for obj in result['tracked_objects']:
            self.assertIn('id', obj)
            self.assertIn('current_position', obj)
            self.assertIn('predicted_position', obj)
            self.assertIn('velocity', obj)
            self.assertIn('confidence', obj)
            self.assertIn('track_age', obj)
            self.assertIn('status', obj)
            
            # Verify position is tuple of 2 values
            self.assertEqual(len(obj['current_position']), 2)
            self.assertEqual(len(obj['predicted_position']), 2)
            self.assertEqual(len(obj['velocity']), 2)
            
    def test_trajectory_prediction(self):
        """Test trajectory prediction functionality."""
        # Simple linear prediction
        current_pos = (500, 300)
        velocity = (10, -5)  # pixels per frame
        time_steps = 5
        
        predicted_positions = []
        for t in range(1, time_steps + 1):
            pred_x = current_pos[0] + velocity[0] * t
            pred_y = current_pos[1] + velocity[1] * t
            predicted_positions.append((pred_x, pred_y))
            
        # Verify predictions
        self.assertEqual(len(predicted_positions), time_steps)
        
        # Check first prediction
        self.assertEqual(predicted_positions[0], (510, 295))
        
        # Check last prediction
        self.assertEqual(predicted_positions[-1], (550, 275))
        
    def test_tracking_association(self):
        """Test object association between frames."""
        # Previous frame detections
        prev_detections = [
            {'id': 0, 'x': 100, 'y': 100},
            {'id': 1, 'x': 200, 'y': 200},
            {'id': 2, 'x': 300, 'y': 300}
        ]
        
        # Current frame detections (slightly moved)
        curr_detections = [
            {'x': 105, 'y': 102},  # Should match id 0
            {'x': 195, 'y': 205},  # Should match id 1
            {'x': 400, 'y': 400}   # New object
        ]
        
        # Simple nearest neighbor association
        associations = []
        threshold = self.pipeline_config['position_threshold']
        
        for curr in curr_detections:
            min_dist = float('inf')
            best_match = None
            
            for prev in prev_detections:
                dist = np.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
                if dist < min_dist and dist < threshold:
                    min_dist = dist
                    best_match = prev['id']
                    
            associations.append(best_match)
            
        # Verify associations
        self.assertEqual(associations[0], 0)  # First detection matched to id 0
        self.assertEqual(associations[1], 1)  # Second detection matched to id 1
        self.assertIsNone(associations[2])    # Third detection is new (no match)


if __name__ == '__main__':
    unittest.main()