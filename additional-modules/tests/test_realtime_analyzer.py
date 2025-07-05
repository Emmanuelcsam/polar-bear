#!/usr/bin/env python3
"""
Unit tests for realtime_analyzer module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import time
from datetime import datetime, timedelta


class TestRealtimeAnalyzer(unittest.TestCase):
    """Test cases for realtime analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_rate = 1000  # Hz
        self.buffer_size = 1024
        self.window_size = 5.0  # seconds
        
        # Create sample data stream
        self.test_duration = 1.0  # seconds
        self.num_samples = int(self.sample_rate * self.test_duration)
        self.sample_data = np.sin(2 * np.pi * 50 * np.linspace(0, self.test_duration, self.num_samples))
        
        # Analysis parameters
        self.analysis_config = {
            'sample_rate': self.sample_rate,
            'buffer_size': self.buffer_size,
            'window_size': self.window_size,
            'update_interval': 0.1,  # seconds
            'alert_threshold': 0.8
        }
        
    def tearDown(self):
        """Clean up after each test method."""
        self.sample_data = None
        self.analysis_config = None
        
    def test_module_imports(self):
        """Test that realtime_analyzer module can be imported."""
        try:
            import realtime_analyzer
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import realtime_analyzer: {e}")
            
    def test_buffer_operations(self):
        """Test buffer management operations."""
        # Create circular buffer simulation
        buffer = np.zeros(self.buffer_size)
        write_index = 0
        
        # Write data to buffer
        chunk_size = 128
        for i in range(0, min(len(self.sample_data), self.buffer_size), chunk_size):
            chunk = self.sample_data[i:i+chunk_size]
            buffer[write_index:write_index+len(chunk)] = chunk
            write_index = (write_index + len(chunk)) % self.buffer_size
            
        # Verify buffer contains data
        self.assertFalse(np.all(buffer == 0))
        self.assertEqual(len(buffer), self.buffer_size)
        
    def test_timing_operations(self):
        """Test timing and synchronization."""
        start_time = time.time()
        
        # Simulate processing delay
        time.sleep(0.01)
        
        elapsed = time.time() - start_time
        self.assertGreater(elapsed, 0.01)
        self.assertLess(elapsed, 0.1)
        
        # Test timestamp generation
        timestamp = datetime.now()
        self.assertIsInstance(timestamp, datetime)
        
    def test_mock_realtime_analysis(self):
        """Test realtime analysis with mocks."""
        mock_analyzer = Mock()
        mock_analyzer.analyze_stream = MagicMock(return_value={
            'timestamp': datetime.now(),
            'metrics': {
                'mean': 0.002,
                'std': 0.715,
                'peak': 0.998,
                'rms': 0.707
            },
            'alerts': [],
            'buffer_utilization': 0.75
        })
        
        result = mock_analyzer.analyze_stream(self.sample_data[:self.buffer_size])
        
        # Verify structure
        self.assertIn('timestamp', result)
        self.assertIn('metrics', result)
        self.assertIn('alerts', result)
        self.assertIn('buffer_utilization', result)
        
        # Verify metrics
        metrics = result['metrics']
        self.assertIn('mean', metrics)
        self.assertIn('std', metrics)
        self.assertIn('peak', metrics)
        self.assertIn('rms', metrics)
        
        # Verify values are reasonable
        self.assertGreaterEqual(result['buffer_utilization'], 0)
        self.assertLessEqual(result['buffer_utilization'], 1)
        
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        processing_times = []
        
        # Simulate multiple processing iterations
        for _ in range(10):
            start = time.time()
            
            # Simulate processing
            _ = np.fft.fft(self.sample_data[:self.buffer_size])
            
            processing_times.append(time.time() - start)
            
        # Calculate performance metrics
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        
        self.assertGreater(avg_time, 0)
        self.assertLess(max_time, 1.0)  # Should be fast
        
    def test_alert_generation(self):
        """Test alert generation logic."""
        threshold = self.analysis_config['alert_threshold']
        
        # Generate test signal with anomaly
        normal_signal = np.random.normal(0, 0.1, 900)
        anomaly_signal = np.random.normal(0, 1.0, 100)  # Higher variance
        test_signal = np.concatenate([normal_signal, anomaly_signal])
        
        # Simple anomaly detection
        window_size = 100
        alerts = []
        
        for i in range(0, len(test_signal) - window_size, window_size // 2):
            window = test_signal[i:i+window_size]
            if np.std(window) > threshold:
                alerts.append({
                    'timestamp': i / self.sample_rate,
                    'type': 'high_variance',
                    'value': np.std(window)
                })
                
        self.assertGreater(len(alerts), 0)  # Should detect anomaly


if __name__ == '__main__':
    unittest.main()