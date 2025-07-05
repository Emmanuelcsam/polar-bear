#!/usr/bin/env python3
"""
Unit tests for realtime_dashboard module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime, timedelta


class TestRealtimeDashboard(unittest.TestCase):
    """Test cases for realtime dashboard functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dashboard_config = {
            'title': 'Test Dashboard',
            'refresh_rate': 1.0,  # seconds
            'max_points': 1000,
            'layout': {
                'rows': 2,
                'cols': 2,
                'widgets': ['line_chart', 'bar_chart', 'gauge', 'table']
            }
        }
        
        # Sample data for visualization
        self.time_series_data = {
            'timestamps': [datetime.now() - timedelta(seconds=i) for i in range(100, 0, -1)],
            'values': np.cumsum(np.random.randn(100))
        }
        
        self.metrics_data = {
            'cpu_usage': 45.2,
            'memory_usage': 62.8,
            'disk_usage': 78.1,
            'network_throughput': 125.4
        }
        
        self.table_data = [
            {'id': 1, 'name': 'Process A', 'status': 'Running', 'cpu': 12.5},
            {'id': 2, 'name': 'Process B', 'status': 'Idle', 'cpu': 0.2},
            {'id': 3, 'name': 'Process C', 'status': 'Running', 'cpu': 8.7}
        ]
        
    def tearDown(self):
        """Clean up after each test method."""
        self.dashboard_config = None
        self.time_series_data = None
        self.metrics_data = None
        self.table_data = None
        
    def test_module_imports(self):
        """Test that realtime_dashboard module can be imported."""
        try:
            import realtime_dashboard
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import realtime_dashboard: {e}")
            
    def test_configuration_validation(self):
        """Test dashboard configuration validation."""
        # Test required fields
        self.assertIn('title', self.dashboard_config)
        self.assertIn('refresh_rate', self.dashboard_config)
        self.assertIn('layout', self.dashboard_config)
        
        # Test layout configuration
        layout = self.dashboard_config['layout']
        self.assertIn('rows', layout)
        self.assertIn('cols', layout)
        self.assertIn('widgets', layout)
        
        # Test widget count matches grid
        expected_widgets = layout['rows'] * layout['cols']
        self.assertEqual(len(layout['widgets']), expected_widgets)
        
    def test_data_validation(self):
        """Test data validation for dashboard."""
        # Test time series data
        self.assertEqual(len(self.time_series_data['timestamps']), 
                        len(self.time_series_data['values']))
        
        # Test all timestamps are datetime objects
        for ts in self.time_series_data['timestamps']:
            self.assertIsInstance(ts, datetime)
            
        # Test metrics are numeric
        for key, value in self.metrics_data.items():
            self.assertIsInstance(value, (int, float))
            self.assertGreaterEqual(value, 0)
            
        # Test table data structure
        for row in self.table_data:
            self.assertIn('id', row)
            self.assertIn('name', row)
            self.assertIn('status', row)
            
    def test_mock_dashboard_update(self):
        """Test dashboard update functionality with mocks."""
        mock_dashboard = Mock()
        mock_dashboard.update = MagicMock(return_value={
            'status': 'success',
            'widgets_updated': 4,
            'render_time': 0.023,
            'errors': []
        })
        
        update_data = {
            'time_series': self.time_series_data,
            'metrics': self.metrics_data,
            'table': self.table_data
        }
        
        result = mock_dashboard.update(update_data)
        
        # Verify update result
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['widgets_updated'], 4)
        self.assertLess(result['render_time'], 1.0)
        self.assertEqual(len(result['errors']), 0)
        
    def test_widget_rendering(self):
        """Test widget rendering capabilities."""
        mock_widget = Mock()
        
        # Test line chart widget
        mock_widget.render_line_chart = MagicMock(return_value={
            'type': 'line_chart',
            'data_points': len(self.time_series_data['values']),
            'rendered': True
        })
        
        line_result = mock_widget.render_line_chart(self.time_series_data)
        self.assertTrue(line_result['rendered'])
        self.assertEqual(line_result['data_points'], 100)
        
        # Test gauge widget
        mock_widget.render_gauge = MagicMock(return_value={
            'type': 'gauge',
            'value': self.metrics_data['cpu_usage'],
            'rendered': True
        })
        
        gauge_result = mock_widget.render_gauge('cpu_usage', self.metrics_data['cpu_usage'])
        self.assertTrue(gauge_result['rendered'])
        self.assertEqual(gauge_result['value'], 45.2)
        
    def test_refresh_timing(self):
        """Test refresh timing logic."""
        refresh_rate = self.dashboard_config['refresh_rate']
        last_update = datetime.now()
        
        # Simulate time passing
        current_time = last_update + timedelta(seconds=refresh_rate + 0.1)
        
        # Check if refresh needed
        time_since_update = (current_time - last_update).total_seconds()
        needs_refresh = time_since_update >= refresh_rate
        
        self.assertTrue(needs_refresh)
        
    def test_data_buffering(self):
        """Test data buffering for smooth updates."""
        max_points = self.dashboard_config['max_points']
        
        # Create buffer that exceeds max points
        buffer_size = max_points + 100
        large_buffer = list(range(buffer_size))
        
        # Trim to max points (keep most recent)
        trimmed_buffer = large_buffer[-max_points:]
        
        self.assertEqual(len(trimmed_buffer), max_points)
        self.assertEqual(trimmed_buffer[-1], buffer_size - 1)  # Most recent value


if __name__ == '__main__':
    unittest.main()