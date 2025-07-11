#!/usr/bin/env python3
"""
Unit tests for geometry_analyzer.py
"""
import unittest
import numpy as np
from unittest.mock import patch
import geometry_analyzer
import data_store

class TestGeometryAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()

    def test_analyze_geometry_with_data(self):
        """Test analyze_geometry with intensity data."""
        # Create a 2x2 intensity pattern
        intensities = [100, 200, 150, 250]  # Creates a gradient
        for intensity in intensities:
            data_store.save_event({"intensity": intensity})

        gx, gy = geometry_analyzer.analyze_geometry(width=2, height=2)

        # Check that gradients were computed
        self.assertIsNotNone(gx)
        self.assertIsNotNone(gy)
        self.assertEqual(gx.shape, (2, 2))
        self.assertEqual(gy.shape, (2, 2))

        # Check gradient values (numpy gradient computes central differences)
        expected_arr = np.array([[100, 200], [150, 250]])
        expected_gx, expected_gy = np.gradient(expected_arr)

        np.testing.assert_array_equal(gx, expected_gx)
        np.testing.assert_array_equal(gy, expected_gy)

    def test_analyze_geometry_no_data(self):
        """Test analyze_geometry with no intensity data."""
        gx, gy = geometry_analyzer.analyze_geometry(width=2, height=2)

        # Should return None for both gradients
        self.assertIsNone(gx)
        self.assertIsNone(gy)

    def test_analyze_geometry_insufficient_data(self):
        """Test analyze_geometry with insufficient data."""
        # Save only 2 intensity events for a 2x2 image
        data_store.save_event({"intensity": 100})
        data_store.save_event({"intensity": 200})

        gx, gy = geometry_analyzer.analyze_geometry(width=2, height=2)

        # Should pad with zeros and compute gradients
        self.assertEqual(gx.shape, (2, 2))
        self.assertEqual(gy.shape, (2, 2))

        # Check that padding worked
        expected_arr = np.array([[100, 200], [0, 0]])
        expected_gx, expected_gy = np.gradient(expected_arr)

        np.testing.assert_array_equal(gx, expected_gx)
        np.testing.assert_array_equal(gy, expected_gy)

    def test_analyze_geometry_mixed_events(self):
        """Test analyze_geometry with mixed event types."""
        # Save mixed events
        data_store.save_event({"pixel": 255})
        data_store.save_event({"intensity": 100})
        data_store.save_event({"intensity": 200})
        data_store.save_event({"other": "data"})

        gx, gy = geometry_analyzer.analyze_geometry(width=2, height=2)

        # Should only use intensity events
        self.assertEqual(gx.shape, (2, 2))
        self.assertEqual(gy.shape, (2, 2))

        # Check that only intensity values were used
        expected_arr = np.array([[100, 200], [0, 0]])
        expected_gx, expected_gy = np.gradient(expected_arr)

        np.testing.assert_array_equal(gx, expected_gx)
        np.testing.assert_array_equal(gy, expected_gy)

    @patch('geometry_analyzer.print')
    def test_analyze_geometry_prints_results(self, mock_print):
        """Test that analyze_geometry prints results."""
        # Save some intensity data
        data_store.save_event({"intensity": 100})
        data_store.save_event({"intensity": 200})

        geometry_analyzer.analyze_geometry(width=2, height=2)

        # Check that gradients were printed
        self.assertTrue(any("[Geom] ∂x:" in str(call) for call in mock_print.call_args_list))
        self.assertTrue(any("[Geom] ∂y:" in str(call) for call in mock_print.call_args_list))

    @patch('geometry_analyzer.print')
    def test_analyze_geometry_no_data_message(self, mock_print):
        """Test message when no data is found."""
        geometry_analyzer.analyze_geometry()

        mock_print.assert_called_with("[Geom] No intensity data found")

if __name__ == "__main__":
    unittest.main()
