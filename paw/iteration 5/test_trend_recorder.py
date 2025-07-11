#!/usr/bin/env python3
"""
Unit tests for trend_recorder.py
"""
import unittest
import json
from unittest.mock import patch
import trend_recorder
import data_store

class TestTrendRecorder(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()

    def test_record_trends_with_data(self):
        """Test record_trends with various data."""
        # Save some events
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 200})
        data_store.save_event({"intensity": 50})
        data_store.save_event({"intensity": 250})

        result = trend_recorder.record_trends()

        # Check statistics
        expected = {
            "min": 50,
            "max": 250,
            "mean": 150.0,  # (100 + 200 + 50 + 250) / 4
            "count": 4
        }
        self.assertEqual(result, expected)

    def test_record_trends_no_data(self):
        """Test record_trends with no data."""
        result = trend_recorder.record_trends()

        # Should return None
        self.assertIsNone(result)

    def test_record_trends_single_value(self):
        """Test record_trends with single value."""
        data_store.save_event({"pixel": 123})

        result = trend_recorder.record_trends()

        # All stats should be the same value
        expected = {
            "min": 123,
            "max": 123,
            "mean": 123.0,
            "count": 1
        }
        self.assertEqual(result, expected)

    def test_record_trends_mixed_events(self):
        """Test record_trends with mixed event types."""
        # Save mixed events
        data_store.save_event({"pixel": 100})
        data_store.save_event({"intensity": 200})
        data_store.save_event({"other": "data"})
        data_store.save_event({"pixel": 300})

        result = trend_recorder.record_trends()

        # Should only use pixel and intensity values
        expected = {
            "min": 100,
            "max": 300,
            "mean": 200.0,  # (100 + 200 + 300) / 3
            "count": 3
        }
        self.assertEqual(result, expected)

    @patch('trend_recorder.print')
    def test_record_trends_prints_results(self, mock_print):
        """Test that record_trends prints results."""
        # Save some events
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 200})

        trend_recorder.record_trends()

        # Check that JSON was printed
        expected_stats = {
            "min": 100,
            "max": 200,
            "mean": 150.0,
            "count": 2
        }
        mock_print.assert_called_with("[Trend]", json.dumps(expected_stats))

    @patch('trend_recorder.print')
    def test_record_trends_no_data_message(self, mock_print):
        """Test message when no data is found."""
        trend_recorder.record_trends()

        mock_print.assert_called_with("[Trend] No data")

if __name__ == "__main__":
    unittest.main()
