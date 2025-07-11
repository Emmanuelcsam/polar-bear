#!/usr/bin/env python3
"""
Unit tests for pattern_recognizer.py
"""
import unittest
from unittest.mock import patch
import pattern_recognizer
import data_store

class TestPatternRecognizer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()

    def test_find_patterns_with_data(self):
        """Test find_patterns with various data."""
        # Save some events
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 150})
        data_store.save_event({"pixel": 100})
        data_store.save_event({"intensity": 200})
        data_store.save_event({"intensity": 100})

        result = pattern_recognizer.find_patterns()

        # Check patterns
        expected = {100: 3, 150: 1, 200: 1}
        self.assertEqual(result, expected)

    def test_find_patterns_no_data(self):
        """Test find_patterns with no data."""
        result = pattern_recognizer.find_patterns()

        # Should return empty dict
        self.assertEqual(result, {})

    def test_find_patterns_mixed_events(self):
        """Test find_patterns with mixed event types."""
        # Save events with various keys
        data_store.save_event({"pixel": 100})
        data_store.save_event({"intensity": 100})
        data_store.save_event({"other": "data"})
        data_store.save_event({"pixel": 200})

        result = pattern_recognizer.find_patterns()

        # Should count pixel and intensity values, ignore others
        expected = {100: 2, 200: 1}
        self.assertEqual(result, expected)

    @patch('pattern_recognizer.print')
    def test_find_patterns_prints_results(self, mock_print):
        """Test that find_patterns prints results."""
        # Save some events
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 100})
        data_store.save_event({"intensity": 200})

        pattern_recognizer.find_patterns()

        # Check that print was called
        mock_print.assert_any_call("[Pattern] Value 100: 2 times")
        mock_print.assert_any_call("[Pattern] Value 200: 1 times")

    @patch('pattern_recognizer.print')
    def test_find_patterns_no_data_message(self, mock_print):
        """Test message when no data is found."""
        pattern_recognizer.find_patterns()

        mock_print.assert_called_with("[Pattern] No data found")

if __name__ == "__main__":
    unittest.main()
