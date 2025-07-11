#!/usr/bin/env python3
"""
Unit tests for pixel_generator.py
"""
import unittest
import random
from unittest.mock import patch, MagicMock
import pixel_generator
import data_store

class TestPixelGenerator(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()

    @patch('pixel_generator.time.sleep')
    @patch('pixel_generator.random.randint')
    def test_generate_pixels_limited(self, mock_randint, mock_sleep):
        """Test generate_pixels with limited iterations."""
        mock_randint.return_value = 128

        # Generate 3 pixels
        pixel_generator.generate_pixels(interval=0.01, max_iterations=3)

        # Check that randint was called 3 times
        self.assertEqual(mock_randint.call_count, 3)

        # Check that sleep was called 3 times
        self.assertEqual(mock_sleep.call_count, 3)

        # Check that events were saved
        events = data_store.load_events()
        self.assertEqual(len(events), 3)
        for event in events:
            self.assertEqual(event["pixel"], 128)

    @patch('pixel_generator.time.sleep')
    def test_generate_pixels_random_values(self, mock_sleep):
        """Test that generate_pixels creates random values in correct range."""
        # Set a fixed seed for reproducible testing
        random.seed(42)

        # Generate 5 pixels
        pixel_generator.generate_pixels(interval=0.01, max_iterations=5)

        # Check that events were saved
        events = data_store.load_events()
        self.assertEqual(len(events), 5)

        # Check that all pixel values are in valid range
        for event in events:
            self.assertIn("pixel", event)
            self.assertGreaterEqual(event["pixel"], 0)
            self.assertLessEqual(event["pixel"], 255)

    @patch('pixel_generator.time.sleep')
    def test_generate_pixels_interval(self, mock_sleep):
        """Test that generate_pixels respects the interval parameter."""
        # Generate 2 pixels with specific interval
        pixel_generator.generate_pixels(interval=0.5, max_iterations=2)

        # Check that sleep was called with correct interval
        mock_sleep.assert_called_with(0.5)
        self.assertEqual(mock_sleep.call_count, 2)

if __name__ == "__main__":
    unittest.main()
