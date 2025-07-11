#!/usr/bin/env python3
"""
Unit tests for image_guided_generator.py
"""
import unittest
import numpy as np
from unittest.mock import patch
import image_guided_generator
import data_store

class TestImageGuidedGenerator(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()

    def test_generate_image_with_data(self):
        """Test generate_image with intensity data."""
        # Save some intensity events
        intensities = [100, 150, 200, 50]
        for intensity in intensities:
            data_store.save_event({"intensity": intensity})

        # Generate 2x2 image
        result = image_guided_generator.generate_image(width=2, height=2)

        # Check result shape and values
        self.assertEqual(result.shape, (2, 2))
        expected = np.array([[100, 150], [200, 50]])
        np.testing.assert_array_equal(result, expected)

    def test_generate_image_no_data(self):
        """Test generate_image with no intensity data."""
        result = image_guided_generator.generate_image(width=2, height=2)

        # Should return array of zeros
        self.assertEqual(result.shape, (2, 2))
        expected = np.zeros((2, 2))
        np.testing.assert_array_equal(result, expected)

    def test_generate_image_insufficient_data(self):
        """Test generate_image with insufficient intensity data."""
        # Save only 2 intensity events for a 2x2 image
        data_store.save_event({"intensity": 100})
        data_store.save_event({"intensity": 150})

        result = image_guided_generator.generate_image(width=2, height=2)

        # Should pad with zeros
        self.assertEqual(result.shape, (2, 2))
        expected = np.array([[100, 150], [0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_generate_image_excess_data(self):
        """Test generate_image with more data than needed."""
        # Save 6 intensity events for a 2x2 image
        intensities = [100, 150, 200, 50, 75, 125]
        for intensity in intensities:
            data_store.save_event({"intensity": intensity})

        result = image_guided_generator.generate_image(width=2, height=2)

        # Should use only first 4 values
        self.assertEqual(result.shape, (2, 2))
        expected = np.array([[100, 150], [200, 50]])
        np.testing.assert_array_equal(result, expected)

    def test_generate_image_mixed_events(self):
        """Test generate_image with mixed event types."""
        # Save mixed events
        data_store.save_event({"pixel": 255})
        data_store.save_event({"intensity": 100})
        data_store.save_event({"intensity": 150})
        data_store.save_event({"other": "data"})

        result = image_guided_generator.generate_image(width=2, height=2)

        # Should only use intensity events
        self.assertEqual(result.shape, (2, 2))
        expected = np.array([[100, 150], [0, 0]])
        np.testing.assert_array_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
