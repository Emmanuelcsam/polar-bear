#!/usr/bin/env python3
"""
Unit tests for intensity_reader.py
"""
import unittest
import os
import tempfile
from PIL import Image
import numpy as np
from unittest.mock import patch
import intensity_reader
import data_store

class TestIntensityReader(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

        # Create a simple test image
        self.test_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        self.test_image.close()

        # Create a 2x2 grayscale image
        img_array = np.array([[100, 150], [200, 50]], dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(self.test_image.name)

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()
        if os.path.exists(self.test_image.name):
            os.remove(self.test_image.name)

    def test_read_image_saves_intensities(self):
        """Test that read_image saves pixel intensities to data store."""
        intensity_reader.read_image(self.test_image.name)

        # Check that events were saved
        events = data_store.load_events()
        self.assertEqual(len(events), 4)  # 2x2 image = 4 pixels

        # Check that all events have intensity values
        intensities = [event["intensity"] for event in events]
        expected_intensities = [100, 150, 200, 50]  # Row-major order
        self.assertEqual(intensities, expected_intensities)

    def test_read_image_nonexistent_file(self):
        """Test read_image with non-existent file."""
        with self.assertRaises(Exception):
            intensity_reader.read_image("nonexistent_file.png")

    @patch('intensity_reader.print')
    def test_read_image_prints_values(self, mock_print):
        """Test that read_image prints pixel values."""
        intensity_reader.read_image(self.test_image.name)

        # Check that print was called for each pixel
        self.assertEqual(mock_print.call_count, 4)

        # Check print messages
        expected_calls = [
            "[Intensity] pixel#0 = 100",
            "[Intensity] pixel#1 = 150",
            "[Intensity] pixel#2 = 200",
            "[Intensity] pixel#3 = 50"
        ]

        for i, expected_call in enumerate(expected_calls):
            mock_print.assert_any_call(expected_call)

if __name__ == "__main__":
    unittest.main()
