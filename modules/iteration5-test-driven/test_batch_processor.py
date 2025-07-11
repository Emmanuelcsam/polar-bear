#!/usr/bin/env python3
"""
Unit tests for batch_processor.py
"""
import unittest
import os
import tempfile
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock
import batch_processor
import data_store

class TestBatchProcessor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Create some test images
        self.test_images = []
        for i in range(3):
            img_path = os.path.join(self.temp_dir, f"test_{i}.png")
            img_array = np.random.randint(0, 256, (2, 2), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(img_path)
            self.test_images.append(img_path)

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()

        # Clean up test images
        for img_path in self.test_images:
            if os.path.exists(img_path):
                os.remove(img_path)

        # Remove temp directory
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    @patch('batch_processor.print')
    def test_process_folder_with_images(self, mock_print):
        """Test process_folder with valid images."""
        batch_processor.process_folder(self.temp_dir)

        # Check that all images were processed
        events = data_store.load_events()
        self.assertEqual(len(events), 12)  # 3 images × 4 pixels each

        # Check print messages
        mock_print.assert_any_call("[Batch] Done")
        for img_path in self.test_images:
            mock_print.assert_any_call(f"[Batch] Reading {img_path}")

    @patch('batch_processor.print')
    def test_process_folder_nonexistent(self, mock_print):
        """Test process_folder with non-existent folder."""
        batch_processor.process_folder("/nonexistent/folder")

        mock_print.assert_called_with("[Batch] Error: Folder '/nonexistent/folder' does not exist")

    @patch('batch_processor.print')
    def test_process_folder_no_images(self, mock_print):
        """Test process_folder with no image files."""
        # Create empty directory
        empty_dir = tempfile.mkdtemp()

        try:
            batch_processor.process_folder(empty_dir)
            mock_print.assert_called_with(f"[Batch] No image files found in {empty_dir}")
        finally:
            os.rmdir(empty_dir)

    @patch('batch_processor.read_image')
    @patch('batch_processor.print')
    def test_process_folder_error_handling(self, mock_print, mock_read_image):
        """Test process_folder with error in reading image."""
        # Make read_image raise an exception
        mock_read_image.side_effect = Exception("Test error")

        batch_processor.process_folder(self.temp_dir)

        # Check that error was handled
        for img_path in self.test_images:
            mock_print.assert_any_call(f"[Batch] Error processing {img_path}: Test error")

if __name__ == "__main__":
    unittest.main()
