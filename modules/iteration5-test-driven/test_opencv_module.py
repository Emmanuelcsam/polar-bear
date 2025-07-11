#!/usr/bin/env python3
"""
Unit tests for opencv_module.py
"""
import unittest
import os
import tempfile
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
import opencv_module

class TestOpenCVModule(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Create a test image
        self.test_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        self.test_image.close()

        # Create a simple test image
        img_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(self.test_image.name)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_image.name):
            os.remove(self.test_image.name)

    @patch('opencv_module.OPENCV_AVAILABLE', False)
    def test_cv_analyze_not_available(self):
        """Test cv_analyze when OpenCV is not available."""
        result = opencv_module.cv_analyze(self.test_image.name)

        # Should return None
        self.assertIsNone(result)

    @patch('opencv_module.OPENCV_AVAILABLE', True)
    def test_cv_analyze_with_mock_opencv(self):
        """Test cv_analyze with mocked OpenCV."""
        with patch('opencv_module.cv2') as mock_cv2:
            # Mock image loading
            mock_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            mock_cv2.imread.return_value = mock_img
            mock_cv2.IMREAD_GRAYSCALE = 0

            # Mock edge detection
            mock_edges = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            mock_cv2.Canny.return_value = mock_edges

            result = opencv_module.cv_analyze(self.test_image.name)

            # Check that OpenCV functions were called
            mock_cv2.imread.assert_called_once_with(self.test_image.name, mock_cv2.IMREAD_GRAYSCALE)
            mock_cv2.Canny.assert_called_once_with(mock_img, 100, 200)

            # Check result
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result, mock_edges)

    def test_cv_analyze_with_real_opencv(self):
        """Test cv_analyze with real OpenCV if available."""
        # Skip if OpenCV not available
        if not opencv_module.OPENCV_AVAILABLE:
            self.skipTest("OpenCV not available")

        result = opencv_module.cv_analyze(self.test_image.name)

        # Should return edge array
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 2)  # Should be 2D array

    def test_cv_analyze_nonexistent_file(self):
        """Test cv_analyze with non-existent file."""
        # Skip if OpenCV not available
        if not opencv_module.OPENCV_AVAILABLE:
            self.skipTest("OpenCV not available")

        result = opencv_module.cv_analyze("nonexistent_file.png")

        # Should return None
        self.assertIsNone(result)

    @patch('opencv_module.OPENCV_AVAILABLE', True)
    @patch('opencv_module.print')
    def test_cv_analyze_prints_shape(self, mock_print, ):
        """Test that cv_analyze prints edge shape."""
        with patch('opencv_module.cv2') as mock_cv2:
            # Mock image loading
            mock_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            mock_cv2.imread.return_value = mock_img
            mock_cv2.IMREAD_GRAYSCALE = 0

            # Mock edge detection
            mock_edges = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            mock_cv2.Canny.return_value = mock_edges

            opencv_module.cv_analyze(self.test_image.name)

            # Check that shape was printed
            mock_print.assert_called_with(f"[OpenCV] edges shape: {mock_edges.shape}")

    @patch('opencv_module.OPENCV_AVAILABLE', True)
    @patch('opencv_module.print')
    def test_cv_analyze_error_handling(self, mock_print):
        """Test cv_analyze error handling."""
        with patch('opencv_module.cv2') as mock_cv2:
            # Mock cv2.imread to raise an exception
            mock_cv2.imread.side_effect = Exception("Test error")

            result = opencv_module.cv_analyze(self.test_image.name)

            # Should return None and print error
            self.assertIsNone(result)
            mock_print.assert_called_with("[OpenCV] Error: Test error")

if __name__ == "__main__":
    unittest.main()
