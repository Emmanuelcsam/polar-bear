"""
Unit tests for batch_processor_refactored module
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock
import batch_processor_refactored as batch_processor


class TestBatchProcessor(unittest.TestCase):
    """Test cases for batch_processor_refactored module"""

    def test_get_image_files(self):
        """Test getting image files from directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            open(os.path.join(tmp_dir, 'image1.jpg'), 'w').close()
            open(os.path.join(tmp_dir, 'image2.png'), 'w').close()
            open(os.path.join(tmp_dir, 'not_image.txt'), 'w').close()
            open(os.path.join(tmp_dir, 'image3.JPEG'), 'w').close()
            open(os.path.join(tmp_dir, 'image4.bmp'), 'w').close()

            image_files = batch_processor.get_image_files(tmp_dir)

            self.assertEqual(len(image_files), 4)
            self.assertIn('image1.jpg', image_files)
            self.assertIn('image2.png', image_files)
            self.assertIn('image3.JPEG', image_files)
            self.assertIn('image4.bmp', image_files)
            self.assertNotIn('not_image.txt', image_files)

    def test_get_image_files_nonexistent_dir(self):
        """Test getting image files from nonexistent directory"""
        result = batch_processor.get_image_files('/nonexistent/path')
        self.assertEqual(result, [])

    def test_get_image_files_empty_dir(self):
        """Test getting image files from empty directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = batch_processor.get_image_files(tmp_dir)
            self.assertEqual(result, [])

    def test_save_load_results(self):
        """Test saving and loading results"""
        test_results = {
            'image1.jpg': {
                'category': 'cat1',
                'confidence': 0.85,
                'scores': {'cat1': 0.8, 'cat2': 0.2},
                'timestamp': '2024-01-01T12:00:00'
            },
            'image2.jpg': {
                'category': 'cat2',
                'confidence': 0.92,
                'scores': {'cat1': 0.1, 'cat2': 0.9},
                'timestamp': '2024-01-01T12:01:00'
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Test saving
            output_file = batch_processor.save_results(test_results, tmp_path)
            self.assertEqual(output_file, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            # Test loading
            loaded_results = batch_processor.load_results(tmp_path)
            self.assertEqual(loaded_results, test_results)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_save_results_auto_filename(self):
        """Test saving results with auto-generated filename"""
        test_results = {'test.jpg': {'category': 'cat1'}}

        try:
            output_file = batch_processor.save_results(test_results)
            self.assertTrue(output_file.startswith('results_'))
            self.assertTrue(output_file.endswith('.json'))
            self.assertTrue(os.path.exists(output_file))

            # Verify content
            loaded_results = batch_processor.load_results(output_file)
            self.assertEqual(loaded_results, test_results)

        finally:
            if 'output_file' in locals() and os.path.exists(output_file):
                os.unlink(output_file)

    def test_load_results_nonexistent_file(self):
        """Test loading results from nonexistent file"""
        with self.assertRaises(ValueError) as context:
            batch_processor.load_results('nonexistent_file.json')
        self.assertIn("Error loading results", str(context.exception))

    def test_get_category_distribution(self):
        """Test category distribution calculation"""
        test_results = {
            'img1.jpg': {'category': 'cat1'},
            'img2.jpg': {'category': 'cat1'},
            'img3.jpg': {'category': 'cat2'},
            'img4.jpg': {'category': 'cat1'},
            'img5.jpg': {'category': 'cat3'},
            'img6.jpg': {'category': 'cat2'}
        }

        distribution = batch_processor.get_category_distribution(test_results)

        self.assertEqual(distribution['cat1'], 3)
        self.assertEqual(distribution['cat2'], 2)
        self.assertEqual(distribution['cat3'], 1)

    def test_get_category_distribution_empty(self):
        """Test category distribution with empty results"""
        distribution = batch_processor.get_category_distribution({})
        self.assertEqual(distribution, {})

    def test_get_category_distribution_missing_category(self):
        """Test category distribution with missing category fields"""
        test_results = {
            'img1.jpg': {'category': 'cat1'},
            'img2.jpg': {},  # Missing category
            'img3.jpg': {'category': 'cat2'}
        }

        distribution = batch_processor.get_category_distribution(test_results)

        self.assertEqual(distribution['cat1'], 1)
        self.assertEqual(distribution['cat2'], 1)
        self.assertEqual(distribution['unknown'], 1)  # Default for missing category

    def test_print_progress_stats(self):
        """Test progress statistics printing"""
        # This test mainly ensures the function runs without errors
        category_dist = {'cat1': 5, 'cat2': 3, 'cat3': 1}

        # Should not raise any exceptions
        batch_processor.print_progress_stats(9, 20, category_dist)

    def test_default_progress_callback(self):
        """Test default progress callback"""
        # Should not raise any exceptions
        batch_processor.default_progress_callback(5, 10, 'test.jpg', 'cat1', 0.85)

    @patch('batch_processor_refactored.ca.analyze_image')
    def test_process_batch_success(self, mock_analyze):
        """Test successful batch processing"""
        # Setup mock
        mock_analyze.return_value = ('cat1', {'cat1': 0.8, 'cat2': 0.2}, 0.8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test images
            open(os.path.join(tmp_dir, 'test1.jpg'), 'w').close()
            open(os.path.join(tmp_dir, 'test2.jpg'), 'w').close()
            open(os.path.join(tmp_dir, 'not_image.txt'), 'w').close()  # Should be ignored

            pixel_db = {'cat1': []}
            weights = {'cat1': 1.0}

            results = batch_processor.process_batch(tmp_dir, pixel_db, weights)

            self.assertEqual(len(results), 2)
            self.assertIn('test1.jpg', results)
            self.assertIn('test2.jpg', results)
            self.assertEqual(results['test1.jpg']['category'], 'cat1')
            self.assertEqual(results['test2.jpg']['category'], 'cat1')
            self.assertEqual(results['test1.jpg']['confidence'], 0.8)
            self.assertIn('timestamp', results['test1.jpg'])
            self.assertIn('path', results['test1.jpg'])

    def test_process_batch_nonexistent_dir(self):
        """Test batch processing with nonexistent directory"""
        with self.assertRaises(ValueError) as context:
            batch_processor.process_batch('/nonexistent/path', {}, {})
        self.assertIn("Batch directory does not exist", str(context.exception))

    def test_process_batch_no_images(self):
        """Test batch processing with no images"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create non-image file
            open(os.path.join(tmp_dir, 'document.txt'), 'w').close()

            with self.assertRaises(ValueError) as context:
                batch_processor.process_batch(tmp_dir, {}, {})
            self.assertIn("No image files found", str(context.exception))

    @patch('batch_processor_refactored.ca.analyze_image')
    def test_process_batch_with_errors(self, mock_analyze):
        """Test batch processing with analysis errors"""
        # Setup mock to raise exception
        mock_analyze.side_effect = Exception("Analysis failed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test image
            open(os.path.join(tmp_dir, 'error_test.jpg'), 'w').close()

            pixel_db = {'cat1': []}
            weights = {'cat1': 1.0}

            results = batch_processor.process_batch(tmp_dir, pixel_db, weights)

            self.assertEqual(len(results), 1)
            self.assertIn('error_test.jpg', results)
            self.assertEqual(results['error_test.jpg']['category'], 'error')
            self.assertEqual(results['error_test.jpg']['confidence'], 0.0)
            self.assertIn('error', results['error_test.jpg'])

    @patch('batch_processor_refactored.ca.analyze_image')
    def test_process_batch_with_callback(self, mock_analyze):
        """Test batch processing with progress callback"""
        # Setup mock
        mock_analyze.return_value = ('cat1', {'cat1': 0.8}, 0.8)

        # Create callback tracker
        callback_calls = []
        def test_callback(current, total, filename, category, confidence):
            callback_calls.append((current, total, filename, category, confidence))

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test images
            open(os.path.join(tmp_dir, 'test1.jpg'), 'w').close()
            open(os.path.join(tmp_dir, 'test2.jpg'), 'w').close()

            pixel_db = {'cat1': []}
            weights = {'cat1': 1.0}

            batch_processor.process_batch(tmp_dir, pixel_db, weights, test_callback)

            # Check callback was called
            self.assertEqual(len(callback_calls), 2)
            self.assertEqual(callback_calls[0][0], 1)  # First call: current = 1
            self.assertEqual(callback_calls[1][0], 2)  # Second call: current = 2
            self.assertEqual(callback_calls[0][1], 2)  # Both calls: total = 2
            self.assertEqual(callback_calls[1][1], 2)


if __name__ == '__main__':
    unittest.main()
