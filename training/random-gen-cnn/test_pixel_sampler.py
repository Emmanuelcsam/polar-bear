"""
Unit tests for pixel_sampler_refactored module
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import tempfile
import numpy as np
from PIL import Image
import pixel_sampler_refactored as pixel_sampler


class TestPixelSampler(unittest.TestCase):
    """Test cases for pixel_sampler_refactored module"""

    def test_is_image_file(self):
        """Test image file detection"""
        self.assertTrue(pixel_sampler.is_image_file('test.jpg'))
        self.assertTrue(pixel_sampler.is_image_file('test.PNG'))
        self.assertTrue(pixel_sampler.is_image_file('test.jpeg'))
        self.assertTrue(pixel_sampler.is_image_file('test.bmp'))
        self.assertTrue(pixel_sampler.is_image_file('test.tiff'))
        self.assertFalse(pixel_sampler.is_image_file('test.txt'))
        self.assertFalse(pixel_sampler.is_image_file('test.py'))
        self.assertFalse(pixel_sampler.is_image_file('test.pdf'))

    def test_load_image_success(self):
        """Test successful image loading"""
        # Create a temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = pixel_sampler.load_image(tmp_path)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (100, 100, 3))
            self.assertEqual(result.dtype, np.uint8)
            # Check that it's roughly red (allowing for some variation)
            self.assertGreater(np.mean(result[:, :, 0]), 200)  # Red channel
            self.assertLess(np.mean(result[:, :, 1]), 50)     # Green channel
            self.assertLess(np.mean(result[:, :, 2]), 50)     # Blue channel
        finally:
            os.unlink(tmp_path)

    def test_load_image_failure(self):
        """Test failed image loading"""
        result = pixel_sampler.load_image('nonexistent_file.jpg')
        self.assertIsNone(result)

    def test_sample_pixels_from_image(self):
        """Test pixel sampling from image"""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Sample pixels
        pixels = pixel_sampler.sample_pixels_from_image(test_image, 10)

        self.assertEqual(len(pixels), 10)
        for pixel in pixels:
            self.assertEqual(pixel.shape, (3,))
            self.assertEqual(pixel.dtype, np.uint8)
            # All values should be in valid range
            self.assertTrue(all(0 <= val <= 255 for val in pixel))

    def test_sample_pixels_edge_cases(self):
        """Test pixel sampling edge cases"""
        # Test with 1x1 image
        tiny_image = np.array([[[255, 128, 64]]], dtype=np.uint8)
        pixels = pixel_sampler.sample_pixels_from_image(tiny_image, 5)
        self.assertEqual(len(pixels), 5)
        # All pixels should be the same since there's only one pixel
        for pixel in pixels:
            np.testing.assert_array_equal(pixel, [255, 128, 64])

    def test_build_pixel_database_nonexistent_dir(self):
        """Test pixel database building with nonexistent directory"""
        with self.assertRaises(ValueError) as context:
            pixel_sampler.build_pixel_database('/nonexistent/path')
        self.assertIn("Reference directory does not exist", str(context.exception))

    def test_build_pixel_database_success(self):
        """Test successful pixel database building"""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create category directories
            cat1_dir = os.path.join(tmp_dir, 'category1')
            cat2_dir = os.path.join(tmp_dir, 'category2')
            os.makedirs(cat1_dir)
            os.makedirs(cat2_dir)

            # Create test images
            img1 = Image.new('RGB', (50, 50), color='red')
            img2 = Image.new('RGB', (50, 50), color='blue')
            img1.save(os.path.join(cat1_dir, 'test1.jpg'))
            img2.save(os.path.join(cat2_dir, 'test2.jpg'))

            # Build database
            pixel_db = pixel_sampler.build_pixel_database(tmp_dir, sample_size=5)

            self.assertEqual(len(pixel_db), 2)
            self.assertIn('category1', pixel_db)
            self.assertIn('category2', pixel_db)
            self.assertEqual(len(pixel_db['category1']), 5)
            self.assertEqual(len(pixel_db['category2']), 5)

    def test_build_pixel_database_nested_structure(self):
        """Test pixel database with nested directory structure"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create nested structure
            nested_dir = os.path.join(tmp_dir, 'parent', 'child')
            os.makedirs(nested_dir)

            # Create test image
            img = Image.new('RGB', (30, 30), color='green')
            img.save(os.path.join(nested_dir, 'test.jpg'))

            # Build database
            pixel_db = pixel_sampler.build_pixel_database(tmp_dir, sample_size=3)

            self.assertEqual(len(pixel_db), 1)
            # Should have the relative path as category
            expected_category = os.path.join('parent', 'child')
            self.assertIn(expected_category, pixel_db)
            self.assertEqual(len(pixel_db[expected_category]), 3)

    def test_build_pixel_database_root_level_images(self):
        """Test pixel database with images at root level"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create image at root level
            img = Image.new('RGB', (30, 30), color='yellow')
            img.save(os.path.join(tmp_dir, 'root_image.jpg'))

            # Build database
            pixel_db = pixel_sampler.build_pixel_database(tmp_dir, sample_size=3)

            self.assertEqual(len(pixel_db), 1)
            self.assertIn('root', pixel_db)
            self.assertEqual(len(pixel_db['root']), 3)

    def test_save_load_pixel_database(self):
        """Test saving and loading pixel database"""
        # Create test database
        test_db = {
            'cat1': [np.array([255, 0, 0]), np.array([0, 255, 0])],
            'cat2': [np.array([0, 0, 255]), np.array([255, 255, 0])]
        }

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Test saving
            result = pixel_sampler.save_pixel_database(test_db, tmp_path)
            self.assertTrue(result)
            self.assertTrue(os.path.exists(tmp_path))

            # Test loading
            loaded_db = pixel_sampler.load_pixel_database(tmp_path)
            self.assertIsNotNone(loaded_db)
            self.assertEqual(len(loaded_db), 2)
            self.assertIn('cat1', loaded_db)
            self.assertIn('cat2', loaded_db)

            # Check pixel values
            np.testing.assert_array_equal(loaded_db['cat1'][0], [255, 0, 0])
            np.testing.assert_array_equal(loaded_db['cat1'][1], [0, 255, 0])

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_pixel_database_nonexistent_file(self):
        """Test loading nonexistent pixel database"""
        result = pixel_sampler.load_pixel_database('nonexistent_file.pkl')
        self.assertIsNone(result)

    def test_get_database_stats(self):
        """Test database statistics calculation"""
        test_db = {
            'cat1': [np.array([255, 0, 0]), np.array([0, 255, 0])],
            'cat2': [np.array([0, 0, 255])],
            'cat3': [np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255])]
        }

        stats = pixel_sampler.get_database_stats(test_db)

        self.assertEqual(stats['categories'], 3)
        self.assertEqual(stats['total_pixels'], 6)
        self.assertEqual(stats['pixels_per_category']['cat1'], 2)
        self.assertEqual(stats['pixels_per_category']['cat2'], 1)
        self.assertEqual(stats['pixels_per_category']['cat3'], 3)

    def test_get_database_stats_empty(self):
        """Test database statistics with empty database"""
        empty_db = {}
        stats = pixel_sampler.get_database_stats(empty_db)

        self.assertEqual(stats['categories'], 0)
        self.assertEqual(stats['total_pixels'], 0)
        self.assertEqual(stats['pixels_per_category'], {})


if __name__ == '__main__':
    unittest.main()
