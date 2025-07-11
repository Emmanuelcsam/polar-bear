"""
Unit tests for correlation_analyzer_refactored module
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import tempfile
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
import correlation_analyzer_refactored as correlation_analyzer


class TestCorrelationAnalyzer(unittest.TestCase):
    """Test cases for correlation_analyzer_refactored module"""
    
    def test_calculate_pixel_similarity_identical(self):
        """Test pixel similarity calculation with identical pixels"""
        pixel1 = np.array([255, 0, 0])  # Red
        pixel2 = np.array([255, 0, 0])  # Red (identical)
        
        similarity = correlation_analyzer.calculate_pixel_similarity(pixel1, pixel2)
        self.assertEqual(similarity, 1.0)
    
    def test_calculate_pixel_similarity_different(self):
        """Test pixel similarity calculation with different pixels"""
        pixel1 = np.array([255, 0, 0])  # Red
        pixel2 = np.array([0, 255, 0])  # Green (very different)
        
        similarity = correlation_analyzer.calculate_pixel_similarity(pixel1, pixel2)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
    
    def test_calculate_pixel_similarity_similar(self):
        """Test pixel similarity calculation with similar pixels"""
        pixel1 = np.array([255, 0, 0])  # Red
        pixel2 = np.array([250, 5, 5])  # Slightly different red
        
        similarity = correlation_analyzer.calculate_pixel_similarity(pixel1, pixel2)
        self.assertGreater(similarity, 0.8)  # Should be quite similar
    
    def test_load_save_weights(self):
        """Test weights loading and saving"""
        test_weights = {'cat1': 1.5, 'cat2': 0.8, 'cat3': 1.2}
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test saving
            result = correlation_analyzer.save_weights(test_weights, tmp_path)
            self.assertTrue(result)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Test loading
            loaded_weights = correlation_analyzer.load_weights(tmp_path)
            self.assertEqual(loaded_weights, test_weights)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_weights_nonexistent_file(self):
        """Test loading weights from nonexistent file"""
        result = correlation_analyzer.load_weights('nonexistent_file.pkl')
        self.assertEqual(result, {})
    
    def test_analyze_image_nonexistent_file(self):
        """Test image analysis with nonexistent file"""
        pixel_db = {'cat1': [np.array([255, 0, 0])]}
        weights = {'cat1': 1.0}
        
        with self.assertRaises(ValueError) as context:
            correlation_analyzer.analyze_image('nonexistent.jpg', pixel_db, weights)
        self.assertIn("Image file does not exist", str(context.exception))
    
    def test_analyze_image_success(self):
        """Test successful image analysis"""
        # Create temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Create test database - red pixels should match better
            pixel_db = {
                'red': [np.array([255, 0, 0]), np.array([250, 5, 5]), np.array([240, 10, 10])],
                'blue': [np.array([0, 0, 255]), np.array([5, 5, 250]), np.array([10, 10, 240])]
            }
            weights = {'red': 1.0, 'blue': 1.0}
            
            category, scores, confidence = correlation_analyzer.analyze_image(
                tmp_path, pixel_db, weights, comparisons=20
            )
            
            # Should classify as red since image is red
            self.assertEqual(category, 'red')
            self.assertIn('red', scores)
            self.assertIn('blue', scores)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            self.assertGreater(scores['red'], scores['blue'])
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_analyze_image_empty_pixel_db(self):
        """Test image analysis with empty pixel database"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('RGB', (50, 50), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            pixel_db = {}
            weights = {}
            
            with self.assertRaises(ValueError) as context:
                correlation_analyzer.analyze_image(tmp_path, pixel_db, weights)
            self.assertIn("No valid categories found", str(context.exception))
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_analyze_image_empty_category(self):
        """Test image analysis with empty category in pixel database"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('RGB', (50, 50), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            pixel_db = {
                'red': [np.array([255, 0, 0])],
                'empty': []  # Empty category
            }
            weights = {'red': 1.0, 'empty': 1.0}
            
            category, scores, confidence = correlation_analyzer.analyze_image(
                tmp_path, pixel_db, weights, comparisons=10
            )
            
            # Should only score the non-empty category
            self.assertEqual(category, 'red')
            self.assertIn('red', scores)
            self.assertNotIn('empty', scores)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_update_weights_from_feedback(self):
        """Test weight updates from feedback"""
        initial_weights = {'cat1': 1.0, 'cat2': 1.0, 'cat3': 1.0}
        
        # Test feedback where prediction was wrong
        updated_weights = correlation_analyzer.update_weights_from_feedback(
            initial_weights, 'cat1', 'cat2', learning_rate=0.1
        )
        
        self.assertGreater(updated_weights['cat2'], initial_weights['cat2'])  # Correct category boosted
        self.assertLess(updated_weights['cat1'], initial_weights['cat1'])    # Predicted category reduced
        self.assertEqual(updated_weights['cat3'], initial_weights['cat3'])   # Unchanged
    
    def test_update_weights_new_category(self):
        """Test weight updates when correct category is new"""
        initial_weights = {'cat1': 1.0, 'cat2': 1.0}
        
        updated_weights = correlation_analyzer.update_weights_from_feedback(
            initial_weights, 'cat1', 'cat3', learning_rate=0.2
        )
        
        self.assertIn('cat3', updated_weights)  # New category added
        self.assertEqual(updated_weights['cat3'], 1.2)  # New category gets boost
        self.assertLess(updated_weights['cat1'], initial_weights['cat1'])  # Predicted reduced
    
    def test_get_image_files_from_directory(self):
        """Test getting image files from directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            open(os.path.join(tmp_dir, 'image1.jpg'), 'w').close()
            open(os.path.join(tmp_dir, 'image2.png'), 'w').close()
            open(os.path.join(tmp_dir, 'document.txt'), 'w').close()
            open(os.path.join(tmp_dir, 'image3.JPEG'), 'w').close()
            open(os.path.join(tmp_dir, 'image4.bmp'), 'w').close()
            
            image_files = correlation_analyzer.get_image_files_from_directory(tmp_dir)
            
            self.assertEqual(len(image_files), 4)
            self.assertTrue(any('image1.jpg' in f for f in image_files))
            self.assertTrue(any('image2.png' in f for f in image_files))
            self.assertTrue(any('image3.JPEG' in f for f in image_files))
            self.assertTrue(any('image4.bmp' in f for f in image_files))
            self.assertFalse(any('document.txt' in f for f in image_files))
    
    def test_get_image_files_nonexistent_directory(self):
        """Test getting image files from nonexistent directory"""
        result = correlation_analyzer.get_image_files_from_directory('/nonexistent/path')
        self.assertEqual(result, [])
    
    def test_batch_analyze_images(self):
        """Test batch analysis of multiple images"""
        # Create test images
        image_paths = []
        for i, color in enumerate(['red', 'blue']):
            with tempfile.NamedTemporaryFile(suffix=f'_{i}.jpg', delete=False) as tmp:
                img = Image.new('RGB', (30, 30), color=color)
                img.save(tmp.name)
                image_paths.append(tmp.name)
        
        try:
            pixel_db = {
                'red': [np.array([255, 0, 0]), np.array([250, 5, 5])],
                'blue': [np.array([0, 0, 255]), np.array([5, 5, 250])]
            }
            weights = {'red': 1.0, 'blue': 1.0}
            
            results = correlation_analyzer.batch_analyze_images(
                image_paths, pixel_db, weights, comparisons=10
            )
            
            self.assertEqual(len(results), 2)
            for path in image_paths:
                self.assertIn(path, results)
                self.assertIn('category', results[path])
                self.assertIn('scores', results[path])
                self.assertIn('confidence', results[path])
            
        finally:
            for path in image_paths:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_batch_analyze_images_with_errors(self):
        """Test batch analysis with some problematic images"""
        # Mix of valid and invalid paths
        image_paths = ['/nonexistent/image1.jpg', '/nonexistent/image2.jpg']
        
        pixel_db = {'cat1': [np.array([255, 0, 0])]}
        weights = {'cat1': 1.0}
        
        results = correlation_analyzer.batch_analyze_images(
            image_paths, pixel_db, weights
        )
        
        self.assertEqual(len(results), 2)
        for path in image_paths:
            self.assertIn(path, results)
            self.assertEqual(results[path]['category'], 'error')
            self.assertEqual(results[path]['confidence'], 0.0)
            self.assertIn('error', results[path])


if __name__ == '__main__':
    unittest.main()
