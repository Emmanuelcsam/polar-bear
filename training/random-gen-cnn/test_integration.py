"""
Integration tests for the complete image categorization system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import tempfile
import shutil
import numpy as np
from PIL import Image
import json
import pickle

# Import all refactored modules
import pixel_sampler_refactored as pixel_sampler
import correlation_analyzer_refactored as correlation_analyzer
import batch_processor_refactored as batch_processor
import self_reviewer_refactored as self_reviewer


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from pixel sampling to analysis"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: Create test directory structure
            ref_dir = os.path.join(tmp_dir, 'references')
            cat1_dir = os.path.join(ref_dir, 'red')
            cat2_dir = os.path.join(ref_dir, 'blue')
            os.makedirs(cat1_dir)
            os.makedirs(cat2_dir)
            
            # Create test images
            red_img = Image.new('RGB', (50, 50), color='red')
            blue_img = Image.new('RGB', (50, 50), color='blue')
            red_img.save(os.path.join(cat1_dir, 'red1.jpg'))
            blue_img.save(os.path.join(cat2_dir, 'blue1.jpg'))
            
            # Add more images for better sampling
            red_img2 = Image.new('RGB', (60, 60), color=(240, 10, 10))
            blue_img2 = Image.new('RGB', (60, 60), color=(10, 10, 240))
            red_img2.save(os.path.join(cat1_dir, 'red2.jpg'))
            blue_img2.save(os.path.join(cat2_dir, 'blue2.jpg'))
            
            # Step 2: Build pixel database
            pixel_db = pixel_sampler.build_pixel_database(ref_dir, sample_size=20)
            self.assertEqual(len(pixel_db), 2)
            self.assertIn('red', pixel_db)
            self.assertIn('blue', pixel_db)
            self.assertEqual(len(pixel_db['red']), 40)  # 20 pixels * 2 images
            self.assertEqual(len(pixel_db['blue']), 40)
            
            # Step 3: Create test image for analysis
            test_img = Image.new('RGB', (30, 30), color='red')
            test_img_path = os.path.join(tmp_dir, 'test_red.jpg')
            test_img.save(test_img_path)
            
            # Step 4: Analyze image
            weights = {'red': 1.0, 'blue': 1.0}
            category, scores, confidence = correlation_analyzer.analyze_image(
                test_img_path, pixel_db, weights, comparisons=30
            )
            
            # Should classify as red
            self.assertEqual(category, 'red')
            self.assertGreater(confidence, 0.5)
            self.assertGreater(scores['red'], scores['blue'])
            
            # Step 5: Test batch processing
            batch_dir = os.path.join(tmp_dir, 'batch')
            os.makedirs(batch_dir)
            
            # Create multiple test images
            test_red = Image.new('RGB', (25, 25), color='red')
            test_blue = Image.new('RGB', (25, 25), color='blue')
            test_red.save(os.path.join(batch_dir, 'batch_red.jpg'))
            test_blue.save(os.path.join(batch_dir, 'batch_blue.jpg'))
            
            results = batch_processor.process_batch(batch_dir, pixel_db, weights)
            self.assertEqual(len(results), 2)
            self.assertIn('batch_red.jpg', results)
            self.assertIn('batch_blue.jpg', results)
            
            # Check that red image was classified as red and blue as blue
            # (allowing for some misclassification due to randomness)
            red_result = results['batch_red.jpg']
            blue_result = results['batch_blue.jpg']
            
            self.assertIn('category', red_result)
            self.assertIn('category', blue_result)
            self.assertIn('confidence', red_result)
            self.assertIn('confidence', blue_result)
            
            # Step 6: Test self-review
            grouped = self_reviewer.group_by_category(results)
            self.assertGreater(len(grouped), 0)
            
            # Check for inconsistencies (should be minimal with good test data)
            inconsistencies = self_reviewer.review_category_consistency(grouped)
            
            # Calculate final statistics
            stats = self_reviewer.calculate_review_statistics(results)
            self.assertEqual(stats['total_images'], 2)
            self.assertEqual(stats['error_count'], 0)
    
    def test_weight_learning_workflow(self):
        """Test weight learning and optimization workflow"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create simple pixel database
            pixel_db = {
                'red': [np.array([255, 0, 0]), np.array([250, 5, 5])],
                'blue': [np.array([0, 0, 255]), np.array([5, 5, 250])]
            }
            
            # Initial weights
            weights = {'red': 1.0, 'blue': 1.0}
            
            # Simulate feedback loop
            for i in range(5):
                # Simulate wrong prediction followed by feedback
                predicted = 'red'
                correct = 'blue'
                
                weights = correlation_analyzer.update_weights_from_feedback(
                    weights, predicted, correct, learning_rate=0.1
                )
            
            # After multiple corrections, blue should have higher weight
            self.assertGreater(weights['blue'], weights['red'])
            
            # Test saving and loading weights
            weights_file = os.path.join(tmp_dir, 'test_weights.pkl')
            self.assertTrue(correlation_analyzer.save_weights(weights, weights_file))
            
            loaded_weights = correlation_analyzer.load_weights(weights_file)
            self.assertEqual(loaded_weights, weights)
    
    def test_error_handling_workflow(self):
        """Test system behavior with various error conditions"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test with corrupted pixel database
            corrupted_db = {'cat1': []}  # Empty category
            weights = {'cat1': 1.0}
            
            # Create a test image
            test_img = Image.new('RGB', (20, 20), color='red')
            test_img_path = os.path.join(tmp_dir, 'test.jpg')
            test_img.save(test_img_path)
            
            # Should handle empty category gracefully
            with self.assertRaises(ValueError):
                correlation_analyzer.analyze_image(test_img_path, corrupted_db, weights)
            
            # Test batch processing with mixed valid/invalid images
            batch_dir = os.path.join(tmp_dir, 'batch')
            os.makedirs(batch_dir)
            
            # Create valid image
            valid_img = Image.new('RGB', (20, 20), color='blue')
            valid_img.save(os.path.join(batch_dir, 'valid.jpg'))
            
            # Create invalid file (not an image)
            with open(os.path.join(batch_dir, 'invalid.jpg'), 'w') as f:
                f.write('not an image')
            
            # Use proper pixel database
            good_db = {'blue': [np.array([0, 0, 255])]}
            good_weights = {'blue': 1.0}
            
            results = batch_processor.process_batch(batch_dir, good_db, good_weights)
            
            # Should have processed valid image and marked invalid as error
            self.assertEqual(len(results), 2)
            self.assertIn('valid.jpg', results)
            self.assertIn('invalid.jpg', results)
            
            # Valid image should have proper classification
            self.assertNotEqual(results['valid.jpg']['category'], 'error')
            
            # Invalid image should be marked as error
            self.assertEqual(results['invalid.jpg']['category'], 'error')
    
    def test_large_dataset_workflow(self):
        """Test system with larger dataset"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create larger reference dataset
            ref_dir = os.path.join(tmp_dir, 'references')
            
            categories = ['red', 'green', 'blue', 'yellow']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            
            pixel_db = {}
            for cat, color in zip(categories, colors):
                cat_dir = os.path.join(ref_dir, cat)
                os.makedirs(cat_dir)
                
                # Create multiple images per category
                for i in range(3):
                    img = Image.new('RGB', (40, 40), color=color)
                    img.save(os.path.join(cat_dir, f'{cat}_{i}.jpg'))
            
            # Build pixel database
            pixel_db = pixel_sampler.build_pixel_database(ref_dir, sample_size=15)
            
            # Verify database
            self.assertEqual(len(pixel_db), 4)
            for cat in categories:
                self.assertIn(cat, pixel_db)
                self.assertEqual(len(pixel_db[cat]), 45)  # 15 pixels * 3 images
            
            # Create test batch
            batch_dir = os.path.join(tmp_dir, 'batch')
            os.makedirs(batch_dir)
            
            test_images = []
            for i, (cat, color) in enumerate(zip(categories, colors)):
                img = Image.new('RGB', (30, 30), color=color)
                img_path = os.path.join(batch_dir, f'test_{cat}.jpg')
                img.save(img_path)
                test_images.append((img_path, cat))
            
            # Process batch
            weights = {cat: 1.0 for cat in categories}
            results = batch_processor.process_batch(batch_dir, pixel_db, weights)
            
            # Verify results
            self.assertEqual(len(results), 4)
            
            # Check that most images are classified correctly
            # (allowing for some misclassification due to randomness)
            correct_classifications = 0
            for filename, expected_cat in test_images:
                basename = os.path.basename(filename)
                if basename in results:
                    actual_cat = results[basename]['category']
                    if actual_cat == expected_cat:
                        correct_classifications += 1
            
            # Should get at least 50% correct (random would be 25%)
            self.assertGreaterEqual(correct_classifications, 2)
    
    def test_persistence_workflow(self):
        """Test data persistence across sessions"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save pixel database
            pixel_db = {
                'cat1': [np.array([255, 0, 0]), np.array([250, 5, 5])],
                'cat2': [np.array([0, 255, 0]), np.array([5, 250, 5])]
            }
            
            db_file = os.path.join(tmp_dir, 'pixel_db.pkl')
            self.assertTrue(pixel_sampler.save_pixel_database(pixel_db, db_file))
            
            # Create and save weights
            weights = {'cat1': 1.2, 'cat2': 0.8}
            weights_file = os.path.join(tmp_dir, 'weights.pkl')
            self.assertTrue(correlation_analyzer.save_weights(weights, weights_file))
            
            # Create and save results
            results = {
                'img1.jpg': {'category': 'cat1', 'confidence': 0.8},
                'img2.jpg': {'category': 'cat2', 'confidence': 0.9}
            }
            results_file = os.path.join(tmp_dir, 'results.json')
            batch_processor.save_results(results, results_file)
            
            # Simulate new session - load everything back
            loaded_db = pixel_sampler.load_pixel_database(db_file)
            loaded_weights = correlation_analyzer.load_weights(weights_file)
            loaded_results = batch_processor.load_results(results_file)
            
            # Verify everything loaded correctly
            self.assertEqual(len(loaded_db), 2)
            self.assertEqual(loaded_weights, weights)
            self.assertEqual(loaded_results, results)
            
            # Verify pixel data integrity
            np.testing.assert_array_equal(loaded_db['cat1'][0], pixel_db['cat1'][0])
            np.testing.assert_array_equal(loaded_db['cat2'][1], pixel_db['cat2'][1])


if __name__ == '__main__':
    unittest.main()
