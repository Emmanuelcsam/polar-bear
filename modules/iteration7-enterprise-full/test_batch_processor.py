import unittest
import os
import json
import tempfile
import shutil
from PIL import Image
import numpy as np
from batch_processor import process_batch

class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test images
        self.test_dir = tempfile.mkdtemp()
        self.batch_results_file = 'batch_results.json'
        
        # Create test images
        self.create_test_images()
    
    def tearDown(self):
        # Clean up test directory
        shutil.rmtree(self.test_dir)
        
        # Clean up output files
        if os.path.exists(self.batch_results_file):
            os.remove(self.batch_results_file)
        
        # Clean up individual batch files
        for file in os.listdir('.'):
            if file.startswith('batch_') and file.endswith('.json'):
                os.remove(file)
    
    def create_test_images(self):
        """Create various test images"""
        # Simple grayscale image
        img1 = Image.new('L', (10, 10))
        img1.putdata([128] * 100)
        img1.save(os.path.join(self.test_dir, 'test1.png'))
        
        # Gradient image
        img2 = Image.new('L', (20, 20))
        gradient = list(range(0, 400))
        img2.putdata(gradient)
        img2.save(os.path.join(self.test_dir, 'test2.jpg'))
        
        # Random noise image
        img3 = Image.new('L', (15, 15))
        np.random.seed(42)
        noise = list(np.random.randint(0, 256, 225))
        img3.putdata(noise)
        img3.save(os.path.join(self.test_dir, 'test3.bmp'))
        
        # Color image (will be converted to grayscale)
        img4 = Image.new('RGB', (8, 8))
        color_data = [(i, i*2, i*3) for i in range(64)]
        img4.putdata(color_data)
        img4.save(os.path.join(self.test_dir, 'test4.jpeg'))
        
        # Non-image file (should be ignored)
        with open(os.path.join(self.test_dir, 'not_an_image.txt'), 'w') as f:
            f.write('This is not an image')
    
    def test_process_batch_basic(self):
        # Test basic batch processing
        results = process_batch(self.test_dir)
        
        # Check that results were returned
        self.assertEqual(len(results), 4)  # 4 image files
        
        # Check that batch results file was created
        self.assertTrue(os.path.exists(self.batch_results_file))
        
        # Verify batch results content
        with open(self.batch_results_file, 'r') as f:
            saved_results = json.load(f)
        
        self.assertEqual(len(saved_results), 4)
    
    def test_process_batch_statistics(self):
        # Test that statistics are calculated correctly
        results = process_batch(self.test_dir)
        
        # Find the uniform image (test1.png)
        test1_result = next(r for r in results if r['filename'] == 'test1.png')
        
        # Check statistics for uniform image
        self.assertAlmostEqual(test1_result['mean'], 128.0, places=1)
        self.assertAlmostEqual(test1_result['std'], 0.0, places=1)
        self.assertEqual(test1_result['min'], 128)
        self.assertEqual(test1_result['max'], 128)
        self.assertEqual(test1_result['unique_values'], 1)
        self.assertEqual(tuple(test1_result['size']), (10, 10))
    
    def test_individual_files_created(self):
        # Test that individual batch files are created
        # Clean up any existing batch files first
        for f in os.listdir('.'):
            if f.startswith('batch_') and f.endswith('.json'):
                os.remove(f)
                
        process_batch(self.test_dir)
        
        # Check for individual files - should match our test image names
        expected_files = ['batch_test1.png.json', 'batch_test2.jpg.json', 
                         'batch_test3.bmp.json', 'batch_test4.jpeg.json']
        
        for expected_file in expected_files:
            self.assertTrue(os.path.exists(expected_file), f"Missing {expected_file}")
        
        # Verify content of one file
        with open('batch_test1.png.json', 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['filename'], 'test1.png')
        self.assertIn('pixels', data)
        self.assertIn('stats', data)
        self.assertLessEqual(len(data['pixels']), 1000)  # Should be limited to 1000
    
    def test_process_batch_empty_directory(self):
        # Test with empty directory
        empty_dir = tempfile.mkdtemp()
        
        try:
            results = process_batch(empty_dir)
            
            self.assertEqual(len(results), 0)
            
            # Batch results file should still be created
            self.assertTrue(os.path.exists(self.batch_results_file))
            
            with open(self.batch_results_file, 'r') as f:
                saved_results = json.load(f)
            self.assertEqual(len(saved_results), 0)
            
        finally:
            os.rmdir(empty_dir)
    
    def test_process_batch_large_image(self):
        # Test with image larger than 1000 pixels
        large_img = Image.new('L', (50, 50))  # 2500 pixels
        # Create data that fits exactly
        pattern = list(range(256))
        full_data = (pattern * 10)[:2500]  # Take exactly 2500 pixels
        large_img.putdata(full_data)
        large_img.save(os.path.join(self.test_dir, 'large.png'))
        
        process_batch(self.test_dir)
        
        # Check that pixel data is limited
        with open('batch_large.png.json', 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data['pixels']), 1000)  # Should be limited
        self.assertEqual(tuple(data['stats']['size']), (50, 50))
    
    def test_process_batch_invalid_image(self):
        # Create a corrupted image file
        corrupt_file = os.path.join(self.test_dir, 'corrupt.png')
        with open(corrupt_file, 'wb') as f:
            f.write(b'This is not a valid PNG file')
        
        # Should handle the error gracefully
        results = process_batch(self.test_dir)
        
        # Should process the valid images
        self.assertEqual(len(results), 4)  # Only the 4 valid images
    
    def test_color_to_grayscale_conversion(self):
        # Test that color images are converted to grayscale
        results = process_batch(self.test_dir)
        
        # Find the color image result
        test4_result = next(r for r in results if r['filename'] == 'test4.jpeg')
        
        # Check that statistics are for grayscale values
        self.assertTrue(0 <= test4_result['min'] <= 255)
        self.assertTrue(0 <= test4_result['max'] <= 255)
        self.assertTrue(0 <= test4_result['mean'] <= 255)
    
    def test_file_extensions(self):
        # Test that only image extensions are processed
        # Already created a .txt file in setUp
        results = process_batch(self.test_dir)
        
        # Check that text file was ignored
        filenames = [r['filename'] for r in results]
        self.assertNotIn('not_an_image.txt', filenames)

if __name__ == '__main__':
    unittest.main()