import unittest
import os
import json
import tempfile
from PIL import Image
from pixel_reader import read_pixels

class TestPixelReader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.test_dir, 'test_image.png')
        self.output_path = 'pixel_data.json'
        
        # Create a simple test image (3x3 grayscale)
        img = Image.new('L', (3, 3))
        pixels = [0, 128, 255, 64, 192, 32, 96, 160, 224]
        img.putdata(pixels)
        img.save(self.test_image_path)
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.rmdir(self.test_dir)
    
    def test_read_pixels_basic(self):
        # Test basic pixel reading functionality
        pixels = read_pixels(self.test_image_path)
        
        # Check if pixels were read correctly
        expected_pixels = [0, 128, 255, 64, 192, 32, 96, 160, 224]
        self.assertEqual(pixels, expected_pixels)
        
        # Check if output file was created
        self.assertTrue(os.path.exists(self.output_path))
        
        # Verify JSON output
        with open(self.output_path, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['image'], self.test_image_path)
        self.assertEqual(data['pixels'], expected_pixels)
        self.assertEqual(data['size'], [3, 3])
        self.assertIn('timestamp', data)
    
    def test_read_pixels_color_to_grayscale(self):
        # Test conversion from color to grayscale
        color_image_path = os.path.join(self.test_dir, 'color_image.png')
        img = Image.new('RGB', (2, 2))
        img.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)])
        img.save(color_image_path)
        
        pixels = read_pixels(color_image_path)
        
        # Check that we got grayscale values
        self.assertEqual(len(pixels), 4)
        for pixel in pixels:
            self.assertIsInstance(pixel, int)
            self.assertTrue(0 <= pixel <= 255)
        
        os.remove(color_image_path)
    
    def test_read_pixels_large_image(self):
        # Test with a larger image
        large_image_path = os.path.join(self.test_dir, 'large_image.png')
        img = Image.new('L', (100, 100))
        img.save(large_image_path)
        
        pixels = read_pixels(large_image_path)
        
        # Check correct number of pixels
        self.assertEqual(len(pixels), 10000)
        
        os.remove(large_image_path)
    
    def test_invalid_image_path(self):
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            read_pixels('non_existent_image.png')

if __name__ == '__main__':
    unittest.main()