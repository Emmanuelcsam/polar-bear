import unittest
import json
import os
from pattern_recognizer import recognize_patterns

class TestPatternRecognizer(unittest.TestCase):
    def setUp(self):
        self.pixel_data_file = 'pixel_data.json'
        self.patterns_file = 'patterns.json'
    
    def tearDown(self):
        # Clean up files created during tests
        for file in [self.pixel_data_file, self.patterns_file]:
            if os.path.exists(file):
                os.remove(file)
    
    def create_test_data(self, pixels):
        """Helper method to create test pixel data"""
        data = {
            'timestamp': 1234567890,
            'image': 'test.png',
            'pixels': pixels,
            'size': [len(pixels), 1]
        }
        with open(self.pixel_data_file, 'w') as f:
            json.dump(data, f)
    
    def test_recognize_patterns_basic(self):
        # Create test data with known patterns
        pixels = [100, 100, 100, 100, 50, 60, 70, 80, 200, 190, 180, 170]
        self.create_test_data(pixels)
        
        recognize_patterns()
        
        # Check if patterns file was created
        self.assertTrue(os.path.exists(self.patterns_file))
        
        # Load and verify patterns
        with open(self.patterns_file, 'r') as f:
            patterns = json.load(f)
        
        # Check structure
        self.assertIn('frequency', patterns)
        self.assertIn('sequences', patterns)
        self.assertIn('statistics', patterns)
    
    def test_frequency_patterns(self):
        # Test frequency counting
        pixels = [10, 20, 10, 30, 10, 20, 40, 10, 20, 10]
        self.create_test_data(pixels)
        
        recognize_patterns()
        
        with open(self.patterns_file, 'r') as f:
            patterns = json.load(f)
        
        freq = patterns['frequency']
        self.assertEqual(freq['10'], 5)
        self.assertEqual(freq['20'], 3)
        self.assertEqual(freq['30'], 1)
        self.assertEqual(freq['40'], 1)
    
    def test_sequential_patterns(self):
        # Test detection of sequential patterns
        pixels = [
            100, 100, 100, 100,  # Repeat pattern
            10, 20, 30, 40,      # Ascending pattern
            90, 80, 70, 60,      # Descending pattern
            50, 55, 45, 60       # No pattern
        ]
        self.create_test_data(pixels)
        
        recognize_patterns()
        
        with open(self.patterns_file, 'r') as f:
            patterns = json.load(f)
        
        sequences = patterns['sequences']
        
        # Check for repeat pattern
        repeat_found = any(seq[0] == 'repeat' and seq[1] == 100 for seq in sequences)
        self.assertTrue(repeat_found)
        
        # Check for ascending pattern
        ascending_found = any(seq[0] == 'ascending' and seq[1] == [10, 20, 30, 40] for seq in sequences)
        self.assertTrue(ascending_found)
        
        # Check for descending pattern
        descending_found = any(seq[0] == 'descending' and seq[1] == [90, 80, 70, 60] for seq in sequences)
        self.assertTrue(descending_found)
    
    def test_statistical_patterns(self):
        # Test statistical calculations
        pixels = [0, 50, 100, 150, 200, 250]
        self.create_test_data(pixels)
        
        recognize_patterns()
        
        with open(self.patterns_file, 'r') as f:
            patterns = json.load(f)
        
        stats = patterns['statistics'][0]
        self.assertAlmostEqual(stats['mean'], 125.0, places=1)
        self.assertEqual(stats['min'], 0)
        self.assertEqual(stats['max'], 250)
        self.assertGreater(stats['std'], 0)
    
    def test_sequence_limit(self):
        # Test that sequences are limited to 100
        # Create data with many patterns
        pixels = []
        for i in range(150):
            pixels.extend([i, i, i, i])  # Many repeat patterns
        
        self.create_test_data(pixels)
        recognize_patterns()
        
        with open(self.patterns_file, 'r') as f:
            patterns = json.load(f)
        
        # Check that sequences are limited
        self.assertLessEqual(len(patterns['sequences']), 100)
    
    def test_empty_data(self):
        # Test with empty pixel data
        self.create_test_data([])
        
        recognize_patterns()
        
        with open(self.patterns_file, 'r') as f:
            patterns = json.load(f)
        
        # Should handle empty data gracefully
        self.assertEqual(len(patterns['frequency']), 0)
        self.assertEqual(len(patterns['sequences']), 0)
    
    def test_missing_file(self):
        # Test behavior when pixel_data.json doesn't exist
        if os.path.exists(self.pixel_data_file):
            os.remove(self.pixel_data_file)
        
        # Should handle the exception gracefully
        recognize_patterns()
        
        # Patterns file should not be created
        self.assertFalse(os.path.exists(self.patterns_file))
    
    def test_complex_patterns(self):
        # Test with more complex data
        import numpy as np
        np.random.seed(42)
        
        # Create data with various patterns
        repeat = [128] * 10
        ascending = list(range(0, 100, 10))
        descending = list(range(255, 155, -10))
        random_data = [int(x) for x in np.random.randint(0, 256, 50)]
        
        pixels = repeat + ascending + descending + random_data
        self.create_test_data(pixels)
        
        recognize_patterns()
        
        with open(self.patterns_file, 'r') as f:
            patterns = json.load(f)
        
        # Verify all components are present and reasonable
        self.assertGreater(len(patterns['frequency']), 0)
        self.assertGreater(len(patterns['sequences']), 0)
        self.assertEqual(len(patterns['statistics']), 1)
        
        # Check statistics are within expected range
        stats = patterns['statistics'][0]
        self.assertTrue(0 <= stats['mean'] <= 255)
        self.assertTrue(0 <= stats['min'] <= 255)
        self.assertTrue(0 <= stats['max'] <= 255)

if __name__ == '__main__':
    unittest.main()