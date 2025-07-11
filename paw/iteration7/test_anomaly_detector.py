import unittest
import json
import os
import numpy as np
from anomaly_detector import detect_anomalies

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        self.pixel_data_file = 'pixel_data.json'
        self.anomalies_file = 'anomalies.json'
    
    def tearDown(self):
        # Clean up files created during tests
        for file in [self.pixel_data_file, self.anomalies_file]:
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
    
    def test_detect_anomalies_normal_distribution(self):
        # Create data with normal distribution and some outliers
        np.random.seed(42)
        normal_data = [int(x) for x in np.random.normal(128, 20, 100)]
        # Add clear outliers
        outliers = [0, 255, 255, 0]
        pixels = normal_data + outliers
        
        self.create_test_data(pixels)
        detect_anomalies()
        
        # Check if anomalies file was created
        self.assertTrue(os.path.exists(self.anomalies_file))
        
        # Load and check anomalies
        with open(self.anomalies_file, 'r') as f:
            anomalies = json.load(f)
        
        # Check structure
        self.assertIn('z_score_anomalies', anomalies)
        self.assertIn('iqr_anomalies', anomalies)
        self.assertIn('bounds', anomalies)
        
        # Check that some anomalies were detected
        self.assertGreater(len(anomalies['z_score_anomalies']), 0)
        self.assertGreater(len(anomalies['iqr_anomalies']), 0)
        
        # Check bounds structure
        bounds = anomalies['bounds']
        self.assertIn('lower', bounds)
        self.assertIn('upper', bounds)
        self.assertIn('mean', bounds)
        self.assertIn('std', bounds)
    
    def test_detect_anomalies_uniform_data(self):
        # Test with uniform data (no anomalies expected)
        pixels = [128] * 100
        
        self.create_test_data(pixels)
        detect_anomalies()
        
        with open(self.anomalies_file, 'r') as f:
            anomalies = json.load(f)
        
        # With uniform data, z-score method should find no anomalies
        self.assertEqual(len(anomalies['z_score_anomalies']), 0)
    
    def test_detect_anomalies_extreme_outliers(self):
        # Test with extreme outliers
        pixels = [128] * 50 + [0, 255] * 5 + [128] * 50
        
        self.create_test_data(pixels)
        detect_anomalies()
        
        with open(self.anomalies_file, 'r') as f:
            anomalies = json.load(f)
        
        # Check that extreme values are detected
        z_anomaly_values = [a['value'] for a in anomalies['z_score_anomalies']]
        self.assertIn(0, z_anomaly_values)
        self.assertIn(255, z_anomaly_values)
    
    def test_detect_anomalies_limit(self):
        # Test that anomalies are limited to 50
        # Create data with many outliers
        pixels = [128] * 50 + list(range(0, 255, 3))
        
        self.create_test_data(pixels)
        detect_anomalies()
        
        with open(self.anomalies_file, 'r') as f:
            anomalies = json.load(f)
        
        # Check that results are limited
        self.assertLessEqual(len(anomalies['z_score_anomalies']), 50)
        self.assertLessEqual(len(anomalies['iqr_anomalies']), 50)
    
    def test_detect_anomalies_missing_file(self):
        # Test behavior when pixel_data.json doesn't exist
        if os.path.exists(self.pixel_data_file):
            os.remove(self.pixel_data_file)
        
        # Should handle the exception gracefully
        detect_anomalies()
        
        # Anomalies file should not be created
        self.assertFalse(os.path.exists(self.anomalies_file))
    
    def test_anomaly_detection_accuracy(self):
        # Test that known anomalies are detected correctly
        # Create bimodal distribution with clear outliers
        np.random.seed(42)
        group1 = [int(x) for x in np.random.normal(50, 5, 40)]
        group2 = [int(x) for x in np.random.normal(200, 5, 40)]
        outliers = [0, 1, 254, 255]
        pixels = group1 + group2 + outliers
        
        self.create_test_data(pixels)
        detect_anomalies()
        
        with open(self.anomalies_file, 'r') as f:
            anomalies = json.load(f)
        
        # Extract indices of detected anomalies
        z_indices = [a['index'] for a in anomalies['z_score_anomalies']]
        
        # Check if our known outliers are in the detected anomalies
        outlier_indices = list(range(80, 84))  # Where we placed outliers
        detected_outliers = [i for i in outlier_indices if i in z_indices]
        
        # At least some of our outliers should be detected
        self.assertGreater(len(detected_outliers), 0)

if __name__ == '__main__':
    unittest.main()