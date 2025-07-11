import unittest
import os
import json
import tempfile
from data_store import DataStore

class TestDataStore(unittest.TestCase):
    def setUp(self):
        # Save original data file path and use a temporary one
        self.original_data_file = 'learned_data.json'
        self.temp_data_file = tempfile.mktemp(suffix='.json')
        
        # Create a test instance
        self.store = DataStore()
        self.store.data_file = self.temp_data_file
    
    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.temp_data_file):
            os.remove(self.temp_data_file)
        if os.path.exists(self.original_data_file):
            os.remove(self.original_data_file)
    
    def test_initialization(self):
        # Test that a new DataStore initializes correctly
        store = DataStore()
        self.assertIn('pixel_frequencies', store.data)
        self.assertIn('patterns', store.data)
        self.assertIn('anomalies', store.data)
        self.assertIn('image_profiles', store.data)
        
        # Check default values
        self.assertEqual(len(store.data['patterns']), 0)
        self.assertEqual(len(store.data['anomalies']), 0)
        self.assertEqual(len(store.data['image_profiles']), 0)
    
    def test_update_frequency(self):
        # Test frequency updating
        self.store.update_frequency(128)
        self.assertEqual(self.store.data['pixel_frequencies']['128'], 1)
        
        self.store.update_frequency(128)
        self.assertEqual(self.store.data['pixel_frequencies']['128'], 2)
        
        self.store.update_frequency(255)
        self.assertEqual(self.store.data['pixel_frequencies']['255'], 1)
    
    def test_add_pattern(self):
        # Test pattern adding
        pattern = {'type': 'gradient', 'values': [0, 128, 255]}
        self.store.add_pattern(pattern)
        
        self.assertEqual(len(self.store.data['patterns']), 1)
        self.assertEqual(self.store.data['patterns'][0], pattern)
    
    def test_pattern_limit(self):
        # Test that patterns are limited to 100
        for i in range(150):
            self.store.add_pattern({'id': i})
        
        self.assertEqual(len(self.store.data['patterns']), 100)
        # Check that we kept the last 100 patterns
        self.assertEqual(self.store.data['patterns'][0]['id'], 50)
        self.assertEqual(self.store.data['patterns'][-1]['id'], 149)
    
    def test_add_anomaly(self):
        # Test anomaly adding
        anomaly = {'type': 'spike', 'value': 255, 'position': 100}
        self.store.add_anomaly(anomaly)
        
        self.assertEqual(len(self.store.data['anomalies']), 1)
        self.assertEqual(self.store.data['anomalies'][0], anomaly)
    
    def test_anomaly_limit(self):
        # Test that anomalies are limited to 100
        for i in range(150):
            self.store.add_anomaly({'id': i})
        
        self.assertEqual(len(self.store.data['anomalies']), 100)
        # Check that we kept the last 100 anomalies
        self.assertEqual(self.store.data['anomalies'][0]['id'], 50)
        self.assertEqual(self.store.data['anomalies'][-1]['id'], 149)
    
    def test_save_and_load(self):
        # Test saving and loading data
        self.store.update_frequency(64)
        self.store.add_pattern({'type': 'test'})
        self.store.add_anomaly({'anomaly': 'test'})
        
        # Create a new store instance to test loading
        new_store = DataStore()
        new_store.data_file = self.temp_data_file
        new_store.load()
        
        # Check that data was loaded correctly
        self.assertEqual(new_store.data['pixel_frequencies']['64'], 1)
        self.assertEqual(len(new_store.data['patterns']), 1)
        self.assertEqual(len(new_store.data['anomalies']), 1)
        self.assertEqual(new_store.data['patterns'][0]['type'], 'test')
        self.assertEqual(new_store.data['anomalies'][0]['anomaly'], 'test')
    
    def test_defaultdict_behavior(self):
        # Test that defaultdict works correctly after loading
        self.store.update_frequency(100)
        self.store.save()
        
        # Load in a new instance
        new_store = DataStore()
        new_store.data_file = self.temp_data_file
        new_store.load()
        
        # Test that we can still update frequencies (defaultdict should work)
        new_store.update_frequency(200)
        self.assertEqual(new_store.data['pixel_frequencies']['100'], 1)
        self.assertEqual(new_store.data['pixel_frequencies']['200'], 1)

if __name__ == '__main__':
    unittest.main()