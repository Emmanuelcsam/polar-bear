#!/usr/bin/env python3
"""
Unit tests for ml_dataset_builder module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime


class TestMLDatasetBuilder(unittest.TestCase):
    """Test cases for ML dataset builder functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.num_samples = 1000
        self.num_features = 20
        
        # Create sample dataset
        self.sample_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(self.num_samples) 
            for i in range(self.num_features)
        })
        self.sample_data['target'] = np.random.randint(0, 3, self.num_samples)
        self.sample_data['timestamp'] = pd.date_range(
            start='2024-01-01', 
            periods=self.num_samples, 
            freq='H'
        )
        
        self.builder_config = {
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_seed': 42,
            'stratify': True
        }
        
    def tearDown(self):
        """Clean up after each test method."""
        self.sample_data = None
        self.builder_config = None
        
    def test_module_imports(self):
        """Test that ml_dataset_builder module can be imported."""
        try:
            import ml_dataset_builder
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import ml_dataset_builder: {e}")
            
    def test_data_validation(self):
        """Test data validation functionality."""
        # Check data shape
        self.assertEqual(self.sample_data.shape, (self.num_samples, self.num_features + 2))
        
        # Check for missing values
        self.assertFalse(self.sample_data.isnull().any().any())
        
        # Check data types
        numeric_cols = [f'feature_{i}' for i in range(self.num_features)]
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data[col]))
            
    def test_split_ratios(self):
        """Test dataset split ratios."""
        total_ratio = (self.builder_config['train_split'] + 
                      self.builder_config['val_split'] + 
                      self.builder_config['test_split'])
        
        self.assertAlmostEqual(total_ratio, 1.0, places=5)
        self.assertGreater(self.builder_config['train_split'], 0)
        self.assertGreater(self.builder_config['val_split'], 0)
        self.assertGreater(self.builder_config['test_split'], 0)
        
    def test_mock_dataset_building(self):
        """Test dataset building with mocks."""
        mock_builder = Mock()
        mock_builder.build_dataset = MagicMock(return_value={
            'train': {'X': np.random.randn(700, 20), 'y': np.random.randint(0, 3, 700)},
            'val': {'X': np.random.randn(150, 20), 'y': np.random.randint(0, 3, 150)},
            'test': {'X': np.random.randn(150, 20), 'y': np.random.randint(0, 3, 150)},
            'metadata': {
                'num_features': 20,
                'num_classes': 3,
                'feature_names': [f'feature_{i}' for i in range(20)]
            }
        })
        
        result = mock_builder.build_dataset(self.sample_data, self.builder_config)
        
        # Verify structure
        self.assertIn('train', result)
        self.assertIn('val', result)
        self.assertIn('test', result)
        self.assertIn('metadata', result)
        
        # Verify shapes
        self.assertEqual(result['train']['X'].shape[1], self.num_features)
        self.assertEqual(result['val']['X'].shape[1], self.num_features)
        self.assertEqual(result['test']['X'].shape[1], self.num_features)
        
        # Verify splits sum to total
        total_samples = (result['train']['X'].shape[0] + 
                        result['val']['X'].shape[0] + 
                        result['test']['X'].shape[0])
        self.assertEqual(total_samples, self.num_samples)
        
    def test_feature_engineering(self):
        """Test feature engineering capabilities."""
        # Add some derived features
        self.sample_data['feature_sum'] = self.sample_data[[f'feature_{i}' for i in range(5)]].sum(axis=1)
        self.sample_data['feature_mean'] = self.sample_data[[f'feature_{i}' for i in range(5)]].mean(axis=1)
        
        self.assertEqual(self.sample_data.shape[1], self.num_features + 4)
        self.assertTrue('feature_sum' in self.sample_data.columns)
        self.assertTrue('feature_mean' in self.sample_data.columns)


if __name__ == '__main__':
    unittest.main()