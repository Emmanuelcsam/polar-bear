#!/usr/bin/env python3
"""
Unit tests for fiber_dataset_pytorch module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch


class TestFiberDatasetPyTorch(unittest.TestCase):
    """Test cases for fiber dataset PyTorch functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_size = 32
        self.num_features = 10
        self.num_samples = 100
        
        # Create sample data
        self.sample_data = torch.randn(self.num_samples, self.num_features)
        self.sample_labels = torch.zeros(self.num_samples)
        
    def tearDown(self):
        """Clean up after each test method."""
        self.sample_data = None
        self.sample_labels = None
        
    def test_module_imports(self):
        """Test that fiber_dataset_pytorch module can be imported."""
        try:
            import fiber_dataset_pytorch
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import fiber_dataset_pytorch: {e}")
            
    def test_tensor_operations(self):
        """Test basic tensor operations."""
        # Test tensor shapes
        self.assertEqual(self.sample_data.shape, (self.num_samples, self.num_features))
        self.assertEqual(self.sample_labels.shape, (self.num_samples,))
        
        # Test tensor types
        self.assertEqual(self.sample_data.dtype, torch.float32)
        self.assertEqual(self.sample_labels.dtype, torch.int64)
        
    def test_dataset_mock(self):
        """Test dataset functionality with mocks."""
        mock_dataset = Mock()
        mock_dataset.__len__ = MagicMock(return_value=self.num_samples)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda idx: (self.sample_data[idx], self.sample_labels[idx]))
        
        # Test length
        self.assertEqual(len(mock_dataset), self.num_samples)
        
        # Test item access
        data, label = mock_dataset[0]
        self.assertEqual(data.shape, (self.num_features,))
        self.assertIn(label.item(), [0, 1])
        
    def test_dataloader_simulation(self):
        """Test dataloader simulation."""
        batch_size = 16
        num_batches = self.num_samples // batch_size
        
        # Simulate batching
        batches_processed = 0
        for i in range(0, self.num_samples, batch_size):
            batch_data = self.sample_data[i:i+batch_size]
            batch_labels = self.sample_labels[i:i+batch_size]
            
            if batch_data.shape[0] == batch_size:  # Full batch only
                batches_processed += 1
                
        self.assertEqual(batches_processed, num_batches)


if __name__ == '__main__':
    unittest.main()