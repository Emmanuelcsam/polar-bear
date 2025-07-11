#!/usr/bin/env python3
"""
Unit tests for pytorch_module.py
"""
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import pytorch_module
import data_store

class TestPyTorchModule(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()

    @patch('pytorch_module.TORCH_AVAILABLE', False)
    def test_torch_learn_not_available(self):
        """Test torch_learn when PyTorch is not available."""
        result = pytorch_module.torch_learn()

        # Should return None
        self.assertIsNone(result)

    def test_torch_learn_no_data(self):
        """Test torch_learn with no data."""
        # Skip if torch not available
        if not pytorch_module.TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        result = pytorch_module.torch_learn()

        # Should return None
        self.assertIsNone(result)

    @patch('pytorch_module.TORCH_AVAILABLE', True)
    def test_torch_learn_with_mock_torch(self):
        """Test torch_learn with mocked PyTorch."""
        # Save some test data
        data_store.save_event({"intensity": 100})
        data_store.save_event({"pixel": 200})
        data_store.save_event({"intensity": 150})

        # Mock torch functionality
        with patch('pytorch_module.torch') as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.tensor.return_value = mock_tensor
            mock_tensor.unsqueeze.return_value = mock_tensor

            mock_layer = MagicMock()
            mock_torch.nn.Linear.return_value = mock_layer

            mock_output = MagicMock()
            mock_layer.return_value = mock_output
            mock_output.__len__ = MagicMock(return_value=5)
            mock_output.__getitem__ = MagicMock(return_value=mock_output)
            mock_output.detach.return_value = mock_output
            mock_output.numpy.return_value = np.array([1.0, 2.0, 3.0])

            result = pytorch_module.torch_learn()

            # Check that torch functions were called
            mock_torch.tensor.assert_called_once_with([100, 200, 150], dtype=mock_torch.float32)
            mock_torch.nn.Linear.assert_called_once_with(1, 1)

            # Check result
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_torch_learn_with_real_torch(self):
        """Test torch_learn with real PyTorch if available."""
        # Skip if torch not available
        if not pytorch_module.TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Save some test data
        data_store.save_event({"intensity": 100})
        data_store.save_event({"pixel": 200})
        data_store.save_event({"intensity": 150})

        result = pytorch_module.torch_learn()

        # Should return array with 3 elements (all our data)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result, np.ndarray)

    def test_torch_learn_mixed_events(self):
        """Test torch_learn with mixed event types."""
        # Skip if torch not available
        if not pytorch_module.TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Save mixed events
        data_store.save_event({"intensity": 100})
        data_store.save_event({"pixel": 200})
        data_store.save_event({"other": "data"})
        data_store.save_event({"intensity": 150})

        result = pytorch_module.torch_learn()

        # Should process 3 values (intensity and pixel, not other)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

    @patch('pytorch_module.print')
    def test_torch_learn_prints_no_data(self, mock_print):
        """Test torch_learn prints message when no data."""
        pytorch_module.torch_learn()

        if pytorch_module.TORCH_AVAILABLE:
            mock_print.assert_called_with("[Torch] No data")
        else:
            mock_print.assert_called_with("[Torch] PyTorch not available, skipping")

if __name__ == "__main__":
    unittest.main()
