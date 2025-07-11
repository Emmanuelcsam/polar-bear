#!/usr/bin/env python3
"""
Unit tests for hpc_module.py
"""
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import hpc_module

class TestHPCModule(unittest.TestCase):

    def test_cpu_multiply(self):
        """Test CPU multiplication."""
        data = [1, 2, 3, 4]
        result = hpc_module.cpu_multiply(data, multiplier=3)

        expected = np.array([3, 6, 9, 12])
        np.testing.assert_array_equal(result, expected)

    def test_cpu_multiply_default_multiplier(self):
        """Test CPU multiplication with default multiplier."""
        data = [1, 2, 3]
        result = hpc_module.cpu_multiply(data)

        expected = np.array([2, 4, 6])
        np.testing.assert_array_equal(result, expected)

    @patch('hpc_module.print')
    def test_cpu_multiply_prints_result(self, mock_print):
        """Test that CPU multiply prints result."""
        data = [1, 2, 3]
        hpc_module.cpu_multiply(data)

        mock_print.assert_called_with("[HPC] CPU Result:", np.array([2, 4, 6]))

    def test_gpu_multiply_fallback_to_cpu(self):
        """Test GPU multiply falls back to CPU when CuPy not available."""
        # Test when CuPy is not available
        original_cupy_available = hpc_module.CUPY_AVAILABLE
        hpc_module.CUPY_AVAILABLE = False

        try:
            data = [1, 2, 3]
            result = hpc_module.gpu_multiply(data)

            expected = np.array([2, 4, 6])
            np.testing.assert_array_equal(result, expected)
        finally:
            hpc_module.CUPY_AVAILABLE = original_cupy_available

    @patch('hpc_module.CUPY_AVAILABLE', True)
    def test_gpu_multiply_with_cupy_mock(self):
        """Test GPU multiply with mocked CuPy."""
        # Mock CuPy functionality
        with patch('hpc_module.cp') as mock_cp:
            mock_array = MagicMock()
            mock_cp.array.return_value = mock_array
            mock_array.__mul__ = MagicMock(return_value=mock_array)
            mock_cp.asnumpy.return_value = np.array([2, 4, 6])

            data = [1, 2, 3]
            result = hpc_module.gpu_multiply(data)

            # Check that CuPy functions were called
            mock_cp.array.assert_called_once_with(data)
            mock_cp.asnumpy.assert_called_once_with(mock_array)

            # Check result
            expected = np.array([2, 4, 6])
            np.testing.assert_array_equal(result, expected)

    @patch('hpc_module.CUPY_AVAILABLE', True)
    @patch('hpc_module.cpu_multiply')
    def test_gpu_multiply_exception_fallback(self, mock_cpu_multiply):
        """Test GPU multiply falls back to CPU on exception."""
        # Mock CuPy to raise an exception
        with patch('hpc_module.cp') as mock_cp:
            mock_cp.array.side_effect = Exception("GPU error")

            data = [1, 2, 3]
            hpc_module.gpu_multiply(data)

            # Check that CPU multiply was called as fallback
            mock_cpu_multiply.assert_called_once_with(data, 2)

    def test_gpu_multiply_custom_multiplier(self):
        """Test GPU multiply with custom multiplier."""
        # Test with CuPy not available to use CPU path
        original_cupy_available = hpc_module.CUPY_AVAILABLE
        hpc_module.CUPY_AVAILABLE = False

        try:
            data = [1, 2, 3]
            result = hpc_module.gpu_multiply(data, multiplier=5)

            expected = np.array([5, 10, 15])
            np.testing.assert_array_equal(result, expected)
        finally:
            hpc_module.CUPY_AVAILABLE = original_cupy_available

if __name__ == "__main__":
    unittest.main()
