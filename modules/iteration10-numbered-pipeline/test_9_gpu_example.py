# test_9_gpu_example.py
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gpu_example_import():
    """Test that the GPU example module can be imported successfully."""
    try:
        from importlib import import_module
        gpu_example = import_module('9_gpu_example')
        assert gpu_example is not None
        assert hasattr(gpu_example, 'check_gpu_capabilities')
    except ImportError as e:
        raise AssertionError(f"Failed to import GPU example module: {e}")

def test_check_gpu_capabilities_no_torch():
    """Test GPU capability check when PyTorch is not available."""
    from importlib import import_module
    gpu_example = import_module('9_gpu_example')

    # Patch torch to be None
    with patch.object(gpu_example, 'torch', None):
        try:
            result = gpu_example.check_gpu_capabilities()
            # Should handle missing torch gracefully
            assert result is False
        except Exception as e:
            raise AssertionError(f"check_gpu_capabilities failed without torch: {e}")

def test_check_gpu_capabilities_no_cuda():
    """Test GPU capability check when CUDA is not available."""
    from importlib import import_module
    gpu_example = import_module('9_gpu_example')

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("PyTorch not available, skipping CUDA-specific tests")
        return

    # Mock torch.cuda.is_available to return False
    with patch.object(gpu_example, 'torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = False

        try:
            result = gpu_example.check_gpu_capabilities()
            # Should handle no CUDA gracefully
            assert result is False
        except Exception as e:
            raise AssertionError(f"check_gpu_capabilities failed without CUDA: {e}")

def test_check_gpu_capabilities_with_cuda():
    """Test GPU capability check when CUDA is available."""
    from importlib import import_module
    gpu_example = import_module('9_gpu_example')

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("PyTorch not available, skipping CUDA simulation test")
        return

    # Mock torch.cuda.is_available to return True
    with patch.object(gpu_example, 'torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Mock GPU"
        mock_torch.device.return_value = "cuda:0"

        # Create mock tensor
        mock_tensor = MagicMock()
        mock_tensor.device = "cuda:0"
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor

        mock_torch.randn.return_value = mock_tensor
        mock_torch.matmul.return_value = mock_tensor

        try:
            result = gpu_example.check_gpu_capabilities()
            # Should complete GPU operations successfully
            assert result is True

            # Verify that GPU operations were called
            mock_torch.cuda.is_available.assert_called()
            mock_torch.cuda.get_device_name.assert_called()
            mock_torch.randn.assert_called()
            mock_torch.matmul.assert_called()

        except Exception as e:
            raise AssertionError(f"check_gpu_capabilities failed with mocked CUDA: {e}")

def test_check_gpu_capabilities_tensor_operations():
    """Test that tensor operations are performed correctly."""
    from importlib import import_module
    gpu_example = import_module('9_gpu_example')

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("PyTorch not available, skipping tensor operations test")
        return

    # Mock torch with detailed tensor operations
    with patch.object(gpu_example, 'torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"

        # Create mock device
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        # Create mock tensors
        mock_cpu_tensor = MagicMock()
        mock_gpu_tensor = MagicMock()
        mock_result_tensor = MagicMock()

        mock_cpu_tensor.to.return_value = mock_gpu_tensor
        mock_gpu_tensor.device = "cuda:0"
        mock_result_tensor.cpu.return_value = mock_cpu_tensor

        mock_torch.randn.return_value = mock_cpu_tensor
        mock_torch.matmul.return_value = mock_result_tensor

        try:
            result = gpu_example.check_gpu_capabilities()
            assert result is True

            # Verify tensor operations were called in correct order
            mock_torch.randn.assert_called_with(5, 5)
            mock_cpu_tensor.to.assert_called_with(mock_device)
            mock_torch.matmul.assert_called_with(mock_gpu_tensor, mock_gpu_tensor)
            mock_result_tensor.cpu.assert_called()

        except Exception as e:
            raise AssertionError(f"Tensor operations test failed: {e}")

def test_check_gpu_capabilities_error_handling():
    """Test GPU capability check error handling."""
    from importlib import import_module
    gpu_example = import_module('9_gpu_example')

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("PyTorch not available, skipping error handling test")
        return

    # Mock torch.cuda.is_available to raise an exception
    with patch.object(gpu_example, 'torch') as mock_torch:
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")

        try:
            result = gpu_example.check_gpu_capabilities()
            # Should handle errors gracefully
            assert result is False
        except Exception as e:
            # Should not raise unhandled exceptions
            raise AssertionError(f"Unhandled exception in GPU capability check: {e}")

def test_check_gpu_capabilities_real_pytorch():
    """Test GPU capability check with real PyTorch if available."""
    from importlib import import_module
    gpu_example = import_module('9_gpu_example')

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("PyTorch not available, skipping real PyTorch test")
        return

    try:
        # Run the actual function (will use CPU if no GPU)
        result = gpu_example.check_gpu_capabilities()
        # Should complete without error regardless of GPU availability
        assert isinstance(result, bool)
    except Exception as e:
        raise AssertionError(f"Real PyTorch test failed: {e}")

def test_config_integration():
    """Test that the config module is properly integrated."""
    from importlib import import_module
    gpu_example = import_module('9_gpu_example')

    # Test that config object exists
    assert hasattr(gpu_example, 'config')
    config = gpu_example.config

    # Config should be accessible even if not used directly in this module
    assert config is not None

def test_torch_import_handling():
    """Test that torch import is handled correctly."""
    from importlib import import_module
    gpu_example = import_module('9_gpu_example')

    # Check that torch attribute exists (might be None)
    assert hasattr(gpu_example, 'torch')

    # If torch is None, check_gpu_capabilities should handle it
    if gpu_example.torch is None:
        result = gpu_example.check_gpu_capabilities()
        assert result is False

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_gpu_example_import,
        test_check_gpu_capabilities_no_torch,
        test_check_gpu_capabilities_no_cuda,
        test_check_gpu_capabilities_with_cuda,
        test_check_gpu_capabilities_tensor_operations,
        test_check_gpu_capabilities_error_handling,
        test_check_gpu_capabilities_real_pytorch,
        test_config_integration,
        test_torch_import_handling
    ]

    print("Running GPU example tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("GPU example tests completed.")
