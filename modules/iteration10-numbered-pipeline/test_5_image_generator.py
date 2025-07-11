# test_5_image_generator.py
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_model(filepath):
    """Create a test model file."""
    try:
        import torch
        # Create a simple uniform distribution
        distribution = torch.ones(256) / 256.0
        torch.save(distribution, filepath)
        return True
    except ImportError:
        # Create a dummy file if torch is not available
        with open(filepath, 'wb') as f:
            f.write(b'dummy_model_data')
        return False

def test_image_generator_import():
    """Test that the image generator module can be imported successfully."""
    try:
        from importlib import import_module
        image_generator = import_module('5_image_generator')
        assert image_generator is not None
        assert hasattr(image_generator, 'generate_image')
    except ImportError as e:
        raise AssertionError(f"Failed to import image generator module: {e}")

def test_generate_image_no_model():
    """Test image generator behavior when no model file exists."""
    from importlib import import_module
    image_generator = import_module('5_image_generator')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Mock config with non-existent model file
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.OUTPUT_DIR = output_dir
        mock_config.MODEL_PATH = os.path.join(data_dir, 'nonexistent.pth')

        # Patch the config in the image generator module
        with patch.object(image_generator, 'config', mock_config):
            try:
                result = image_generator.generate_image()
                # Should handle missing model file gracefully
                assert result is None
            except Exception as e:
                raise AssertionError(f"generate_image failed with no model file: {e}")

def test_generate_image_no_torch():
    """Test image generator behavior when PyTorch is not available."""
    from importlib import import_module
    image_generator = import_module('5_image_generator')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.OUTPUT_DIR = output_dir
        mock_config.MODEL_PATH = os.path.join(data_dir, 'model.pth')

        # Patch torch to be None
        with patch.object(image_generator, 'torch', None):
            with patch.object(image_generator, 'config', mock_config):
                try:
                    result = image_generator.generate_image()
                    # Should handle missing torch gracefully
                    assert result is None
                except Exception as e:
                    raise AssertionError(f"generate_image failed without torch: {e}")

def test_generate_image_with_model():
    """Test image generator behavior with valid model."""
    from importlib import import_module
    image_generator = import_module('5_image_generator')

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("PyTorch not available, skipping torch-specific tests")
        return

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create test model file
        model_file = os.path.join(data_dir, 'model.pth')
        torch_available = create_test_model(model_file)

        if not torch_available:
            print("Could not create test model, skipping test")
            return

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.OUTPUT_DIR = output_dir
        mock_config.MODEL_PATH = model_file

        # Patch the config in the image generator module
        with patch.object(image_generator, 'config', mock_config):
            try:
                result = image_generator.generate_image(width=10, height=10)

                # Should return a numpy array
                assert result is not None
                assert isinstance(result, np.ndarray)
                assert result.shape == (10, 10)
                assert result.dtype == np.uint8

                # Check that output image was saved
                output_path = os.path.join(output_dir, "generated_image.png")
                assert os.path.exists(output_path), "Generated image was not saved"

            except Exception as e:
                raise AssertionError(f"generate_image failed with valid model: {e}")

def test_generate_image_dimensions():
    """Test that generated images have correct dimensions."""
    from importlib import import_module
    image_generator = import_module('5_image_generator')

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("PyTorch not available, skipping torch-specific tests")
        return

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create test model file
        model_file = os.path.join(data_dir, 'model.pth')
        torch_available = create_test_model(model_file)

        if not torch_available:
            print("Could not create test model, skipping test")
            return

        # Test different dimensions
        test_dimensions = [(5, 5), (10, 20), (50, 30)]

        for width, height in test_dimensions:
            # Mock config
            mock_config = MagicMock()
            mock_config.DATA_DIR = data_dir
            mock_config.OUTPUT_DIR = output_dir
            mock_config.MODEL_PATH = model_file

            # Patch the config in the image generator module
            with patch.object(image_generator, 'config', mock_config):
                try:
                    result = image_generator.generate_image(width=width, height=height)

                    # Check dimensions
                    assert result.shape == (height, width), f"Wrong dimensions for {width}x{height}: {result.shape}"

                except Exception as e:
                    raise AssertionError(f"Dimension test failed for {width}x{height}: {e}")

def test_generate_image_pixel_values():
    """Test that generated images have valid pixel values."""
    from importlib import import_module
    image_generator = import_module('5_image_generator')

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("PyTorch not available, skipping torch-specific tests")
        return

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create test model file
        model_file = os.path.join(data_dir, 'model.pth')
        torch_available = create_test_model(model_file)

        if not torch_available:
            print("Could not create test model, skipping test")
            return

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.OUTPUT_DIR = output_dir
        mock_config.MODEL_PATH = model_file

        # Patch the config in the image generator module
        with patch.object(image_generator, 'config', mock_config):
            try:
                result = image_generator.generate_image(width=20, height=20)

                # Check pixel value range
                assert np.all(result >= 0), "Generated image has negative pixel values"
                assert np.all(result <= 255), "Generated image has pixel values > 255"

                # Check data type
                assert result.dtype == np.uint8, f"Wrong data type: {result.dtype}"

            except Exception as e:
                raise AssertionError(f"Pixel value test failed: {e}")

def test_default_dimensions():
    """Test that default dimensions are used when not specified."""
    from importlib import import_module
    image_generator = import_module('5_image_generator')

    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("PyTorch not available, skipping torch-specific tests")
        return

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create test model file
        model_file = os.path.join(data_dir, 'model.pth')
        torch_available = create_test_model(model_file)

        if not torch_available:
            print("Could not create test model, skipping test")
            return

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.OUTPUT_DIR = output_dir
        mock_config.MODEL_PATH = model_file

        # Patch the config in the image generator module
        with patch.object(image_generator, 'config', mock_config):
            try:
                result = image_generator.generate_image()  # No dimensions specified

                # Should use default dimensions (256x256)
                assert result.shape == (256, 256), f"Wrong default dimensions: {result.shape}"

            except Exception as e:
                raise AssertionError(f"Default dimensions test failed: {e}")

def test_config_integration():
    """Test that the config module is properly integrated."""
    from importlib import import_module
    image_generator = import_module('5_image_generator')

    # Test that config object exists and has required attributes
    assert hasattr(image_generator, 'config')
    config = image_generator.config

    # Test that config has the required attributes
    required_attrs = ['DATA_DIR', 'OUTPUT_DIR', 'MODEL_PATH']
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing attribute: {attr}"

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_image_generator_import,
        test_generate_image_no_model,
        test_generate_image_no_torch,
        test_generate_image_with_model,
        test_generate_image_dimensions,
        test_generate_image_pixel_values,
        test_default_dimensions,
        test_config_integration
    ]

    print("Running image generator tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("Image generator tests completed.")
