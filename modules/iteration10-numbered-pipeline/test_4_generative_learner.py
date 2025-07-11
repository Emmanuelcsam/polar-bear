# test_4_generative_learner.py
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_intensity_data(filepath, data_values=None):
    """Create test intensity data file."""
    if data_values is None:
        # Create sample intensity data with known distribution
        data_values = np.array([100, 100, 100, 150, 150, 200, 200, 200, 200, 255], dtype=np.uint8)

    np.save(filepath, data_values)
    return data_values

def test_generative_learner_import():
    """Test that the generative learner module can be imported successfully."""
    try:
        from importlib import import_module
        generative_learner = import_module('4_generative_learner')
        assert generative_learner is not None
        assert hasattr(generative_learner, 'learn_pixel_distribution')
    except ImportError as e:
        raise AssertionError(f"Failed to import generative learner module: {e}")

def test_learn_pixel_distribution_no_data():
    """Test generative learner behavior when no data file exists."""
    from importlib import import_module
    generative_learner = import_module('4_generative_learner')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Mock config with non-existent data file
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = os.path.join(data_dir, 'nonexistent.npy')
        mock_config.MODEL_PATH = os.path.join(data_dir, 'model.pth')

        # Patch the config in the generative learner module
        with patch.object(generative_learner, 'config', mock_config):
            try:
                result = generative_learner.learn_pixel_distribution()
                # Should handle missing file gracefully
                assert result is None
            except Exception as e:
                raise AssertionError(f"learn_pixel_distribution failed with no data file: {e}")

def test_learn_pixel_distribution_empty_data():
    """Test generative learner behavior with empty data file."""
    from importlib import import_module
    generative_learner = import_module('4_generative_learner')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Create empty data file
        intensity_file = os.path.join(data_dir, 'intensities.npy')
        empty_data = np.array([], dtype=np.uint8)
        np.save(intensity_file, empty_data)

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = intensity_file
        mock_config.MODEL_PATH = os.path.join(data_dir, 'model.pth')

        # Patch the config in the generative learner module
        with patch.object(generative_learner, 'config', mock_config):
            try:
                result = generative_learner.learn_pixel_distribution()
                # Should handle empty data gracefully
                assert result is None
            except Exception as e:
                raise AssertionError(f"learn_pixel_distribution failed with empty data: {e}")

def test_learn_pixel_distribution_with_data():
    """Test generative learner behavior with valid data."""
    from importlib import import_module
    generative_learner = import_module('4_generative_learner')

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
        os.makedirs(data_dir, exist_ok=True)

        # Create test data file
        intensity_file = os.path.join(data_dir, 'intensities.npy')
        model_file = os.path.join(data_dir, 'model.pth')
        test_data = create_test_intensity_data(intensity_file)

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = intensity_file
        mock_config.MODEL_PATH = model_file

        # Patch the config in the generative learner module
        with patch.object(generative_learner, 'config', mock_config):
            try:
                result = generative_learner.learn_pixel_distribution()

                # Should return a tensor with distribution
                assert result is not None
                assert torch.is_tensor(result)

                # Check that model file was saved
                assert os.path.exists(model_file)

                # Check distribution properties
                assert result.shape[0] == 256  # Should have 256 values (0-255)
                assert torch.sum(result) > 0  # Should have some probability mass

            except Exception as e:
                raise AssertionError(f"learn_pixel_distribution failed with valid data: {e}")

def test_distribution_properties():
    """Test that the learned distribution has correct properties."""
    from importlib import import_module
    generative_learner = import_module('4_generative_learner')

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
        os.makedirs(data_dir, exist_ok=True)

        # Create test data with known distribution
        intensity_file = os.path.join(data_dir, 'intensities.npy')
        model_file = os.path.join(data_dir, 'model.pth')

        # Create data with specific values
        test_data = np.array([100] * 50 + [200] * 30 + [255] * 20, dtype=np.uint8)
        np.save(intensity_file, test_data)

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = intensity_file
        mock_config.MODEL_PATH = model_file

        # Patch the config in the generative learner module
        with patch.object(generative_learner, 'config', mock_config):
            try:
                result = generative_learner.learn_pixel_distribution()

                # Check that distribution sums to 1 (probability distribution)
                assert abs(torch.sum(result).item() - 1.0) < 0.001, f"Distribution doesn't sum to 1: {torch.sum(result)}"

                # Check that non-zero probabilities are at the right places
                assert result[100] > 0, "Should have probability at pixel value 100"
                assert result[200] > 0, "Should have probability at pixel value 200"
                assert result[255] > 0, "Should have probability at pixel value 255"

                # Check that highest probability is at most frequent value (100)
                max_prob_index = torch.argmax(result).item()
                assert max_prob_index == 100, f"Highest probability should be at value 100, got {max_prob_index}"

            except Exception as e:
                raise AssertionError(f"Distribution properties test failed: {e}")

def test_model_saving_and_loading():
    """Test that the model can be saved and loaded correctly."""
    from importlib import import_module
    generative_learner = import_module('4_generative_learner')

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
        os.makedirs(data_dir, exist_ok=True)

        # Create test data file
        intensity_file = os.path.join(data_dir, 'intensities.npy')
        model_file = os.path.join(data_dir, 'model.pth')
        test_data = create_test_intensity_data(intensity_file)

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = intensity_file
        mock_config.MODEL_PATH = model_file

        # Patch the config in the generative learner module
        with patch.object(generative_learner, 'config', mock_config):
            try:
                # Learn the distribution
                result = generative_learner.learn_pixel_distribution()

                # Check that model file exists
                assert os.path.exists(model_file), "Model file was not saved"

                # Try to load the saved model
                loaded_model = torch.load(model_file)

                # Check that loaded model matches the returned result
                assert torch.allclose(result, loaded_model), "Loaded model doesn't match saved model"

            except Exception as e:
                raise AssertionError(f"Model saving/loading test failed: {e}")

def test_different_data_sizes():
    """Test that the learner works with different data sizes."""
    from importlib import import_module
    generative_learner = import_module('4_generative_learner')

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
        os.makedirs(data_dir, exist_ok=True)

        # Test with different data sizes
        for size in [10, 100, 1000]:
            intensity_file = os.path.join(data_dir, f'intensities_{size}.npy')
            model_file = os.path.join(data_dir, f'model_{size}.pth')

            # Create test data
            np.random.seed(42)
            test_data = np.random.randint(0, 256, size=size, dtype=np.uint8)
            np.save(intensity_file, test_data)

            # Mock config
            mock_config = MagicMock()
            mock_config.DATA_DIR = data_dir
            mock_config.INTENSITY_DATA_PATH = intensity_file
            mock_config.MODEL_PATH = model_file

            # Patch the config in the generative learner module
            with patch.object(generative_learner, 'config', mock_config):
                try:
                    result = generative_learner.learn_pixel_distribution()

                    # Check that distribution is valid regardless of data size
                    assert result is not None
                    assert torch.is_tensor(result)
                    assert result.shape[0] == 256
                    assert abs(torch.sum(result).item() - 1.0) < 0.001

                except Exception as e:
                    raise AssertionError(f"Different data sizes test failed for size {size}: {e}")

def test_config_integration():
    """Test that the config module is properly integrated."""
    from importlib import import_module
    generative_learner = import_module('4_generative_learner')

    # Test that config object exists and has required attributes
    assert hasattr(generative_learner, 'config')
    config = generative_learner.config

    # Test that config has the required attributes
    required_attrs = ['DATA_DIR', 'INTENSITY_DATA_PATH', 'MODEL_PATH']
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing attribute: {attr}"

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_generative_learner_import,
        test_learn_pixel_distribution_no_data,
        test_learn_pixel_distribution_empty_data,
        test_learn_pixel_distribution_with_data,
        test_distribution_properties,
        test_model_saving_and_loading,
        test_different_data_sizes,
        test_config_integration
    ]

    print("Running generative learner tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("Generative learner tests completed.")
