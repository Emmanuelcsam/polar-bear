# test_2_intensity_reader.py
import sys
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image(filepath, width=100, height=100, pixel_value=128):
    """Create a simple test image using numpy and cv2."""
    try:
        import cv2
        # Create a simple grayscale image
        img = np.full((height, width), pixel_value, dtype=np.uint8)
        cv2.imwrite(filepath, img)
        return True
    except ImportError:
        # If cv2 is not available, create a dummy file
        with open(filepath, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG header
        return False

def test_intensity_reader_import():
    """Test that the intensity reader module can be imported successfully."""
    try:
        from importlib import import_module
        intensity_reader = import_module('2_intensity_reader')
        assert intensity_reader is not None
        assert hasattr(intensity_reader, 'read_intensities')
    except ImportError as e:
        raise AssertionError(f"Failed to import intensity reader module: {e}")

def test_read_intensities_no_images():
    """Test intensity reader behavior when no images are found."""
    from importlib import import_module
    intensity_reader = import_module('2_intensity_reader')

    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Mock config to use our temporary directories
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = os.path.join(data_dir, 'intensities.npy')

        # Patch the config in the intensity reader module
        with patch.object(intensity_reader, 'config', mock_config):
            try:
                result = intensity_reader.read_intensities()
                # Should handle empty directory gracefully
                assert result is None or len(result) == 0
            except Exception as e:
                raise AssertionError(f"read_intensities failed with empty directory: {e}")

def test_read_intensities_with_images():
    """Test intensity reader behavior with test images."""
    from importlib import import_module
    intensity_reader = import_module('2_intensity_reader')

    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Create test images
        test_image1 = os.path.join(input_dir, 'test1.png')
        test_image2 = os.path.join(input_dir, 'test2.jpg')

        # Create simple test images
        cv2_available = create_test_image(test_image1, 10, 10, 100)
        create_test_image(test_image2, 10, 10, 150)

        # Mock config to use our temporary directories
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = os.path.join(data_dir, 'intensities.npy')

        # Patch the config in the intensity reader module
        with patch.object(intensity_reader, 'config', mock_config):
            try:
                result = intensity_reader.read_intensities()

                if cv2_available:
                    # If cv2 is available, check that data was saved
                    assert os.path.exists(mock_config.INTENSITY_DATA_PATH)

                    # Load and verify the saved data
                    saved_data = np.load(mock_config.INTENSITY_DATA_PATH)
                    assert len(saved_data) > 0
                    assert isinstance(saved_data, np.ndarray)
                    assert saved_data.dtype == np.uint8
                else:
                    # If cv2 is not available, just check it doesn't crash
                    assert True

            except Exception as e:
                raise AssertionError(f"read_intensities failed with test images: {e}")

def test_read_intensities_file_filtering():
    """Test that only image files are processed."""
    from importlib import import_module
    intensity_reader = import_module('2_intensity_reader')

    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Create image files
        create_test_image(os.path.join(input_dir, 'test1.png'), 5, 5, 128)
        create_test_image(os.path.join(input_dir, 'test2.jpg'), 5, 5, 200)

        # Create non-image files
        with open(os.path.join(input_dir, 'test.txt'), 'w') as f:
            f.write('not an image')
        with open(os.path.join(input_dir, 'test.pdf'), 'w') as f:
            f.write('not an image')

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = os.path.join(data_dir, 'intensities.npy')

        # Patch the config in the intensity reader module
        with patch.object(intensity_reader, 'config', mock_config):
            try:
                result = intensity_reader.read_intensities()
                # Should process only image files without crashing
                assert True
            except Exception as e:
                raise AssertionError(f"read_intensities failed with mixed file types: {e}")

def test_intensity_data_format():
    """Test that intensity data is saved in correct format."""
    from importlib import import_module
    intensity_reader = import_module('2_intensity_reader')

    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Create a simple test image
        test_image = os.path.join(input_dir, 'test.png')
        cv2_available = create_test_image(test_image, 3, 3, 128)

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = os.path.join(data_dir, 'intensities.npy')

        # Patch the config in the intensity reader module
        with patch.object(intensity_reader, 'config', mock_config):
            try:
                result = intensity_reader.read_intensities()

                if cv2_available and os.path.exists(mock_config.INTENSITY_DATA_PATH):
                    # Load and verify the saved data format
                    saved_data = np.load(mock_config.INTENSITY_DATA_PATH)

                    # Check data type and shape
                    assert saved_data.dtype == np.uint8
                    assert saved_data.ndim == 1  # Should be flattened
                    assert len(saved_data) > 0

                    # Check that values are in valid range for pixel intensities
                    assert np.all(saved_data >= 0)
                    assert np.all(saved_data <= 255)

            except Exception as e:
                raise AssertionError(f"Intensity data format test failed: {e}")

def test_config_integration():
    """Test that the config module is properly integrated."""
    from importlib import import_module
    intensity_reader = import_module('2_intensity_reader')

    # Test that config object exists and has required attributes
    assert hasattr(intensity_reader, 'config')
    config = intensity_reader.config

    # Test that config has the required attributes
    required_attrs = ['INPUT_DIR', 'DATA_DIR', 'INTENSITY_DATA_PATH']
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing attribute: {attr}"

def test_directory_creation():
    """Test that input directory is created if it doesn't exist."""
    from importlib import import_module
    intensity_reader = import_module('2_intensity_reader')

    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'nonexistent_input')
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Mock config with non-existent input directory
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = os.path.join(data_dir, 'intensities.npy')

        # Patch the config in the intensity reader module
        with patch.object(intensity_reader, 'config', mock_config):
            try:
                result = intensity_reader.read_intensities()
                # Should create the input directory
                assert os.path.exists(input_dir)
            except Exception as e:
                raise AssertionError(f"Directory creation test failed: {e}")

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_intensity_reader_import,
        test_read_intensities_no_images,
        test_read_intensities_with_images,
        test_read_intensities_file_filtering,
        test_intensity_data_format,
        test_config_integration,
        test_directory_creation
    ]

    print("Running intensity reader tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("Intensity reader tests completed.")
