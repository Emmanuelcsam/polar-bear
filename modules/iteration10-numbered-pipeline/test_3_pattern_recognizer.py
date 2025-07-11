# test_3_pattern_recognizer.py
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
        # Create sample intensity data with known statistical properties
        data_values = np.array([100, 100, 100, 150, 150, 200, 200, 200, 200, 255], dtype=np.uint8)

    np.save(filepath, data_values)
    return data_values

def test_pattern_recognizer_import():
    """Test that the pattern recognizer module can be imported successfully."""
    try:
        from importlib import import_module
        pattern_recognizer = import_module('3_pattern_recognizer')
        assert pattern_recognizer is not None
        assert hasattr(pattern_recognizer, 'recognize_patterns')
    except ImportError as e:
        raise AssertionError(f"Failed to import pattern recognizer module: {e}")

def test_recognize_patterns_no_data():
    """Test pattern recognizer behavior when no data file exists."""
    from importlib import import_module
    pattern_recognizer = import_module('3_pattern_recognizer')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Mock config with non-existent data file
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = os.path.join(data_dir, 'nonexistent.npy')

        # Patch the config in the pattern recognizer module
        with patch.object(pattern_recognizer, 'config', mock_config):
            try:
                result = pattern_recognizer.recognize_patterns()
                # Should handle missing file gracefully
                assert result is None
            except Exception as e:
                raise AssertionError(f"recognize_patterns failed with no data file: {e}")

def test_recognize_patterns_empty_data():
    """Test pattern recognizer behavior with empty data file."""
    from importlib import import_module
    pattern_recognizer = import_module('3_pattern_recognizer')

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

        # Patch the config in the pattern recognizer module
        with patch.object(pattern_recognizer, 'config', mock_config):
            try:
                result = pattern_recognizer.recognize_patterns()
                # Should handle empty data gracefully
                assert result is None
            except Exception as e:
                raise AssertionError(f"recognize_patterns failed with empty data: {e}")

def test_recognize_patterns_with_data():
    """Test pattern recognizer behavior with valid data."""
    from importlib import import_module
    pattern_recognizer = import_module('3_pattern_recognizer')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Create test data file with known statistical properties
        intensity_file = os.path.join(data_dir, 'intensities.npy')
        test_data = create_test_intensity_data(intensity_file)

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = intensity_file

        # Patch the config in the pattern recognizer module
        with patch.object(pattern_recognizer, 'config', mock_config):
            try:
                result = pattern_recognizer.recognize_patterns()

                # Should return a dictionary with statistics
                assert result is not None
                assert isinstance(result, dict)

                # Check that all expected statistics are present
                expected_keys = ['mean', 'median', 'std_dev', 'mode']
                for key in expected_keys:
                    assert key in result, f"Missing statistic: {key}"
                    assert isinstance(result[key], (int, float, np.number))

            except Exception as e:
                raise AssertionError(f"recognize_patterns failed with valid data: {e}")

def test_statistical_calculations():
    """Test that statistical calculations are correct."""
    from importlib import import_module
    pattern_recognizer = import_module('3_pattern_recognizer')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Create test data with known statistical properties
        intensity_file = os.path.join(data_dir, 'intensities.npy')
        test_data = np.array([100, 100, 100, 150, 200, 200, 200, 200], dtype=np.uint8)
        np.save(intensity_file, test_data)

        # Calculate expected statistics
        expected_mean = np.mean(test_data)
        expected_median = np.median(test_data)
        expected_std = np.std(test_data)
        expected_mode = np.bincount(test_data).argmax()

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = intensity_file

        # Patch the config in the pattern recognizer module
        with patch.object(pattern_recognizer, 'config', mock_config):
            try:
                result = pattern_recognizer.recognize_patterns()

                # Check that calculated statistics match expected values
                assert abs(result['mean'] - expected_mean) < 0.01, f"Mean mismatch: {result['mean']} vs {expected_mean}"
                assert abs(result['median'] - expected_median) < 0.01, f"Median mismatch: {result['median']} vs {expected_median}"
                assert abs(result['std_dev'] - expected_std) < 0.01, f"Std dev mismatch: {result['std_dev']} vs {expected_std}"
                assert result['mode'] == expected_mode, f"Mode mismatch: {result['mode']} vs {expected_mode}"

            except Exception as e:
                raise AssertionError(f"Statistical calculations test failed: {e}")

def test_data_type_handling():
    """Test that different data types are handled correctly."""
    from importlib import import_module
    pattern_recognizer = import_module('3_pattern_recognizer')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Create test data with different intensity values
        intensity_file = os.path.join(data_dir, 'intensities.npy')
        test_data = np.array([0, 1, 127, 128, 254, 255], dtype=np.uint8)
        np.save(intensity_file, test_data)

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = intensity_file

        # Patch the config in the pattern recognizer module
        with patch.object(pattern_recognizer, 'config', mock_config):
            try:
                result = pattern_recognizer.recognize_patterns()

                # Check that results are within valid ranges
                assert 0 <= result['mean'] <= 255, f"Mean out of range: {result['mean']}"
                assert 0 <= result['median'] <= 255, f"Median out of range: {result['median']}"
                assert result['std_dev'] >= 0, f"Std dev negative: {result['std_dev']}"
                assert 0 <= result['mode'] <= 255, f"Mode out of range: {result['mode']}"

            except Exception as e:
                raise AssertionError(f"Data type handling test failed: {e}")

def test_large_dataset():
    """Test pattern recognizer with a large dataset."""
    from importlib import import_module
    pattern_recognizer = import_module('3_pattern_recognizer')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Create large test dataset
        intensity_file = os.path.join(data_dir, 'intensities.npy')
        np.random.seed(42)  # For reproducible results
        test_data = np.random.randint(0, 256, size=10000, dtype=np.uint8)
        np.save(intensity_file, test_data)

        # Mock config
        mock_config = MagicMock()
        mock_config.DATA_DIR = data_dir
        mock_config.INTENSITY_DATA_PATH = intensity_file

        # Patch the config in the pattern recognizer module
        with patch.object(pattern_recognizer, 'config', mock_config):
            try:
                result = pattern_recognizer.recognize_patterns()

                # Check that results are reasonable for large dataset
                assert result is not None
                assert isinstance(result, dict)
                assert 0 <= result['mean'] <= 255
                assert 0 <= result['median'] <= 255
                assert result['std_dev'] >= 0
                assert 0 <= result['mode'] <= 255

            except Exception as e:
                raise AssertionError(f"Large dataset test failed: {e}")

def test_config_integration():
    """Test that the config module is properly integrated."""
    from importlib import import_module
    pattern_recognizer = import_module('3_pattern_recognizer')

    # Test that config object exists and has required attributes
    assert hasattr(pattern_recognizer, 'config')
    config = pattern_recognizer.config

    # Test that config has the required attributes
    required_attrs = ['DATA_DIR', 'INTENSITY_DATA_PATH']
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing attribute: {attr}"

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_pattern_recognizer_import,
        test_recognize_patterns_no_data,
        test_recognize_patterns_empty_data,
        test_recognize_patterns_with_data,
        test_statistical_calculations,
        test_data_type_handling,
        test_large_dataset,
        test_config_integration
    ]

    print("Running pattern recognizer tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("Pattern recognizer tests completed.")
