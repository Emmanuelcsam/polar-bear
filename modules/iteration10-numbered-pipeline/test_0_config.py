# test_0_config.py
import os
import sys
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_imports():
    """Test that the config module can be imported successfully."""
    try:
        from importlib import import_module
        config = import_module('0_config')
        assert config is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import config module: {e}")

def test_config_constants():
    """Test that all required configuration constants are defined."""
    from importlib import import_module
    config = import_module('0_config')

    # Test that all required constants exist
    required_constants = [
        'LEARNING_MODE', 'ANALYSIS_DURATION_SECONDS', 'INPUT_DIR',
        'DATA_DIR', 'OUTPUT_DIR', 'INTENSITY_DATA_PATH', 'STATS_DATA_PATH',
        'MODEL_PATH', 'AVG_IMAGE_PATH', 'START_TIME', 'END_TIME'
    ]

    for constant in required_constants:
        assert hasattr(config, constant), f"Missing constant: {constant}"

def test_config_learning_mode():
    """Test that LEARNING_MODE is set to a valid value."""
    from importlib import import_module
    config = import_module('0_config')

    assert config.LEARNING_MODE in ['auto', 'manual'], \
        f"Invalid LEARNING_MODE: {config.LEARNING_MODE}"

def test_config_directories_created():
    """Test that required directories are created when config is imported."""
    from importlib import import_module
    config = import_module('0_config')

    # Check that DATA_DIR and OUTPUT_DIR exist
    assert os.path.exists(config.DATA_DIR), f"DATA_DIR not created: {config.DATA_DIR}"
    assert os.path.exists(config.OUTPUT_DIR), f"OUTPUT_DIR not created: {config.OUTPUT_DIR}"

def test_config_file_paths():
    """Test that all file paths are properly constructed."""
    from importlib import import_module
    config = import_module('0_config')

    # Test that file paths are strings and contain the correct directory
    assert isinstance(config.INTENSITY_DATA_PATH, str)
    assert config.DATA_DIR in config.INTENSITY_DATA_PATH
    assert config.INTENSITY_DATA_PATH.endswith('.npy')

    assert isinstance(config.STATS_DATA_PATH, str)
    assert config.DATA_DIR in config.STATS_DATA_PATH
    assert config.STATS_DATA_PATH.endswith('.csv')

    assert isinstance(config.MODEL_PATH, str)
    assert config.DATA_DIR in config.MODEL_PATH
    assert config.MODEL_PATH.endswith('.pth')

def test_config_timing():
    """Test that timing configuration is properly set."""
    from importlib import import_module
    config = import_module('0_config')

    # Test that timing values are numbers
    assert isinstance(config.ANALYSIS_DURATION_SECONDS, (int, float))
    assert config.ANALYSIS_DURATION_SECONDS > 0

    assert isinstance(config.START_TIME, (int, float))
    assert isinstance(config.END_TIME, (int, float))
    assert config.END_TIME > config.START_TIME

def test_config_directory_names():
    """Test that directory names are correctly set."""
    from importlib import import_module
    config = import_module('0_config')

    assert config.INPUT_DIR == 'images_input'
    assert config.DATA_DIR == 'data'
    assert config.OUTPUT_DIR == 'output'

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_config_imports,
        test_config_constants,
        test_config_learning_mode,
        test_config_directories_created,
        test_config_file_paths,
        test_config_timing,
        test_config_directory_names
    ]

    print("Running config tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("Config tests completed.")
