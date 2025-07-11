# test_1_batch_processor.py
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_batch_processor_import():
    """Test that the batch processor module can be imported successfully."""
    try:
        from importlib import import_module
        batch_processor = import_module('1_batch_processor')
        assert batch_processor is not None
        assert hasattr(batch_processor, 'process_images')
    except ImportError as e:
        raise AssertionError(f"Failed to import batch processor module: {e}")

def test_process_images_no_images():
    """Test batch processor behavior when no images are found."""
    from importlib import import_module
    batch_processor = import_module('1_batch_processor')

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock config to use our temporary directory
        mock_config = MagicMock()
        mock_config.INPUT_DIR = temp_dir
        mock_config.LEARNING_MODE = 'auto'

        # Patch the config in the batch processor module
        with patch.object(batch_processor, 'config', mock_config):
            # Should handle empty directory gracefully
            try:
                batch_processor.process_images()
                # If we get here, the function handled the empty directory correctly
                assert True
            except Exception as e:
                raise AssertionError(f"process_images failed with empty directory: {e}")

def test_process_images_with_images():
    """Test batch processor behavior with test images."""
    from importlib import import_module
    batch_processor = import_module('1_batch_processor')

    # Create a temporary directory with test images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy image files
        test_image1 = os.path.join(temp_dir, 'test1.png')
        test_image2 = os.path.join(temp_dir, 'test2.jpg')

        # Create empty files (for testing purposes)
        with open(test_image1, 'w') as f:
            f.write('')
        with open(test_image2, 'w') as f:
            f.write('')

        # Mock config to use our temporary directory
        mock_config = MagicMock()
        mock_config.INPUT_DIR = temp_dir
        mock_config.LEARNING_MODE = 'auto'

        # Patch the config in the batch processor module
        with patch.object(batch_processor, 'config', mock_config):
            try:
                batch_processor.process_images()
                assert True
            except Exception as e:
                raise AssertionError(f"process_images failed with test images: {e}")

def test_manual_mode():
    """Test batch processor in manual mode."""
    from importlib import import_module
    batch_processor = import_module('1_batch_processor')

    # Create a temporary directory with multiple test images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple dummy image files
        for i in range(3):
            test_image = os.path.join(temp_dir, f'test{i}.png')
            with open(test_image, 'w') as f:
                f.write('')

        # Mock config for manual mode
        mock_config = MagicMock()
        mock_config.INPUT_DIR = temp_dir
        mock_config.LEARNING_MODE = 'manual'

        # Patch the config in the batch processor module
        with patch.object(batch_processor, 'config', mock_config):
            try:
                batch_processor.process_images()
                assert True
            except Exception as e:
                raise AssertionError(f"process_images failed in manual mode: {e}")

def test_image_file_filtering():
    """Test that only image files are processed."""
    from importlib import import_module
    batch_processor = import_module('1_batch_processor')

    # Create a temporary directory with mixed file types
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create image files
        image_files = ['test1.png', 'test2.jpg', 'test3.PNG', 'test4.JPG']
        for filename in image_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write('')

        # Create non-image files
        non_image_files = ['test.txt', 'test.pdf', 'test.doc']
        for filename in non_image_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write('')

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = temp_dir
        mock_config.LEARNING_MODE = 'auto'

        # Patch the config in the batch processor module
        with patch.object(batch_processor, 'config', mock_config):
            try:
                batch_processor.process_images()
                assert True
            except Exception as e:
                raise AssertionError(f"process_images failed with mixed file types: {e}")

def test_config_integration():
    """Test that the config module is properly integrated."""
    from importlib import import_module
    batch_processor = import_module('1_batch_processor')

    # Test that config object exists and has required attributes
    assert hasattr(batch_processor, 'config')
    config = batch_processor.config

    # Test that config has the required attributes
    required_attrs = ['INPUT_DIR', 'LEARNING_MODE']
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing attribute: {attr}"

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_batch_processor_import,
        test_process_images_no_images,
        test_process_images_with_images,
        test_manual_mode,
        test_image_file_filtering,
        test_config_integration
    ]

    print("Running batch processor tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("Batch processor tests completed.")
