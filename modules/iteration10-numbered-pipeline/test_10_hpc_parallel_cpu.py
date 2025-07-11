# test_10_hpc_parallel_cpu.py
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image_cv2(filepath, width=50, height=50, pixel_value=128):
    """Create a simple test image using cv2."""
    try:
        import cv2
        # Create a simple grayscale image
        img = np.full((height, width), pixel_value, dtype=np.uint8)
        cv2.imwrite(filepath, img)
        return True
    except ImportError:
        # Create a dummy file if cv2 is not available
        with open(filepath, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG header
        return False

def test_hpc_parallel_cpu_import():
    """Test that the HPC parallel CPU module can be imported successfully."""
    try:
        from importlib import import_module
        hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')
        assert hpc_parallel_cpu is not None
        assert hasattr(hpc_parallel_cpu, 'run_parallel_processing')
        assert hasattr(hpc_parallel_cpu, 'get_image_mean')
    except ImportError as e:
        raise AssertionError(f"Failed to import HPC parallel CPU module: {e}")

def test_get_image_mean_valid_image():
    """Test get_image_mean function with valid image."""
    from importlib import import_module
    hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')

    # Create temporary test image
    with tempfile.TemporaryDirectory() as temp_dir:
        test_image = os.path.join(temp_dir, 'test.png')
        cv2_available = create_test_image_cv2(test_image, 10, 10, 128)

        try:
            result = hpc_parallel_cpu.get_image_mean(test_image)

            # Should return tuple with filename and mean value
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == 'test.png'

            if cv2_available:
                # Should return valid mean value
                assert result[1] is not None
                assert isinstance(result[1], (int, float, np.number))
                assert 0 <= result[1] <= 255
            else:
                # If cv2 not available, mean should be None
                assert result[1] is None

        except Exception as e:
            raise AssertionError(f"get_image_mean failed with valid image: {e}")

def test_get_image_mean_invalid_image():
    """Test get_image_mean function with invalid image."""
    from importlib import import_module
    hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')

    # Create temporary corrupted image
    with tempfile.TemporaryDirectory() as temp_dir:
        test_image = os.path.join(temp_dir, 'corrupted.png')
        with open(test_image, 'wb') as f:
            f.write(b'not_a_real_image')

        try:
            result = hpc_parallel_cpu.get_image_mean(test_image)

            # Should handle corrupted image gracefully
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == 'corrupted.png'
            assert result[1] is None

        except Exception as e:
            raise AssertionError(f"get_image_mean failed with invalid image: {e}")

def test_get_image_mean_nonexistent_file():
    """Test get_image_mean function with nonexistent file."""
    from importlib import import_module
    hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')

    nonexistent_file = '/nonexistent/path/test.png'

    try:
        result = hpc_parallel_cpu.get_image_mean(nonexistent_file)

        # Should handle nonexistent file gracefully
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == 'test.png'
        assert result[1] is None

    except Exception as e:
        raise AssertionError(f"get_image_mean failed with nonexistent file: {e}")

def test_run_parallel_processing_no_images():
    """Test parallel processing with no images."""
    from importlib import import_module
    hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        # Mock config with empty input directory
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir

        # Patch the config in the HPC parallel CPU module
        with patch.object(hpc_parallel_cpu, 'config', mock_config):
            try:
                result = hpc_parallel_cpu.run_parallel_processing()
                # Should handle empty directory gracefully
                assert result == []
            except Exception as e:
                raise AssertionError(f"run_parallel_processing failed with no images: {e}")

def test_run_parallel_processing_with_images():
    """Test parallel processing with test images."""
    from importlib import import_module
    hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        # Create test images
        test_images = [
            ('test1.png', 100),
            ('test2.jpg', 150),
            ('test3.png', 200)
        ]

        cv2_available = True
        for filename, intensity in test_images:
            filepath = os.path.join(input_dir, filename)
            if not create_test_image_cv2(filepath, 20, 20, intensity):
                cv2_available = False

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir

        # Patch the config in the HPC parallel CPU module
        with patch.object(hpc_parallel_cpu, 'config', mock_config):
            try:
                result = hpc_parallel_cpu.run_parallel_processing()

                # Should process all images
                assert isinstance(result, list)
                assert len(result) == 3

                # Check that all results are tuples
                for item in result:
                    assert isinstance(item, tuple)
                    assert len(item) == 2
                    assert item[0] in ['test1.png', 'test2.jpg', 'test3.png']

                    if cv2_available:
                        # Should have valid mean values
                        assert item[1] is not None
                        assert isinstance(item[1], (int, float, np.number))

            except Exception as e:
                raise AssertionError(f"run_parallel_processing failed with test images: {e}")

def test_run_parallel_processing_multiprocessing():
    """Test that multiprocessing is used correctly."""
    from importlib import import_module
    hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        # Create test image
        test_image = os.path.join(input_dir, 'test.png')
        create_test_image_cv2(test_image, 15, 15, 128)

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir

        # Mock multiprocessing Pool
        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool.map.return_value = [('test.png', 128.0)]
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            # Mock cpu_count
            with patch('multiprocessing.cpu_count', return_value=4):
                # Patch the config in the HPC parallel CPU module
                with patch.object(hpc_parallel_cpu, 'config', mock_config):
                    try:
                        result = hpc_parallel_cpu.run_parallel_processing()

                        # Should use multiprocessing
                        mock_pool_class.assert_called_with(processes=4)
                        mock_pool.map.assert_called_once()

                        # Should return results
                        assert result == [('test.png', 128.0)]

                    except Exception as e:
                        raise AssertionError(f"Multiprocessing test failed: {e}")

def test_run_parallel_processing_error_handling():
    """Test error handling in parallel processing."""
    from importlib import import_module
    hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        # Create test image
        test_image = os.path.join(input_dir, 'test.png')
        create_test_image_cv2(test_image, 15, 15, 128)

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir

        # Mock multiprocessing Pool to raise an exception
        with patch('multiprocessing.Pool', side_effect=Exception("Pool error")):
            # Patch the config in the HPC parallel CPU module
            with patch.object(hpc_parallel_cpu, 'config', mock_config):
                try:
                    result = hpc_parallel_cpu.run_parallel_processing()
                    # Should handle error gracefully
                    assert result == []
                except Exception as e:
                    raise AssertionError(f"Error handling test failed: {e}")

def test_run_parallel_processing_mixed_files():
    """Test parallel processing with mixed file types."""
    from importlib import import_module
    hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        # Create image files
        create_test_image_cv2(os.path.join(input_dir, 'test1.png'), 10, 10, 100)
        create_test_image_cv2(os.path.join(input_dir, 'test2.jpg'), 10, 10, 150)

        # Create non-image files
        with open(os.path.join(input_dir, 'readme.txt'), 'w') as f:
            f.write('not an image')
        with open(os.path.join(input_dir, 'data.csv'), 'w') as f:
            f.write('not an image')

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir

        # Patch the config in the HPC parallel CPU module
        with patch.object(hpc_parallel_cpu, 'config', mock_config):
            try:
                result = hpc_parallel_cpu.run_parallel_processing()

                # Should process only image files
                assert isinstance(result, list)
                assert len(result) == 2

                # Check that only image files were processed
                filenames = [item[0] for item in result]
                assert 'test1.png' in filenames
                assert 'test2.jpg' in filenames
                assert 'readme.txt' not in filenames
                assert 'data.csv' not in filenames

            except Exception as e:
                raise AssertionError(f"Mixed files test failed: {e}")

def test_config_integration():
    """Test that the config module is properly integrated."""
    from importlib import import_module
    hpc_parallel_cpu = import_module('10_hpc_parallel_cpu')

    # Test that config object exists and has required attributes
    assert hasattr(hpc_parallel_cpu, 'config')
    config = hpc_parallel_cpu.config

    # Test that config has the required attributes
    required_attrs = ['INPUT_DIR']
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing attribute: {attr}"

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_hpc_parallel_cpu_import,
        test_get_image_mean_valid_image,
        test_get_image_mean_invalid_image,
        test_get_image_mean_nonexistent_file,
        test_run_parallel_processing_no_images,
        test_run_parallel_processing_with_images,
        test_run_parallel_processing_multiprocessing,
        test_run_parallel_processing_error_handling,
        test_run_parallel_processing_mixed_files,
        test_config_integration
    ]

    print("Running HPC parallel CPU tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("HPC parallel CPU tests completed.")
