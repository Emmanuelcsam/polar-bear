# test_6_deviation_detector.py
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image_cv2(filepath, width=100, height=100, pixel_value=128):
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

def test_deviation_detector_import():
    """Test that the deviation detector module can be imported successfully."""
    try:
        from importlib import import_module
        deviation_detector = import_module('6_deviation_detector')
        assert deviation_detector is not None
        assert hasattr(deviation_detector, 'detect_deviations')
    except ImportError as e:
        raise AssertionError(f"Failed to import deviation detector module: {e}")

def test_detect_deviations_no_images():
    """Test deviation detector behavior when no images are found."""
    from importlib import import_module
    deviation_detector = import_module('6_deviation_detector')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Mock config with empty input directory
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.OUTPUT_DIR = output_dir

        # Patch the config in the deviation detector module
        with patch.object(deviation_detector, 'config', mock_config):
            try:
                result = deviation_detector.detect_deviations()
                # Should handle empty directory gracefully
                assert result is None
            except Exception as e:
                raise AssertionError(f"detect_deviations failed with no images: {e}")

def test_detect_deviations_single_image():
    """Test deviation detector behavior with single image."""
    from importlib import import_module
    deviation_detector = import_module('6_deviation_detector')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create single test image
        test_image = os.path.join(input_dir, 'test.png')
        cv2_available = create_test_image_cv2(test_image, 50, 50, 128)

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.OUTPUT_DIR = output_dir

        # Patch the config in the deviation detector module
        with patch.object(deviation_detector, 'config', mock_config):
            try:
                result = deviation_detector.detect_deviations()

                if cv2_available:
                    # Should process single image (comparing to itself)
                    assert result is not None
                    assert isinstance(result, np.ndarray)

                    # Check output file was created
                    output_path = os.path.join(output_dir, "anomaly_detection.png")
                    assert os.path.exists(output_path)
                else:
                    # If cv2 not available, should handle gracefully
                    assert result is None

            except Exception as e:
                raise AssertionError(f"detect_deviations failed with single image: {e}")

def test_detect_deviations_multiple_images():
    """Test deviation detector behavior with multiple images."""
    from importlib import import_module
    deviation_detector = import_module('6_deviation_detector')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create multiple test images with different intensities
        images = [
            ('test1.png', 100),
            ('test2.jpg', 150),
            ('test3.png', 200)
        ]

        cv2_available = True
        for filename, intensity in images:
            filepath = os.path.join(input_dir, filename)
            if not create_test_image_cv2(filepath, 30, 30, intensity):
                cv2_available = False

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.OUTPUT_DIR = output_dir

        # Patch the config in the deviation detector module
        with patch.object(deviation_detector, 'config', mock_config):
            try:
                result = deviation_detector.detect_deviations()

                if cv2_available:
                    # Should process multiple images
                    assert result is not None
                    assert isinstance(result, np.ndarray)
                    assert result.ndim == 3  # Should be color image (BGR)

                    # Check output file was created
                    output_path = os.path.join(output_dir, "anomaly_detection.png")
                    assert os.path.exists(output_path)
                else:
                    # If cv2 not available, should handle gracefully
                    pass

            except Exception as e:
                raise AssertionError(f"detect_deviations failed with multiple images: {e}")

def test_detect_deviations_threshold():
    """Test deviation detector with different threshold values."""
    from importlib import import_module
    deviation_detector = import_module('6_deviation_detector')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create test images
        test_image1 = os.path.join(input_dir, 'test1.png')
        test_image2 = os.path.join(input_dir, 'test2.png')
        cv2_available = create_test_image_cv2(test_image1, 20, 20, 100)
        create_test_image_cv2(test_image2, 20, 20, 150)

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.OUTPUT_DIR = output_dir

        # Test different threshold values
        thresholds = [10, 50, 100]

        for threshold in thresholds:
            # Patch the config in the deviation detector module
            with patch.object(deviation_detector, 'config', mock_config):
                try:
                    result = deviation_detector.detect_deviations(threshold=threshold)

                    if cv2_available:
                        # Should work with different thresholds
                        assert result is not None
                        assert isinstance(result, np.ndarray)

                except Exception as e:
                    raise AssertionError(f"detect_deviations failed with threshold {threshold}: {e}")

def test_detect_deviations_corrupted_image():
    """Test deviation detector behavior with corrupted images."""
    from importlib import import_module
    deviation_detector = import_module('6_deviation_detector')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create a corrupted image file
        corrupted_image = os.path.join(input_dir, 'corrupted.png')
        with open(corrupted_image, 'wb') as f:
            f.write(b'not_a_real_image')

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.OUTPUT_DIR = output_dir

        # Patch the config in the deviation detector module
        with patch.object(deviation_detector, 'config', mock_config):
            try:
                result = deviation_detector.detect_deviations()
                # Should handle corrupted image gracefully
                assert result is None
            except Exception as e:
                raise AssertionError(f"detect_deviations failed with corrupted image: {e}")

def test_detect_deviations_mixed_formats():
    """Test deviation detector with mixed image formats."""
    from importlib import import_module
    deviation_detector = import_module('6_deviation_detector')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create images with different formats
        formats = [
            ('test.png', 128),
            ('test.jpg', 140),
            ('test.PNG', 160),
            ('test.JPG', 180)
        ]

        cv2_available = True
        for filename, intensity in formats:
            filepath = os.path.join(input_dir, filename)
            if not create_test_image_cv2(filepath, 25, 25, intensity):
                cv2_available = False

        # Add non-image files
        with open(os.path.join(input_dir, 'readme.txt'), 'w') as f:
            f.write('not an image')

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.OUTPUT_DIR = output_dir

        # Patch the config in the deviation detector module
        with patch.object(deviation_detector, 'config', mock_config):
            try:
                result = deviation_detector.detect_deviations()

                if cv2_available:
                    # Should process only image files
                    assert result is not None
                    assert isinstance(result, np.ndarray)

            except Exception as e:
                raise AssertionError(f"detect_deviations failed with mixed formats: {e}")

def test_config_integration():
    """Test that the config module is properly integrated."""
    from importlib import import_module
    deviation_detector = import_module('6_deviation_detector')

    # Test that config object exists and has required attributes
    assert hasattr(deviation_detector, 'config')
    config = deviation_detector.config

    # Test that config has the required attributes
    required_attrs = ['INPUT_DIR', 'OUTPUT_DIR']
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing attribute: {attr}"

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_deviation_detector_import,
        test_detect_deviations_no_images,
        test_detect_deviations_single_image,
        test_detect_deviations_multiple_images,
        test_detect_deviations_threshold,
        test_detect_deviations_corrupted_image,
        test_detect_deviations_mixed_formats,
        test_config_integration
    ]

    print("Running deviation detector tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("Deviation detector tests completed.")
