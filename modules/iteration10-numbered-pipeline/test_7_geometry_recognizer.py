# test_7_geometry_recognizer.py
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image_cv2(filepath, width=100, height=100, pattern='solid', value=128):
    """Create a test image with different patterns using cv2."""
    try:
        import cv2

        if pattern == 'solid':
            # Create a solid color image
            img = np.full((height, width), value, dtype=np.uint8)
        elif pattern == 'rectangle':
            # Create image with a rectangle
            img = np.full((height, width), 50, dtype=np.uint8)
            cv2.rectangle(img, (10, 10), (width-10, height-10), 200, 2)
        elif pattern == 'circle':
            # Create image with a circle
            img = np.full((height, width), 50, dtype=np.uint8)
            center = (width//2, height//2)
            radius = min(width, height) // 4
            cv2.circle(img, center, radius, 200, 2)
        elif pattern == 'lines':
            # Create image with lines
            img = np.full((height, width), 50, dtype=np.uint8)
            for i in range(0, width, 20):
                cv2.line(img, (i, 0), (i, height), 200, 2)
        else:
            img = np.full((height, width), value, dtype=np.uint8)

        cv2.imwrite(filepath, img)
        return True
    except ImportError:
        # Create a dummy file if cv2 is not available
        with open(filepath, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG header
        return False

def test_geometry_recognizer_import():
    """Test that the geometry recognizer module can be imported successfully."""
    try:
        from importlib import import_module
        geometry_recognizer = import_module('7_geometry_recognizer')
        assert geometry_recognizer is not None
        assert hasattr(geometry_recognizer, 'find_geometry')
    except ImportError as e:
        raise AssertionError(f"Failed to import geometry recognizer module: {e}")

def test_find_geometry_no_images():
    """Test geometry recognizer behavior when no images are found."""
    from importlib import import_module
    geometry_recognizer = import_module('7_geometry_recognizer')

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

        # Patch the config in the geometry recognizer module
        with patch.object(geometry_recognizer, 'config', mock_config):
            try:
                result = geometry_recognizer.find_geometry()
                # Should handle empty directory gracefully
                assert result is None
            except Exception as e:
                raise AssertionError(f"find_geometry failed with no images: {e}")

def test_find_geometry_with_image():
    """Test geometry recognizer behavior with a test image."""
    from importlib import import_module
    geometry_recognizer = import_module('7_geometry_recognizer')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create test image
        test_image = os.path.join(input_dir, 'test.png')
        cv2_available = create_test_image_cv2(test_image, 50, 50, 'solid', 128)

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.OUTPUT_DIR = output_dir

        # Patch the config in the geometry recognizer module
        with patch.object(geometry_recognizer, 'config', mock_config):
            try:
                result = geometry_recognizer.find_geometry()

                if cv2_available:
                    # Should process the image
                    assert result is not None
                    assert isinstance(result, np.ndarray)

                    # Check output file was created
                    output_path = os.path.join(output_dir, "geometric_patterns.png")
                    assert os.path.exists(output_path)
                else:
                    # If cv2 not available, should handle gracefully
                    assert result is None

            except Exception as e:
                raise AssertionError(f"find_geometry failed with test image: {e}")

def test_find_geometry_with_shapes():
    """Test geometry recognizer with images containing geometric shapes."""
    from importlib import import_module
    geometry_recognizer = import_module('7_geometry_recognizer')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Test different geometric patterns
        patterns = ['rectangle', 'circle', 'lines']

        for pattern in patterns:
            # Create test image with specific pattern
            test_image = os.path.join(input_dir, f'test_{pattern}.png')
            cv2_available = create_test_image_cv2(test_image, 60, 60, pattern)

            if not cv2_available:
                continue

            # Mock config
            mock_config = MagicMock()
            mock_config.INPUT_DIR = input_dir
            mock_config.OUTPUT_DIR = output_dir

            # Patch the config in the geometry recognizer module
            with patch.object(geometry_recognizer, 'config', mock_config):
                try:
                    result = geometry_recognizer.find_geometry()

                    # Should detect edges/geometry
                    assert result is not None
                    assert isinstance(result, np.ndarray)
                    assert result.dtype == np.uint8

                    # Check that edges were detected (should have some non-zero pixels)
                    if pattern != 'solid':
                        assert np.any(result > 0), f"No edges detected for {pattern} pattern"

                    # Check output file was created
                    output_path = os.path.join(output_dir, "geometric_patterns.png")
                    assert os.path.exists(output_path)

                except Exception as e:
                    raise AssertionError(f"find_geometry failed with {pattern} pattern: {e}")

            # Remove test image for next iteration
            os.remove(test_image)

def test_find_geometry_corrupted_image():
    """Test geometry recognizer behavior with corrupted images."""
    from importlib import import_module
    geometry_recognizer = import_module('7_geometry_recognizer')

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

        # Patch the config in the geometry recognizer module
        with patch.object(geometry_recognizer, 'config', mock_config):
            try:
                result = geometry_recognizer.find_geometry()
                # Should handle corrupted image gracefully
                assert result is None
            except Exception as e:
                raise AssertionError(f"find_geometry failed with corrupted image: {e}")

def test_find_geometry_multiple_images():
    """Test geometry recognizer with multiple images (should use first one)."""
    from importlib import import_module
    geometry_recognizer = import_module('7_geometry_recognizer')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create multiple test images
        cv2_available = True
        for i in range(3):
            test_image = os.path.join(input_dir, f'test{i}.png')
            if not create_test_image_cv2(test_image, 40, 40, 'rectangle'):
                cv2_available = False

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.OUTPUT_DIR = output_dir

        # Patch the config in the geometry recognizer module
        with patch.object(geometry_recognizer, 'config', mock_config):
            try:
                result = geometry_recognizer.find_geometry()

                if cv2_available:
                    # Should process the first image
                    assert result is not None
                    assert isinstance(result, np.ndarray)

                    # Check output file was created
                    output_path = os.path.join(output_dir, "geometric_patterns.png")
                    assert os.path.exists(output_path)

            except Exception as e:
                raise AssertionError(f"find_geometry failed with multiple images: {e}")

def test_find_geometry_edge_detection():
    """Test that edge detection produces expected results."""
    from importlib import import_module
    geometry_recognizer = import_module('7_geometry_recognizer')

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create test image with strong edges
        test_image = os.path.join(input_dir, 'test.png')
        cv2_available = create_test_image_cv2(test_image, 80, 80, 'rectangle')

        # Mock config
        mock_config = MagicMock()
        mock_config.INPUT_DIR = input_dir
        mock_config.OUTPUT_DIR = output_dir

        # Patch the config in the geometry recognizer module
        with patch.object(geometry_recognizer, 'config', mock_config):
            try:
                result = geometry_recognizer.find_geometry()

                if cv2_available:
                    # Check edge detection results
                    assert result is not None
                    assert isinstance(result, np.ndarray)
                    assert result.dtype == np.uint8
                    assert result.shape == (80, 80)

                    # Should have detected edges (non-zero pixels)
                    assert np.any(result > 0), "No edges detected"

                    # Edge pixels should be either 0 or 255 (binary edge map)
                    unique_values = np.unique(result)
                    assert len(unique_values) <= 2, f"Too many unique values in edge map: {unique_values}"
                    assert all(val in [0, 255] for val in unique_values), f"Invalid edge values: {unique_values}"

            except Exception as e:
                raise AssertionError(f"Edge detection test failed: {e}")

def test_config_integration():
    """Test that the config module is properly integrated."""
    from importlib import import_module
    geometry_recognizer = import_module('7_geometry_recognizer')

    # Test that config object exists and has required attributes
    assert hasattr(geometry_recognizer, 'config')
    config = geometry_recognizer.config

    # Test that config has the required attributes
    required_attrs = ['INPUT_DIR', 'OUTPUT_DIR']
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing attribute: {attr}"

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_geometry_recognizer_import,
        test_find_geometry_no_images,
        test_find_geometry_with_image,
        test_find_geometry_with_shapes,
        test_find_geometry_corrupted_image,
        test_find_geometry_multiple_images,
        test_find_geometry_edge_detection,
        test_config_integration
    ]

    print("Running geometry recognizer tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("Geometry recognizer tests completed.")
