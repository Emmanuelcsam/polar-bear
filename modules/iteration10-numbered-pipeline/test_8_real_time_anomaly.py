# test_8_real_time_anomaly.py
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_real_time_anomaly_import():
    """Test that the real-time anomaly module can be imported successfully."""
    try:
        from importlib import import_module
        real_time_anomaly = import_module('8_real_time_anomaly')
        assert real_time_anomaly is not None
        assert hasattr(real_time_anomaly, 'run_real_time_detection')
    except ImportError as e:
        raise AssertionError(f"Failed to import real-time anomaly module: {e}")

def test_run_real_time_detection_no_camera():
    """Test real-time anomaly detector behavior when no camera is available."""
    from importlib import import_module
    real_time_anomaly = import_module('8_real_time_anomaly')

    # Mock cv2.VideoCapture to simulate no camera
    with patch('cv2.VideoCapture') as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        try:
            result = real_time_anomaly.run_real_time_detection()
            # Should handle no camera gracefully
            assert result is False
        except Exception as e:
            raise AssertionError(f"run_real_time_detection failed with no camera: {e}")

def test_run_real_time_detection_with_camera():
    """Test real-time anomaly detector behavior with simulated camera."""
    from importlib import import_module
    real_time_anomaly = import_module('8_real_time_anomaly')

    try:
        import cv2
        import numpy as np
        cv2_available = True
    except ImportError:
        cv2_available = False

    if not cv2_available:
        print("OpenCV not available, skipping camera simulation test")
        return

    # Create mock frame data
    frame_data = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock cv2.VideoCapture to simulate camera
    with patch('cv2.VideoCapture') as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame_data)
        mock_videocapture.return_value = mock_cap

        # Mock cv2.waitKey to simulate immediate quit
        with patch('cv2.waitKey', return_value=ord('q')):
            # Mock cv2.imshow to prevent actual window creation
            with patch('cv2.imshow'):
                # Mock cv2.destroyAllWindows
                with patch('cv2.destroyAllWindows'):
                    try:
                        result = real_time_anomaly.run_real_time_detection()
                        # Should run successfully with mocked camera
                        assert result is True
                    except Exception as e:
                        raise AssertionError(f"run_real_time_detection failed with simulated camera: {e}")

def test_run_real_time_detection_background_capture():
    """Test real-time anomaly detector background capture simulation."""
    from importlib import import_module
    real_time_anomaly = import_module('8_real_time_anomaly')

    try:
        import cv2
        import numpy as np
        cv2_available = True
    except ImportError:
        cv2_available = False

    if not cv2_available:
        print("OpenCV not available, skipping background capture test")
        return

    # Create mock frame data
    frame_data = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock cv2.VideoCapture to simulate camera
    with patch('cv2.VideoCapture') as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame_data)
        mock_videocapture.return_value = mock_cap

        # Simulate key presses: 'b' for background capture, then 'q' for quit
        key_sequence = [ord('b'), ord('q')]
        key_iter = iter(key_sequence)

        def mock_waitkey(*args):
            try:
                return next(key_iter)
            except StopIteration:
                return ord('q')

        with patch('cv2.waitKey', side_effect=mock_waitkey):
            with patch('cv2.imshow'):
                with patch('cv2.destroyAllWindows'):
                    try:
                        result = real_time_anomaly.run_real_time_detection()
                        # Should complete background capture sequence
                        assert result is True
                    except Exception as e:
                        raise AssertionError(f"Background capture test failed: {e}")

def test_run_real_time_detection_read_failure():
    """Test real-time anomaly detector behavior when camera read fails."""
    from importlib import import_module
    real_time_anomaly = import_module('8_real_time_anomaly')

    # Mock cv2.VideoCapture with read failure
    with patch('cv2.VideoCapture') as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # Simulate read failure
        mock_videocapture.return_value = mock_cap

        with patch('cv2.destroyAllWindows'):
            try:
                result = real_time_anomaly.run_real_time_detection()
                # Should handle read failure gracefully
                assert result is True
            except Exception as e:
                raise AssertionError(f"run_real_time_detection failed with read failure: {e}")

def test_run_real_time_detection_exception_handling():
    """Test that the real-time detector handles exceptions gracefully."""
    from importlib import import_module
    real_time_anomaly = import_module('8_real_time_anomaly')

    # Mock cv2.VideoCapture to raise an exception
    with patch('cv2.VideoCapture', side_effect=Exception("Camera error")):
        try:
            result = real_time_anomaly.run_real_time_detection()
            # Should handle exception gracefully
            assert result is False
        except Exception as e:
            # Should not raise unhandled exceptions
            raise AssertionError(f"Unhandled exception in real-time detector: {e}")

def test_run_real_time_detection_cv2_functions():
    """Test that the detector uses OpenCV functions correctly."""
    from importlib import import_module
    real_time_anomaly = import_module('8_real_time_anomaly')

    try:
        import cv2
        import numpy as np
        cv2_available = True
    except ImportError:
        cv2_available = False

    if not cv2_available:
        print("OpenCV not available, skipping CV2 functions test")
        return

    # Create mock frame data
    frame_data = np.full((100, 100, 3), 128, dtype=np.uint8)

    # Mock cv2.VideoCapture
    with patch('cv2.VideoCapture') as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame_data)
        mock_videocapture.return_value = mock_cap

        # Mock cv2 functions to track their usage
        with patch('cv2.cvtColor') as mock_cvtcolor:
            with patch('cv2.GaussianBlur') as mock_blur:
                with patch('cv2.waitKey', return_value=ord('q')):
                    with patch('cv2.imshow'):
                        with patch('cv2.destroyAllWindows'):
                            try:
                                result = real_time_anomaly.run_real_time_detection()

                                # Check that cv2 functions were called
                                mock_cvtcolor.assert_called()
                                mock_blur.assert_called()
                                assert result is True

                            except Exception as e:
                                raise AssertionError(f"CV2 functions test failed: {e}")

def test_run_real_time_detection_anomaly_detection():
    """Test anomaly detection functionality."""
    from importlib import import_module
    real_time_anomaly = import_module('8_real_time_anomaly')

    try:
        import cv2
        import numpy as np
        cv2_available = True
    except ImportError:
        cv2_available = False

    if not cv2_available:
        print("OpenCV not available, skipping anomaly detection test")
        return

    # Create mock frame data
    frame_data = np.full((100, 100, 3), 128, dtype=np.uint8)

    # Mock cv2.VideoCapture
    with patch('cv2.VideoCapture') as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame_data)
        mock_videocapture.return_value = mock_cap

        # Mock cv2 functions for anomaly detection
        with patch('cv2.cvtColor', return_value=np.full((100, 100), 128, dtype=np.uint8)):
            with patch('cv2.GaussianBlur', return_value=np.full((100, 100), 128, dtype=np.uint8)):
                with patch('cv2.absdiff', return_value=np.full((100, 100), 50, dtype=np.uint8)):
                    with patch('cv2.threshold', return_value=(None, np.full((100, 100), 255, dtype=np.uint8))):
                        with patch('cv2.findContours', return_value=([], None)):

                            # Simulate background capture followed by quit
                            key_sequence = [ord('b'), ord('q')]
                            key_iter = iter(key_sequence)

                            def mock_waitkey(*args):
                                try:
                                    return next(key_iter)
                                except StopIteration:
                                    return ord('q')

                            with patch('cv2.waitKey', side_effect=mock_waitkey):
                                with patch('cv2.imshow'):
                                    with patch('cv2.destroyAllWindows'):
                                        try:
                                            result = real_time_anomaly.run_real_time_detection()
                                            # Should complete anomaly detection sequence
                                            assert result is True
                                        except Exception as e:
                                            raise AssertionError(f"Anomaly detection test failed: {e}")

if __name__ == "__main__":
    import unittest

    # Create a simple test suite
    test_functions = [
        test_real_time_anomaly_import,
        test_run_real_time_detection_no_camera,
        test_run_real_time_detection_with_camera,
        test_run_real_time_detection_background_capture,
        test_run_real_time_detection_read_failure,
        test_run_real_time_detection_exception_handling,
        test_run_real_time_detection_cv2_functions,
        test_run_real_time_detection_anomaly_detection
    ]

    print("Running real-time anomaly detector tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("Real-time anomaly detector tests completed.")
