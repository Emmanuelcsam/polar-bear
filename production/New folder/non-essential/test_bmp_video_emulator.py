"""
Comprehensive test suite for BMP Video Emulator.
Tests all classes, methods, and functionality.
"""

import unittest
import tempfile
import os
import time
import threading
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
import tkinter as tk

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmp_video_emulator import (
    BMPVideoEmulator, 
    EmulatedPylonGrabber, 
    VideoEmulatorGUI
)
from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE


class TestBMPVideoEmulator(unittest.TestCase):
    """Test cases for BMPVideoEmulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = [255, 255, 255]  # White square
        
        # Save test image
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test.bmp")
        cv2.imwrite(self.test_image_path, self.test_image)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_init_with_valid_image(self):
        """Test initialization with a valid image."""
        emulator = BMPVideoEmulator(self.test_image_path, frame_rate=30)
        self.assertEqual(emulator.frame_rate, 30)
        self.assertEqual(emulator.frame_interval, 1.0/30)
        self.assertIsNotNone(emulator.original_frame)
        self.assertEqual(emulator.original_frame.shape, (100, 100, 3))
        
    def test_init_with_invalid_image_path(self):
        """Test initialization with invalid image path."""
        with self.assertRaises(FileNotFoundError):
            BMPVideoEmulator("nonexistent.bmp")
            
    def test_init_with_invalid_image_file(self):
        """Test initialization with invalid image file."""
        # Create an invalid image file
        invalid_image_path = os.path.join(self.temp_dir, "invalid.bmp")
        with open(invalid_image_path, 'w') as f:
            f.write("This is not a valid image")
            
        with self.assertRaises(ValueError):
            BMPVideoEmulator(invalid_image_path)
            
    def test_start_and_stop(self):
        """Test starting and stopping the emulator."""
        emulator = BMPVideoEmulator(self.test_image_path)
        
        # Test start
        emulator.start()
        self.assertTrue(emulator.is_running.is_set())
        self.assertTrue(emulator.emulation_thread.is_alive())
        
        # Test stop
        emulator.stop()
        self.assertFalse(emulator.is_running.is_set())
        
    def test_read_before_start(self):
        """Test reading frame before starting emulator."""
        emulator = BMPVideoEmulator(self.test_image_path)
        frame = emulator.read()
        self.assertIsNone(frame)
        
    def test_read_after_start(self):
        """Test reading frame after starting emulator."""
        emulator = BMPVideoEmulator(self.test_image_path, frame_rate=60)
        emulator.start()
        
        # Wait a bit for the emulation to start
        time.sleep(0.1)
        
        frame = emulator.read()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (100, 100, 3))
        
        emulator.stop()
        
    def test_frame_count(self):
        """Test frame count tracking."""
        emulator = BMPVideoEmulator(self.test_image_path, frame_rate=10)
        emulator.start()
        
        # Wait for some frames to be processed
        time.sleep(0.2)
        
        frame_count = emulator.get_frame_count()
        self.assertGreater(frame_count, 0)
        
        emulator.stop()
        
    def test_thread_safety(self):
        """Test thread safety of frame reading."""
        emulator = BMPVideoEmulator(self.test_image_path)
        emulator.start()
        
        # Create multiple threads reading frames
        frames = []
        errors = []
        
        def read_frames():
            try:
                for _ in range(10):
                    frame = emulator.read()
                    if frame is not None:
                        frames.append(frame)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
                
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=read_frames)
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        emulator.stop()
        
        # Check that no errors occurred
        self.assertEqual(len(errors), 0)
        self.assertGreater(len(frames), 0)


class TestEmulatedPylonGrabber(unittest.TestCase):
    """Test cases for EmulatedPylonGrabber class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = [255, 255, 255]
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test.bmp")
        cv2.imwrite(self.test_image_path, self.test_image)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_init_with_emulation(self):
        """Test initialization with emulation enabled."""
        grabber = EmulatedPylonGrabber(
            use_emulation=True,
            image_path=self.test_image_path,
            frame_rate=30
        )
        self.assertTrue(grabber.use_emulation)
        self.assertIsNotNone(grabber.emulator)
        
    def test_init_without_emulation(self):
        """Test initialization with emulation disabled."""
        # When Pylon is not available, emulation is always used regardless of setting
        if not PYLON_AVAILABLE:
            grabber = EmulatedPylonGrabber(use_emulation=False)
            self.assertFalse(grabber.use_emulation)
            # Emulator will be created because Pylon is not available
            self.assertIsNotNone(grabber.emulator)
        else:
            grabber = EmulatedPylonGrabber(use_emulation=False)
            self.assertFalse(grabber.use_emulation)
            self.assertIsNone(grabber.emulator)
        
    @patch('bmp_video_emulator.PYLON_AVAILABLE', False)
    def test_run_with_emulation(self):
        """Test running with emulation when Pylon is not available."""
        grabber = EmulatedPylonGrabber(
            use_emulation=True,
            image_path=self.test_image_path
        )
        
        # Start the grabber
        grabber.start()
        time.sleep(0.1)
        
        # Check that it's running
        self.assertTrue(grabber.is_running.is_set())
        
        # Read a frame
        frame = grabber.read()
        self.assertIsNotNone(frame)
        
        # Stop the grabber
        grabber.stop()
        grabber.join(timeout=1.0)
        
    def test_run_with_real_camera(self):
        """Test running with real camera when available."""
        # Mock PYLON_AVAILABLE to be True
        with patch('bmp_video_emulator.PYLON_AVAILABLE', True):
            grabber = EmulatedPylonGrabber(use_emulation=False)
            
            # Mock the parent class run method
            with patch.object(PylonFrameGrabber, 'run') as mock_run:
                grabber.run()
                mock_run.assert_called_once()


class TestVideoEmulatorGUI(unittest.TestCase):
    """Test cases for VideoEmulatorGUI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = [255, 255, 255]
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test.bmp")
        cv2.imwrite(self.test_image_path, self.test_image)
        
        # Create a test root window
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        try:
            self.root.destroy()
        except:
            pass  # Ignore errors if window already destroyed
        
    def test_gui_initialization(self):
        """Test GUI initialization."""
        gui = VideoEmulatorGUI(self.root)
        
        # Check that widgets were created
        self.assertIsNotNone(gui.image_path_var)
        self.assertIsNotNone(gui.frame_rate_var)
        self.assertIsNotNone(gui.use_emulation_var)
        self.assertIsNotNone(gui.start_stop_btn)
        self.assertIsNotNone(gui.status_var)
        
    def test_start_emulation(self):
        """Test starting emulation through GUI."""
        gui = VideoEmulatorGUI(self.root)
        
        # Set test values
        gui.image_path_var.set(self.test_image_path)
        gui.frame_rate_var.set(30)
        gui.use_emulation_var.set(True)
        
        # Mock the EmulatedPylonGrabber to avoid actual threading
        with patch('bmp_video_emulator.EmulatedPylonGrabber') as mock_grabber_class:
            mock_grabber = Mock()
            mock_grabber_class.return_value = mock_grabber
            
            # Start emulation
            gui._start_emulation()
            
            # Check that grabber was created and started
            mock_grabber_class.assert_called_once()
            mock_grabber.start.assert_called_once()
            
            # Check GUI state
            self.assertTrue(gui.is_running)
            self.assertEqual(gui.status_var.get(), "Running")
            
    def test_stop_emulation(self):
        """Test stopping emulation through GUI."""
        gui = VideoEmulatorGUI(self.root)
        gui.is_running = True
        gui.grabber = Mock()
        
        # Stop emulation
        gui._stop_emulation()
        
        # Check that grabber was stopped
        gui.grabber.stop.assert_called_once()
        gui.grabber.join.assert_called_once()
        
        # Check GUI state
        self.assertFalse(gui.is_running)
        self.assertEqual(gui.status_var.get(), "Stopped")
        
    def test_log_message(self):
        """Test logging messages to GUI."""
        gui = VideoEmulatorGUI(self.root)
        
        # Add a test message
        gui._log_message("Test message")
        
        # Check that message was added to log
        log_content = gui.log_text.get("1.0", tk.END)
        self.assertIn("Test message", log_content)
        
    def test_on_closing(self):
        """Test window closing behavior."""
        gui = VideoEmulatorGUI(self.root)
        gui.is_running = True
        gui.grabber = Mock()
        
        # Mock the stop method
        with patch.object(gui, '_stop_emulation') as mock_stop:
            gui._on_closing()
            mock_stop.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = [255, 255, 255]
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test.bmp")
        cv2.imwrite(self.test_image_path, self.test_image)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_end_to_end_emulation(self):
        """Test complete end-to-end emulation workflow."""
        # Create emulator
        emulator = BMPVideoEmulator(self.test_image_path, frame_rate=10)
        
        # Start emulation
        emulator.start()
        time.sleep(0.2)  # Wait for frames to be generated
        
        # Read frames
        frames = []
        for _ in range(5):
            frame = emulator.read()
            if frame is not None:
                frames.append(frame)
            time.sleep(0.05)
            
        # Stop emulation
        emulator.stop()
        
        # Verify results
        self.assertGreater(len(frames), 0)
        self.assertGreater(emulator.get_frame_count(), 0)
        
        for frame in frames:
            self.assertEqual(frame.shape, (100, 100, 3))
            
    def test_grabber_integration(self):
        """Test integration between emulator and grabber."""
        grabber = EmulatedPylonGrabber(
            use_emulation=True,
            image_path=self.test_image_path,
            frame_rate=10
        )
        
        # Start grabber
        grabber.start()
        time.sleep(0.2)
        
        # Read frames through grabber interface
        frames = []
        for _ in range(5):
            frame = grabber.read()
            if frame is not None:
                frames.append(frame)
            time.sleep(0.05)
            
        # Stop grabber
        grabber.stop()
        grabber.join(timeout=1.0)
        
        # Verify results
        self.assertGreater(len(frames), 0)
        
        for frame in frames:
            self.assertEqual(frame.shape, (100, 100, 3))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_emulator_with_missing_file(self):
        """Test emulator with missing image file."""
        with self.assertRaises(FileNotFoundError):
            BMPVideoEmulator("nonexistent.bmp")
            
    def test_emulator_with_corrupted_file(self):
        """Test emulator with corrupted image file."""
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            f.write(b"This is not a valid BMP file")
            corrupted_path = f.name
            
        try:
            with self.assertRaises(ValueError):
                BMPVideoEmulator(corrupted_path)
        finally:
            os.unlink(corrupted_path)
            
    def test_grabber_with_invalid_parameters(self):
        """Test grabber with invalid parameters."""
        with self.assertRaises(FileNotFoundError):
            EmulatedPylonGrabber(
                use_emulation=True,
                image_path="nonexistent.bmp"
            )
            
    def test_thread_safety_under_stress(self):
        """Test thread safety under stress conditions."""
        # Create a small test image
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[10:40, 10:40] = [255, 255, 255]
        
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            cv2.imwrite(f.name, test_image)
            image_path = f.name
            
        try:
            emulator = BMPVideoEmulator(image_path, frame_rate=60)
            emulator.start()
            
            # Create many threads reading frames
            frames = []
            errors = []
            
            def stress_test():
                try:
                    for _ in range(20):
                        frame = emulator.read()
                        if frame is not None:
                            frames.append(frame)
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(e)
                    
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=stress_test)
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
                
            emulator.stop()
            
            # Verify no errors occurred
            self.assertEqual(len(errors), 0)
            self.assertGreater(len(frames), 0)
            
        finally:
            os.unlink(image_path)


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestBMPVideoEmulator,
        TestEmulatedPylonGrabber,
        TestVideoEmulatorGUI,
        TestIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
        
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 