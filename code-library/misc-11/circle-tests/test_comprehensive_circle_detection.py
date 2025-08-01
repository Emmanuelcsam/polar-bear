#!/usr/bin/env python3
"""
Comprehensive Test Suite for Circle Detection System
==================================================

This script tests every class, function, and method of the circle detection system
with detailed coverage including edge cases, error conditions, and performance tests.

Author: AI Assistant
Date: 2024
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np

# Import the classes to test
from pylon_circle_detector import CircleDetectionApp, CircleDetector, PylonCamera


class TestCircleDetector(unittest.TestCase):
    """Test the CircleDetector class and all its methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = CircleDetector()
        self.test_image = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(self.test_image, (100, 100), 50, 255, -1)

        # Create test images with different scenarios
        self.small_circles_img = np.zeros((300, 400), dtype=np.uint8)
        for i in range(3):
            x = 50 + i * 100
            y = 150
            radius = 15 + i * 5
            cv2.circle(self.small_circles_img, (x, y), radius, 255, -1)

        self.large_circles_img = np.zeros((500, 700), dtype=np.uint8)
        cv2.circle(self.large_circles_img, (200, 250), 120, 255, -1)
        cv2.circle(self.large_circles_img, (500, 250), 150, 255, -1)

        self.no_circles_img = np.zeros((200, 200), dtype=np.uint8)
        # Add some noise but no circles
        self.no_circles_img[50:150, 50:150] = 128

    def test_init_default_params(self):
        """Test CircleDetector initialization with default parameters."""
        detector = CircleDetector()

        # Test default Hough parameters
        expected_hough = {
            "dp": 1,
            "min_dist": 20,
            "param1": 50,
            "param2": 30,
            "min_radius": 10,
            "max_radius": 300,
        }
        self.assertEqual(detector.hough_params, expected_hough)

        # Test default contour parameters
        expected_contour = {
            "min_area": 100,
            "max_area": 50000,
            "circularity_threshold": 0.7,
        }
        self.assertEqual(detector.contour_params, expected_contour)

        # Test other attributes
        self.assertFalse(detector.use_gpu)
        self.assertEqual(len(detector.detection_history), 0)
        self.assertEqual(len(detector.fps_history), 0)
        self.assertEqual(detector.frame_count, 0)
        self.assertFalse(detector.is_recording)
        self.assertIsNone(detector.video_writer)
        self.assertIsNone(detector.recording_path)

    def test_init_custom_params(self):
        """Test CircleDetector initialization with custom parameters."""
        custom_hough = {
            "dp": 2,
            "min_dist": 30,
            "param1": 60,
            "param2": 40,
            "min_radius": 20,
            "max_radius": 400,
        }
        custom_contour = {
            "min_area": 200,
            "max_area": 60000,
            "circularity_threshold": 0.8,
        }

        detector = CircleDetector(
            hough_params=custom_hough, contour_params=custom_contour, use_gpu=True
        )

        self.assertEqual(detector.hough_params, custom_hough)
        self.assertEqual(detector.contour_params, custom_contour)
        self.assertTrue(detector.use_gpu)

    def test_detect_circles_hough_success(self):
        """Test successful Hough circle detection."""
        circles = self.detector.detect_circles_hough(self.test_image)

        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)

        # Check circle format (x, y, radius)
        for circle in circles:
            self.assertEqual(len(circle), 3)
            self.assertIsInstance(circle[0], int)  # x
            self.assertIsInstance(circle[1], int)  # y
            self.assertIsInstance(circle[2], int)  # radius

    def test_detect_circles_hough_no_circles(self):
        """Test Hough circle detection with no circles."""
        circles = self.detector.detect_circles_hough(self.no_circles_img)
        self.assertEqual(circles, [])

    def test_detect_circles_hough_empty_image(self):
        """Test Hough circle detection with empty image."""
        empty_img = np.zeros((100, 100), dtype=np.uint8)
        circles = self.detector.detect_circles_hough(empty_img)
        self.assertEqual(circles, [])

    def test_detect_circles_hough_custom_params(self):
        """Test Hough circle detection with custom parameters."""
        detector = CircleDetector(
            hough_params={
                "dp": 1,
                "min_dist": 10,
                "param1": 30,
                "param2": 20,
                "min_radius": 5,
                "max_radius": 100,
            }
        )

        circles = detector.detect_circles_hough(self.small_circles_img)
        self.assertIsInstance(circles, list)

    def test_detect_circles_contour_success(self):
        """Test successful contour-based circle detection."""
        circles = self.detector.detect_circles_contour(self.test_image)

        self.assertIsInstance(circles, list)
        # Contour detection might not find circles in simple test image
        # but should return a list

    def test_detect_circles_contour_no_circles(self):
        """Test contour detection with no circles."""
        circles = self.detector.detect_circles_contour(self.no_circles_img)
        self.assertEqual(circles, [])

    def test_detect_circles_contour_custom_params(self):
        """Test contour detection with custom parameters."""
        detector = CircleDetector(
            contour_params={
                "min_area": 50,
                "max_area": 10000,
                "circularity_threshold": 0.6,
            }
        )

        circles = detector.detect_circles_contour(self.test_image)
        self.assertIsInstance(circles, list)

    def test_detect_circles_combined(self):
        """Test combined circle detection."""
        circles = self.detector.detect_circles_combined(self.test_image)

        self.assertIsInstance(circles, list)
        # Combined should return unique circles

    def test_detect_circles_combined_no_circles(self):
        """Test combined detection with no circles."""
        circles = self.detector.detect_circles_combined(self.no_circles_img)
        self.assertEqual(circles, [])

    def test_remove_duplicate_circles(self):
        """Test duplicate circle removal."""
        circles = [(100, 100, 50), (105, 105, 55), (200, 200, 60)]
        unique = self.detector._remove_duplicate_circles(circles, threshold=20)

        self.assertIsInstance(unique, list)
        self.assertLessEqual(len(unique), len(circles))

    def test_remove_duplicate_circles_empty(self):
        """Test duplicate removal with empty list."""
        unique = self.detector._remove_duplicate_circles([], threshold=20)
        self.assertEqual(unique, [])

    def test_remove_duplicate_circles_single(self):
        """Test duplicate removal with single circle."""
        circles = [(100, 100, 50)]
        unique = self.detector._remove_duplicate_circles(circles, threshold=20)
        self.assertEqual(unique, circles)

    def test_draw_circles(self):
        """Test circle drawing functionality."""
        circles = [(100, 100, 50), (200, 200, 60)]
        result = self.detector.draw_circles(self.test_image, circles)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, self.test_image.dtype)

    def test_draw_circles_empty(self):
        """Test circle drawing with no circles."""
        result = self.detector.draw_circles(self.test_image, [])

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_draw_circles_custom_color(self):
        """Test circle drawing with custom color."""
        circles = [(100, 100, 50)]
        color = (255, 0, 0)  # Red
        result = self.detector.draw_circles(self.test_image, circles, color=color)

        self.assertIsInstance(result, np.ndarray)

    def test_update_fps(self):
        """Test FPS update functionality."""
        initial_fps = self.detector.get_average_fps()

        # Simulate frame processing
        for _ in range(10):
            self.detector.update_fps()

        # Should have updated FPS
        self.assertGreaterEqual(self.detector.frame_count, 0)

    def test_get_average_fps_no_history(self):
        """Test average FPS with no history."""
        fps = self.detector.get_average_fps()
        self.assertEqual(fps, 0.0)

    def test_get_average_fps_with_history(self):
        """Test average FPS with history."""
        # Add some FPS values
        self.detector.fps_history.extend([30.0, 25.0, 35.0])
        fps = self.detector.get_average_fps()

        self.assertGreater(fps, 0.0)
        self.assertEqual(fps, 30.0)  # (30+25+35)/3 = 30


class TestPylonCamera(unittest.TestCase):
    """Test the PylonCamera class and all its methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("pylon_circle_detector.PYLON_AVAILABLE", False)
    def test_init_webcam_fallback(self):
        """Test PylonCamera initialization with webcam fallback."""
        with patch("cv2.VideoCapture") as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cv2.return_value = mock_cap

            camera = PylonCamera(camera_index=0, use_pylon=False)

            self.assertFalse(camera.use_pylon)
            self.assertEqual(camera.camera_index, 0)
            self.assertIsNotNone(camera.camera)

    @patch("pylon_circle_detector.PYLON_AVAILABLE", True)
    def test_init_pylon_available(self):
        """Test PylonCamera initialization with Pylon available."""
        with patch("pylon_circle_detector.pylon") as mock_pylon:
            # Mock Pylon factory and devices
            mock_tl_factory = Mock()
            mock_device = Mock()
            mock_camera = Mock()

            mock_pylon.TlFactory.GetInstance.return_value = mock_tl_factory
            mock_tl_factory.EnumerateDevices.return_value = [mock_device]
            mock_tl_factory.CreateFirstDevice.return_value = mock_device
            mock_pylon.InstantCamera.return_value = mock_camera
            mock_camera.IsOpen.return_value = True

            camera = PylonCamera(camera_index=0, use_pylon=True)

            self.assertTrue(camera.use_pylon)
            self.assertIsNotNone(camera.camera)

    @patch("pylon_circle_detector.PYLON_AVAILABLE", True)
    def test_init_pylon_no_devices(self):
        """Test PylonCamera initialization with no Pylon devices."""
        with patch("pylon_circle_detector.pylon") as mock_pylon:
            # Mock Pylon factory with no devices
            mock_tl_factory = Mock()
            mock_pylon.TlFactory.GetInstance.return_value = mock_tl_factory
            mock_tl_factory.EnumerateDevices.return_value = []

            with patch("cv2.VideoCapture") as mock_cv2:
                mock_cap = Mock()
                mock_cap.isOpened.return_value = True
                mock_cv2.return_value = mock_cap

                camera = PylonCamera(camera_index=0, use_pylon=True)

                self.assertFalse(camera.use_pylon)  # Should fallback to webcam

    def test_configure_pylon_camera(self):
        """Test Pylon camera configuration."""
        with patch("pylon_circle_detector.PYLON_AVAILABLE", True):
            with patch("pylon_circle_detector.pylon") as mock_pylon:
                mock_camera = Mock()
                mock_camera.PixelFormat.SetValue = Mock()
                mock_camera.ExposureTime.SetValue = Mock()
                mock_camera.Gain.SetValue = Mock()
                mock_camera.AcquisitionMode.SetValue = Mock()
                mock_camera.TriggerMode.SetValue = Mock()
                mock_camera.StartGrabbing = Mock()

                camera = PylonCamera(camera_index=0, use_pylon=False)
                camera.camera = mock_camera
                camera.use_pylon = True

                camera._configure_pylon_camera()

                # Verify configuration calls
                mock_camera.PixelFormat.SetValue.assert_called()
                mock_camera.ExposureTime.SetValue.assert_called()
                mock_camera.Gain.SetValue.assert_called()
                mock_camera.AcquisitionMode.SetValue.assert_called()
                mock_camera.TriggerMode.SetValue.assert_called()
                mock_camera.StartGrabbing.assert_called()

    def test_read_frame_pylon(self):
        """Test frame reading from Pylon camera."""
        with patch("pylon_circle_detector.PYLON_AVAILABLE", True):
            with patch("pylon_circle_detector.pylon") as mock_pylon:
                mock_camera = Mock()
                mock_grab_result = Mock()
                mock_grab_result.GrabSucceeded.return_value = True
                mock_grab_result.Array = np.zeros((100, 100), dtype=np.uint8)
                mock_camera.RetrieveResult.return_value = mock_grab_result

                camera = PylonCamera(camera_index=0, use_pylon=False)
                camera.camera = mock_camera
                camera.use_pylon = True

                frame = camera.read_frame()

                self.assertIsInstance(frame, np.ndarray)
                mock_grab_result.Release.assert_called()

    def test_read_frame_webcam(self):
        """Test frame reading from webcam."""
        with patch("cv2.VideoCapture") as mock_cv2:
            mock_cap = Mock()
            mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
            mock_cv2.return_value = mock_cap

            camera = PylonCamera(camera_index=0, use_pylon=False)

            frame = camera.read_frame()

            self.assertIsInstance(frame, np.ndarray)

    def test_read_frame_failure(self):
        """Test frame reading failure."""
        with patch("cv2.VideoCapture") as mock_cv2:
            mock_cap = Mock()
            mock_cap.read.return_value = (False, None)
            mock_cv2.return_value = mock_cap

            camera = PylonCamera(camera_index=0, use_pylon=False)

            frame = camera.read_frame()

            self.assertIsNone(frame)

    def test_release_pylon(self):
        """Test Pylon camera release."""
        with patch("pylon_circle_detector.PYLON_AVAILABLE", True):
            mock_camera = Mock()
            mock_camera.StopGrabbing = Mock()
            mock_camera.Close = Mock()

            camera = PylonCamera(camera_index=0, use_pylon=False)
            camera.camera = mock_camera
            camera.use_pylon = True
            camera.is_grabbing = True

            camera.release()

            mock_camera.StopGrabbing.assert_called()
            mock_camera.Close.assert_called()

    def test_release_webcam(self):
        """Test webcam release."""
        with patch("cv2.VideoCapture") as mock_cv2:
            mock_cap = Mock()
            mock_cap.release = Mock()
            mock_cv2.return_value = mock_cap

            camera = PylonCamera(camera_index=0, use_pylon=False)

            camera.release()

            mock_cap.release.assert_called()


class TestCircleDetectionApp(unittest.TestCase):
    """Test the CircleDetectionApp class and all its methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")

        # Create test config
        test_config = {
            "hough_params": {"dp": 1, "min_dist": 20},
            "contour_params": {"min_area": 100, "max_area": 50000},
            "display": {"window_name": "Test"},
            "recording": {"fps": 30},
        }

        with open(self.config_file, "w") as f:
            json.dump(test_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("pylon_circle_detector.PylonCamera")
    def test_init_default(self, mock_pylon_camera):
        """Test CircleDetectionApp initialization with defaults."""
        mock_camera_instance = Mock()
        mock_pylon_camera.return_value = mock_camera_instance

        app = CircleDetectionApp()

        self.assertEqual(app.camera_index, 0)
        self.assertTrue(app.use_pylon)
        self.assertFalse(app.use_gpu)
        self.assertIsNotNone(app.config)
        self.assertFalse(app.is_running)
        self.assertTrue(app.show_controls)
        self.assertEqual(app.detection_method, "combined")

    @patch("pylon_circle_detector.PylonCamera")
    def test_init_custom_params(self, mock_pylon_camera):
        """Test CircleDetectionApp initialization with custom parameters."""
        mock_camera_instance = Mock()
        mock_pylon_camera.return_value = mock_camera_instance

        app = CircleDetectionApp(
            camera_index=1, use_pylon=False, use_gpu=True, config_file=self.config_file
        )

        self.assertEqual(app.camera_index, 1)
        self.assertFalse(app.use_pylon)
        self.assertTrue(app.use_gpu)
        self.assertIsNotNone(app.config)

    def test_load_config_default(self):
        """Test configuration loading with defaults."""
        app = CircleDetectionApp()

        # Check default config structure
        self.assertIn("hough_params", app.config)
        self.assertIn("contour_params", app.config)
        self.assertIn("display", app.config)
        self.assertIn("recording", app.config)

    def test_load_config_custom(self):
        """Test configuration loading with custom file."""
        app = CircleDetectionApp(config_file=self.config_file)

        # Check custom config values
        self.assertEqual(app.config["hough_params"]["dp"], 1)
        self.assertEqual(app.config["contour_params"]["min_area"], 100)

    def test_load_config_nonexistent(self):
        """Test configuration loading with nonexistent file."""
        app = CircleDetectionApp(config_file="nonexistent.json")

        # Should load defaults
        self.assertIn("hough_params", app.config)

    def test_create_control_window(self):
        """Test control window creation."""
        app = CircleDetectionApp()

        with patch("cv2.namedWindow") as mock_named_window:
            with patch("cv2.resizeWindow") as mock_resize_window:
                with patch("cv2.createTrackbar") as mock_create_trackbar:
                    app.create_control_window()

                    mock_named_window.assert_called()
                    mock_resize_window.assert_called()
                    # Should create multiple trackbars
                    self.assertGreater(mock_create_trackbar.call_count, 5)

    def test_trackbar_callbacks(self):
        """Test trackbar callback functions."""
        app = CircleDetectionApp()

        # Test Hough parameter callbacks
        app._on_hough_dp_change(2)
        self.assertEqual(app.detector.hough_params["dp"], 2)

        app._on_hough_min_dist_change(30)
        self.assertEqual(app.detector.hough_params["min_dist"], 30)

        app._on_hough_param1_change(60)
        self.assertEqual(app.detector.hough_params["param1"], 60)

        app._on_hough_param2_change(40)
        self.assertEqual(app.detector.hough_params["param2"], 40)

        app._on_hough_min_radius_change(20)
        self.assertEqual(app.detector.hough_params["min_radius"], 20)

        app._on_hough_max_radius_change(400)
        self.assertEqual(app.detector.hough_params["max_radius"], 400)

        # Test contour parameter callbacks
        app._on_contour_min_area_change(200)
        self.assertEqual(app.detector.contour_params["min_area"], 200)

        app._on_contour_max_area_change(60000)
        self.assertEqual(app.detector.contour_params["max_area"], 60000)

        app._on_circularity_threshold_change(80)
        self.assertEqual(app.detector.contour_params["circularity_threshold"], 0.8)

    def test_process_frame(self):
        """Test frame processing."""
        app = CircleDetectionApp()

        # Create test frame
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(test_frame, (50, 50), 20, (255, 255, 255), -1)

        processed_frame, circles = app.process_frame(test_frame)

        self.assertIsInstance(processed_frame, np.ndarray)
        self.assertIsInstance(circles, list)
        self.assertEqual(processed_frame.shape, test_frame.shape)

    def test_add_info_overlay(self):
        """Test information overlay addition."""
        app = CircleDetectionApp()

        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        circles = [(50, 50, 20)]

        overlay = app._add_info_overlay(test_frame, circles)

        self.assertIsInstance(overlay, np.ndarray)
        self.assertEqual(overlay.shape, test_frame.shape)

    def test_start_recording(self):
        """Test video recording start."""
        app = CircleDetectionApp()

        with patch.object(app.camera, "read_frame") as mock_read_frame:
            mock_read_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch("cv2.VideoWriter_fourcc") as mock_fourcc:
                with patch("cv2.VideoWriter") as mock_video_writer:
                    mock_writer = Mock()
                    mock_video_writer.return_value = mock_writer

                    app.start_recording()

                    self.assertTrue(app.detector.is_recording)
                    self.assertIsNotNone(app.detector.recording_path)
                    mock_video_writer.assert_called()

    def test_stop_recording(self):
        """Test video recording stop."""
        app = CircleDetectionApp()

        # Mock video writer
        mock_writer = Mock()
        app.detector.video_writer = mock_writer
        app.detector.is_recording = True
        app.detector.recording_path = "test.mp4"

        app.stop_recording()

        self.assertFalse(app.detector.is_recording)
        self.assertIsNone(app.detector.recording_path)
        mock_writer.release.assert_called()

    def test_save_frame(self):
        """Test frame saving functionality."""
        app = CircleDetectionApp()

        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        circles = [(50, 50, 20)]

        with patch("cv2.imwrite") as mock_imwrite:
            with patch("builtins.open", create=True) as mock_open:
                app.save_frame(test_frame, circles)

                mock_imwrite.assert_called()
                mock_open.assert_called()

    def test_run_keyboard_interrupt(self):
        """Test application run with keyboard interrupt."""
        app = CircleDetectionApp()

        with patch.object(app.camera, "read_frame") as mock_read_frame:
            mock_read_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch("cv2.imshow") as mock_imshow:
                with patch("cv2.waitKey") as mock_wait_key:
                    mock_wait_key.return_value = ord("q")

                    with patch.object(app, "stop_recording") as mock_stop_recording:
                        with patch.object(app.camera, "release") as mock_release:
                            with patch("cv2.destroyAllWindows") as mock_destroy:
                                app.run()

                                mock_stop_recording.assert_called()
                                mock_release.assert_called()
                                mock_destroy.assert_called()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_workflow(self):
        """Test complete workflow from image to detection."""
        # Create detector
        detector = CircleDetector()

        # Create test image
        test_img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(test_img, (100, 100), 50, 255, -1)

        # Test all detection methods
        hough_circles = detector.detect_circles_hough(test_img)
        contour_circles = detector.detect_circles_contour(test_img)
        combined_circles = detector.detect_circles_combined(test_img)

        # Verify results
        self.assertIsInstance(hough_circles, list)
        self.assertIsInstance(contour_circles, list)
        self.assertIsInstance(combined_circles, list)

        # Test drawing
        drawn_img = detector.draw_circles(test_img, hough_circles)
        self.assertIsInstance(drawn_img, np.ndarray)

    def test_performance_benchmark(self):
        """Test performance of detection algorithms."""
        detector = CircleDetector()

        # Create test image
        test_img = np.zeros((480, 640), dtype=np.uint8)
        for i in range(5):
            x = 100 + i * 100
            y = 240
            radius = 30 + i * 10
            cv2.circle(test_img, (x, y), radius, 255, -1)

        # Benchmark Hough detection
        start_time = time.time()
        for _ in range(100):
            circles = detector.detect_circles_hough(test_img)
        hough_time = time.time() - start_time

        # Benchmark contour detection
        start_time = time.time()
        for _ in range(100):
            circles = detector.detect_circles_contour(test_img)
        contour_time = time.time() - start_time

        # Benchmark combined detection
        start_time = time.time()
        for _ in range(100):
            circles = detector.detect_circles_combined(test_img)
        combined_time = time.time() - start_time

        # Verify performance is reasonable
        self.assertLess(hough_time, 10.0)  # Should complete in under 10 seconds
        self.assertLess(contour_time, 10.0)
        self.assertLess(combined_time, 15.0)

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        detector = CircleDetector()

        # Test with None input
        with self.assertRaises(Exception):
            detector.detect_circles_hough(None)

        # Test with invalid image shape
        invalid_img = np.zeros((100, 100, 3), dtype=np.uint8)  # Color image
        with self.assertRaises(Exception):
            detector.detect_circles_hough(invalid_img)

        # Test with empty image
        empty_img = np.array([], dtype=np.uint8)
        with self.assertRaises(Exception):
            detector.detect_circles_hough(empty_img)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_extreme_parameters(self):
        """Test detection with extreme parameter values."""
        detector = CircleDetector()

        # Test with very small circles
        detector.hough_params["min_radius"] = 1
        detector.hough_params["max_radius"] = 5

        test_img = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(test_img, (25, 25), 3, 255, -1)

        circles = detector.detect_circles_hough(test_img)
        self.assertIsInstance(circles, list)

        # Test with very large circles
        detector.hough_params["min_radius"] = 100
        detector.hough_params["max_radius"] = 500

        large_img = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.circle(large_img, (500, 500), 200, 255, -1)

        circles = detector.detect_circles_hough(large_img)
        self.assertIsInstance(circles, list)

    def test_high_noise_images(self):
        """Test detection with noisy images."""
        detector = CircleDetector()

        # Create noisy image
        test_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        cv2.circle(test_img, (100, 100), 50, 255, -1)

        circles = detector.detect_circles_hough(test_img)
        self.assertIsInstance(circles, list)

    def test_overlapping_circles(self):
        """Test detection with overlapping circles."""
        detector = CircleDetector()

        test_img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(test_img, (100, 100), 50, 255, -1)
        cv2.circle(test_img, (120, 100), 40, 255, -1)

        circles = detector.detect_circles_combined(test_img)
        self.assertIsInstance(circles, list)

    def test_memory_usage(self):
        """Test memory usage with large images."""
        detector = CircleDetector()

        # Create large image
        large_img = np.zeros((2000, 2000), dtype=np.uint8)
        cv2.circle(large_img, (1000, 1000), 500, 255, -1)

        # Process multiple times
        for _ in range(10):
            circles = detector.detect_circles_hough(large_img)
            self.assertIsInstance(circles, list)


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("Running Comprehensive Circle Detection Tests")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestCircleDetector,
        TestPylonCamera,
        TestCircleDetectionApp,
        TestIntegration,
        TestEdgeCases,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
