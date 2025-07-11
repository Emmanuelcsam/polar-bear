#!/usr/bin/env python3
"""
Unit tests for realtime_processor.py
"""
import unittest
import threading
import time
from unittest.mock import patch, MagicMock
import realtime_processor
import data_store

class TestRealtimeProcessor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()

    @patch('realtime_processor.time.sleep')
    @patch('realtime_processor.generate_pixels')
    def test_run_real_time_basic(self, mock_generate_pixels, mock_sleep):
        """Test basic run_real_time functionality."""
        realtime_processor.run_real_time(duration=2)

        # Check that generate_pixels was called with correct parameters
        mock_generate_pixels.assert_called_once_with(interval=0.1, max_iterations=20)

        # Check that sleep was called with correct duration
        mock_sleep.assert_called_once_with(2)

    @patch('realtime_processor.time.sleep')
    @patch('realtime_processor.generate_pixels')
    def test_run_real_time_custom_duration(self, mock_generate_pixels, mock_sleep):
        """Test run_real_time with custom duration."""
        realtime_processor.run_real_time(duration=5)

        # Check that generate_pixels was called with correct max_iterations
        mock_generate_pixels.assert_called_once_with(interval=0.1, max_iterations=50)

        # Check that sleep was called with correct duration
        mock_sleep.assert_called_once_with(5)

    @patch('realtime_processor.print')
    @patch('realtime_processor.time.sleep')
    @patch('realtime_processor.generate_pixels')
    def test_run_real_time_prints_messages(self, mock_generate_pixels, mock_sleep, mock_print):
        """Test that run_real_time prints start and end messages."""
        realtime_processor.run_real_time(duration=3)

        # Check print calls
        mock_print.assert_any_call("[Realtime] Running for 3s")
        mock_print.assert_any_call("[Realtime] Done")

    @patch('realtime_processor.time.sleep')
    def test_run_real_time_thread_creation(self, mock_sleep):
        """Test that run_real_time creates a daemon thread."""
        # Mock the threading.Thread class
        with patch('realtime_processor.threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            realtime_processor.run_real_time(duration=1)

            # Check that thread was created and configured
            mock_thread.assert_called_once()
            self.assertTrue(mock_thread_instance.daemon)
            mock_thread_instance.start.assert_called_once()

    def test_run_real_time_integration_short(self):
        """Test run_real_time with very short duration for integration."""
        # This test actually runs the function briefly to ensure it works
        start_time = time.time()
        realtime_processor.run_real_time(duration=0.5)
        end_time = time.time()

        # Check that it took approximately the right amount of time
        elapsed = end_time - start_time
        self.assertGreaterEqual(elapsed, 0.4)  # Allow some tolerance
        self.assertLessEqual(elapsed, 0.7)

        # Check that some events were generated
        events = data_store.load_events()
        self.assertGreater(len(events), 0)

if __name__ == "__main__":
    unittest.main()
