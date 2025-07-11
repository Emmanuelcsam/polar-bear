#!/usr/bin/env python3
"""
Unit tests for data_store.py
"""
import unittest
import json
import os
import tempfile
import time
from unittest.mock import patch
import data_store

class TestDataStore(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.test_log_file = "test_events.log"
        self.original_log_file = data_store.LOG_FILE
        data_store.LOG_FILE = self.test_log_file

    def tearDown(self):
        """Clean up after tests."""
        data_store.LOG_FILE = self.original_log_file
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)

    def test_save_event_adds_timestamp(self):
        """Test that save_event adds a timestamp to the event."""
        event = {"pixel": 123}
        data_store.save_event(event)

        # Check that timestamp was added
        self.assertIn("timestamp", event)
        self.assertIsInstance(event["timestamp"], float)

    def test_save_and_load_events(self):
        """Test saving and loading events."""
        events = [
            {"pixel": 123},
            {"intensity": 45},
            {"test": "data"}
        ]

        # Save events
        for event in events:
            data_store.save_event(event.copy())

        # Load events
        loaded_events = data_store.load_events()

        # Check that all events were loaded
        self.assertEqual(len(loaded_events), len(events))

        # Check that each event has original data plus timestamp
        for i, loaded_event in enumerate(loaded_events):
            for key, value in events[i].items():
                self.assertEqual(loaded_event[key], value)
            self.assertIn("timestamp", loaded_event)

    def test_load_events_empty_file(self):
        """Test loading events when file doesn't exist."""
        events = data_store.load_events()
        self.assertEqual(events, [])

    def test_clear_events(self):
        """Test clearing all events."""
        # Save some events
        data_store.save_event({"test": "data"})

        # Clear events
        data_store.clear_events()

        # Check that file is gone and load returns empty list
        self.assertFalse(os.path.exists(self.test_log_file))
        events = data_store.load_events()
        self.assertEqual(events, [])

if __name__ == "__main__":
    unittest.main()
