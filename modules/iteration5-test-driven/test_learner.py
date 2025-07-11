#!/usr/bin/env python3
"""
Unit tests for learner.py
"""
import unittest
import os
import tempfile
from unittest.mock import patch
import learner
import data_store

class TestLearner(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()
        self.test_model_file = "test_model.pkl"

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()
        if os.path.exists(self.test_model_file):
            os.remove(self.test_model_file)

    def test_learn_model_with_data(self):
        """Test learn_model with intensity data."""
        # Save some intensity events
        intensities = [100, 200, 100, 150, 100]
        for intensity in intensities:
            data_store.save_event({"intensity": intensity})

        result = learner.learn_model(self.test_model_file)

        # Check that model was created
        expected = {100: 3, 200: 1, 150: 1}
        self.assertEqual(result, expected)

        # Check that file was saved
        self.assertTrue(os.path.exists(self.test_model_file))

        # Check that model can be loaded
        loaded_model = learner.load_model(self.test_model_file)
        self.assertEqual(loaded_model, expected)

    def test_learn_model_no_data(self):
        """Test learn_model with no intensity data."""
        result = learner.learn_model(self.test_model_file)

        # Should return None
        self.assertIsNone(result)

        # Should not create file
        self.assertFalse(os.path.exists(self.test_model_file))

    def test_learn_model_mixed_events(self):
        """Test learn_model with mixed event types."""
        # Save mixed events
        data_store.save_event({"pixel": 255})
        data_store.save_event({"intensity": 100})
        data_store.save_event({"intensity": 200})
        data_store.save_event({"other": "data"})

        result = learner.learn_model(self.test_model_file)

        # Should only use intensity events
        expected = {100: 1, 200: 1}
        self.assertEqual(result, expected)

    def test_load_model_nonexistent(self):
        """Test load_model with non-existent file."""
        result = learner.load_model("nonexistent.pkl")

        # Should return None
        self.assertIsNone(result)

    @patch('learner.print')
    def test_learn_model_prints_success(self, mock_print):
        """Test that learn_model prints success message."""
        # Save some intensity data
        data_store.save_event({"intensity": 100})

        learner.learn_model(self.test_model_file)

        mock_print.assert_called_with(f"[Learn] Model saved to {self.test_model_file}")

    @patch('learner.print')
    def test_learn_model_no_data_message(self, mock_print):
        """Test message when no data is found."""
        learner.learn_model(self.test_model_file)

        mock_print.assert_called_with("[Learn] No intensity data found")

    @patch('learner.print')
    def test_load_model_not_found_message(self, mock_print):
        """Test message when model file is not found."""
        learner.load_model("nonexistent.pkl")

        mock_print.assert_called_with("[Learn] Model file nonexistent.pkl not found")

if __name__ == "__main__":
    unittest.main()
