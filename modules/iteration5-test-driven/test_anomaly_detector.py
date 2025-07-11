#!/usr/bin/env python3
"""
Unit tests for anomaly_detector.py
"""
import unittest
from unittest.mock import patch
import anomaly_detector
import data_store

class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        data_store.clear_events()

    def tearDown(self):
        """Clean up after tests."""
        data_store.clear_events()

    def test_detect_anomalies_with_data(self):
        """Test detect_anomalies with normal and anomalous data."""
        # Save events: mean will be 100, threshold 50
        # Values 100, 100, 100 -> mean = 100
        # Value 200 -> deviation = 100 > 50 (anomaly)
        # Value 10 -> deviation = 90 > 50 (anomaly)
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 100})
        data_store.save_event({"intensity": 200})
        data_store.save_event({"intensity": 10})

        result = anomaly_detector.detect_anomalies(threshold=50)

        # Should detect 2 anomalies
        self.assertEqual(len(result), 2)

        # Check anomaly values
        anomaly_values = [e.get("pixel", e.get("intensity")) for e in result]
        self.assertIn(200, anomaly_values)
        self.assertIn(10, anomaly_values)

    def test_detect_anomalies_no_data(self):
        """Test detect_anomalies with no data."""
        result = anomaly_detector.detect_anomalies()

        # Should return empty list
        self.assertEqual(result, [])

    def test_detect_anomalies_no_anomalies(self):
        """Test detect_anomalies with no anomalous data."""
        # Save events with values close to mean
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 110})
        data_store.save_event({"pixel": 90})

        result = anomaly_detector.detect_anomalies(threshold=50)

        # Should find no anomalies
        self.assertEqual(len(result), 0)

    def test_detect_anomalies_custom_threshold(self):
        """Test detect_anomalies with custom threshold."""
        # Save events: mean will be 100
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 130})  # deviation = 30

        # With threshold 20, should detect anomaly
        result = anomaly_detector.detect_anomalies(threshold=20)
        self.assertEqual(len(result), 1)

        # With threshold 40, should not detect anomaly
        data_store.clear_events()
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 130})

        result = anomaly_detector.detect_anomalies(threshold=40)
        self.assertEqual(len(result), 0)

    @patch('anomaly_detector.print')
    def test_detect_anomalies_prints_results(self, mock_print):
        """Test that detect_anomalies prints anomalies."""
        # Save events with one anomaly
        data_store.save_event({"pixel": 100})
        data_store.save_event({"pixel": 200})

        anomaly_detector.detect_anomalies(threshold=50)

        # Check that anomaly was printed
        mock_print.assert_any_call("[Anomaly] {'pixel': 200, 'timestamp': unittest.mock.ANY}")

    @patch('anomaly_detector.print')
    def test_detect_anomalies_no_data_message(self, mock_print):
        """Test message when no data is found."""
        anomaly_detector.detect_anomalies()

        mock_print.assert_called_with("[Anomaly] No data")

if __name__ == "__main__":
    unittest.main()
