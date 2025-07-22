#!/usr/bin/env python3
"""
Test suite for FiberOpticsLogger from fiber_logger.py
"""

import pytest
import logging
from pathlib import Path
import sys
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, mock_open

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiber_logger import FiberOpticsLogger, get_logger


class TestFiberOpticsLogger:
    """Test cases for FiberOpticsLogger class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        mock_cfg = Mock()
        mock_cfg.LOGS_PATH = Path('/tmp/logs')
        return mock_cfg
    
    @pytest.fixture
    @patch('fiber_logger.get_config')
    @patch('logging.getLogger')
    def logger(self, mock_get_logger, mock_get_config, mock_config):
        """Create logger instance with mocks"""
        mock_get_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = FiberOpticsLogger("TestLogger", config=mock_config)
        logger.logger = mock_logger
        return logger
    
    def test_initialization(self, logger, mock_config):
        """Test logger initialization"""
        assert logger is not None
        assert logger.name == "TestLogger"
        assert logger.config == mock_config
        assert hasattr(logger, 'process_times')
        assert hasattr(logger, 'statistics')
    
    @patch('fiber_logger.get_config')
    @patch('logging.FileHandler')
    @patch('logging.StreamHandler')
    def test_logger_setup(self, mock_stream_handler, mock_file_handler, mock_get_config, mock_config):
        """Test logger setup with handlers"""
        mock_get_config.return_value = mock_config
        
        logger = FiberOpticsLogger("TestLogger")
        
        # Check handlers were created
        mock_file_handler.assert_called()
        mock_stream_handler.assert_called()
    
    def test_log_class_init(self, logger):
        """Test class initialization logging"""
        logger.log_class_init("TestClass")
        
        logger.logger.info.assert_called_with("[CLASS_INIT] TestClass initialized")
        assert "TestClass" in logger.statistics['classes_initialized']
    
    def test_log_process_start(self, logger):
        """Test process start logging"""
        with patch('time.time', return_value=1000):
            logger.log_process_start("TestProcess")
        
        logger.logger.info.assert_called_with("[PROCESS_START] TestProcess")
        assert "TestProcess" in logger.process_times
        assert logger.process_times["TestProcess"] == 1000
    
    def test_log_process_end(self, logger):
        """Test process end logging"""
        # Start process first
        with patch('time.time', return_value=1000):
            logger.log_process_start("TestProcess")
        
        # End process
        with patch('time.time', return_value=1005):
            logger.log_process_end("TestProcess")
        
        expected_call = "[PROCESS_END] TestProcess (Duration: 5.00s)"
        logger.logger.info.assert_any_call(expected_call)
        assert "TestProcess" not in logger.process_times
        assert logger.statistics['total_processes'] == 1
        assert logger.statistics['total_process_time'] == 5.0
    
    def test_log_tensor_operation(self, logger):
        """Test tensor operation logging"""
        logger.log_tensor_operation("resize", (224, 224), (112, 112))
        
        expected_call = "[TENSOR_OP] resize: (224, 224) -> (112, 112)"
        logger.logger.debug.assert_called_with(expected_call)
        assert logger.statistics['tensor_operations'] == 1
    
    def test_log_feature_extraction(self, logger):
        """Test feature extraction logging"""
        logger.log_feature_extraction("edges", 1024)
        
        expected_call = "[FEATURE] Extracted edges features: 1024 dimensions"
        logger.logger.info.assert_called_with(expected_call)
        assert logger.statistics['features_extracted'] == 1
    
    def test_log_similarity_score(self, logger):
        """Test similarity score logging"""
        logger.log_similarity_score("test_image.jpg", 0.95)
        
        expected_call = "[SIMILARITY] test_image.jpg: 0.9500"
        logger.logger.info.assert_called_with(expected_call)
        assert logger.statistics['images_compared'] == 1
        assert logger.statistics['total_similarity'] == 0.95
    
    def test_log_anomaly_detection(self, logger):
        """Test anomaly detection logging"""
        logger.log_anomaly_detection("test_image.jpg", 0.15, False)
        
        expected_call = "[ANOMALY] test_image.jpg: Score=0.1500, Anomaly=False"
        logger.logger.info.assert_called_with(expected_call)
        assert logger.statistics['anomalies_detected'] == 0
        
        # Test with anomaly detected
        logger.log_anomaly_detection("test_image2.jpg", 0.85, True)
        assert logger.statistics['anomalies_detected'] == 1
    
    def test_log_training_epoch(self, logger):
        """Test training epoch logging"""
        metrics = {
            'loss': 0.5,
            'accuracy': 0.92,
            'val_loss': 0.6,
            'val_accuracy': 0.90
        }
        
        logger.log_training_epoch(5, metrics)
        
        expected_call = "[TRAINING] Epoch 5: loss=0.5000, accuracy=0.9200, val_loss=0.6000, val_accuracy=0.9000"
        logger.logger.info.assert_called_with(expected_call)
        assert logger.statistics['training_epochs'] == 1
    
    def test_log_model_checkpoint(self, logger):
        """Test model checkpoint logging"""
        logger.log_model_checkpoint("best_model.pth", 0.95)
        
        expected_call = "[CHECKPOINT] Saved model to best_model.pth (metric: 0.9500)"
        logger.logger.info.assert_called_with(expected_call)
        assert logger.statistics['checkpoints_saved'] == 1
    
    def test_log_batch_summary(self, logger):
        """Test batch summary logging"""
        results = [
            {'similarity': 0.9, 'anomaly': False},
            {'similarity': 0.8, 'anomaly': True},
            {'similarity': 0.95, 'anomaly': False}
        ]
        
        logger.log_batch_summary(results)
        
        logger.logger.info.assert_any_call("[BATCH_SUMMARY] Processed 3 images")
        logger.logger.info.assert_any_call("  Average similarity: 0.8833")
        logger.logger.info.assert_any_call("  Anomalies detected: 1/3 (33.33%)")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_export_statistics(self, mock_json_dump, mock_file, logger):
        """Test statistics export"""
        # Add some statistics
        logger.statistics['images_processed'] = 100
        logger.statistics['total_similarity'] = 85.5
        
        logger.export_statistics("stats.json")
        
        mock_file.assert_called_with("stats.json", 'w')
        mock_json_dump.assert_called()
        
        # Check exported data
        exported_data = mock_json_dump.call_args[0][0]
        assert 'timestamp' in exported_data
        assert exported_data['statistics']['images_processed'] == 100
    
    def test_get_statistics_summary(self, logger):
        """Test statistics summary generation"""
        logger.statistics['images_processed'] = 50
        logger.statistics['total_similarity'] = 45.0
        logger.statistics['anomalies_detected'] = 5
        
        summary = logger.get_statistics_summary()
        
        assert summary['images_processed'] == 50
        assert summary['average_similarity'] == 0.9
        assert summary['anomaly_rate'] == 0.1
    
    def test_logging_levels(self, logger):
        """Test different logging levels"""
        logger.debug("Debug message")
        logger.logger.debug.assert_called_with("Debug message")
        
        logger.info("Info message")
        logger.logger.info.assert_called_with("Info message")
        
        logger.warning("Warning message")
        logger.logger.warning.assert_called_with("Warning message")
        
        logger.error("Error message")
        logger.logger.error.assert_called_with("Error message")
        
        logger.critical("Critical message")
        logger.logger.critical.assert_called_with("Critical message")


class TestGetLoggerFunction:
    """Test cases for get_logger singleton function"""
    
    @patch('fiber_logger.FiberOpticsLogger')
    def test_singleton_pattern(self, mock_logger_class):
        """Test that get_logger returns same instance for same name"""
        mock_instance = Mock()
        mock_logger_class.return_value = mock_instance
        
        logger1 = get_logger("TestLogger")
        logger2 = get_logger("TestLogger")
        
        assert logger1 is logger2
        mock_logger_class.assert_called_once_with("TestLogger")
    
    @patch('fiber_logger.FiberOpticsLogger')
    def test_different_loggers(self, mock_logger_class):
        """Test that different names create different loggers"""
        mock_instance1 = Mock()
        mock_instance2 = Mock()
        mock_logger_class.side_effect = [mock_instance1, mock_instance2]
        
        logger1 = get_logger("Logger1")
        logger2 = get_logger("Logger2")
        
        assert logger1 is not logger2
        assert mock_logger_class.call_count == 2


class TestLoggerIntegration:
    """Integration tests for logger usage"""
    
    @pytest.fixture
    def temp_log_dir(self, tmp_path):
        """Create temporary log directory"""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        return log_dir
    
    @patch('fiber_logger.get_config')
    def test_full_logging_workflow(self, mock_get_config, temp_log_dir):
        """Test complete logging workflow"""
        # Setup mock config
        mock_config = Mock()
        mock_config.LOGS_PATH = temp_log_dir
        mock_get_config.return_value = mock_config
        
        # Create logger
        logger = FiberOpticsLogger("WorkflowTest")
        
        # Simulate workflow
        logger.log_class_init("TestClass")
        logger.log_process_start("ImageProcessing")
        logger.log_tensor_operation("load", None, (224, 224))
        logger.log_feature_extraction("edges", 512)
        logger.log_similarity_score("test.jpg", 0.92)
        logger.log_anomaly_detection("test.jpg", 0.05, False)
        logger.log_process_end("ImageProcessing")
        
        # Check statistics
        stats = logger.get_statistics_summary()
        assert stats['images_processed'] == 1
        assert stats['average_similarity'] == 0.92
        assert stats['anomaly_rate'] == 0.0
        
        # Check log file exists
        log_files = list(temp_log_dir.glob("*.log"))
        assert len(log_files) > 0
    
    def test_concurrent_logging(self):
        """Test thread-safe logging"""
        import threading
        
        logger = get_logger("ConcurrentTest")
        errors = []
        
        def log_worker(worker_id):
            try:
                for i in range(10):
                    logger.log_process_start(f"Process_{worker_id}_{i}")
                    logger.log_process_end(f"Process_{worker_id}_{i}")
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=log_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0
        assert logger.statistics['total_processes'] == 50
    
    @patch('fiber_logger.get_config')
    def test_error_recovery(self, mock_get_config):
        """Test logger error recovery"""
        mock_config = Mock()
        mock_config.LOGS_PATH = Path("/invalid/path/that/doesnt/exist")
        mock_get_config.return_value = mock_config
        
        # Logger should handle invalid path gracefully
        logger = FiberOpticsLogger("ErrorTest")
        logger.info("Test message")  # Should not raise exception
        
        # Test with None config
        logger2 = FiberOpticsLogger("ErrorTest2", config=None)
        logger2.info("Test message")  # Should not raise exception


if __name__ == '__main__':
    pytest.main([__file__, '-v'])