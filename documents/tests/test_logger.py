#!/usr/bin/env python3
"""
Unit tests for the fiber optics logger module
Tests verbose logging, run/cycle tracking, and all logging functions
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from fiber_logger import FiberOpticsLogger, get_logger, log_function, track_process
from test_base import FiberOpticsTestCase


class TestFiberOpticsLogger(FiberOpticsTestCase):
    """Test the enhanced logger functionality"""
    
    def setUp(self):
        """Set up test logger with temporary directory"""
        super().setUp()
        self.logger = get_logger("TestLogger")
    
    def test_logger_initialization(self):
        """Test logger is properly initialized"""
        self.assertIsInstance(self.logger, FiberOpticsLogger)
        self.assertTrue(hasattr(self.logger, 'run_id'))
        self.assertTrue(hasattr(self.logger, 'run_logger'))
        self.assertEqual(len(self.logger.run_id), 8)
    
    def test_run_logger_creation(self):
        """Test run logger creates files correctly"""
        run_logger = self.logger.run_logger
        self.assertTrue(run_logger.log_file.exists())
        self.assertTrue(run_logger.metadata_file.exists())
        
        # Check metadata structure
        with open(run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertIn('run_id', metadata)
        self.assertIn('start_time', metadata)
        self.assertIn('cycles', metadata)
        self.assertIn('processes', metadata)
        self.assertIn('errors', metadata)
        self.assertIn('warnings', metadata)
        self.assertIn('performance_metrics', metadata)
    
    def test_cycle_management(self):
        """Test cycle start and end functionality"""
        # Start cycle
        cycle_num = self.logger.start_cycle("Test Cycle")
        self.assertEqual(cycle_num, 1)
        
        # End cycle with metrics
        metrics = {
            'accuracy': 0.95,
            'loss': 0.05,
            'processing_time': 10.5
        }
        self.logger.end_cycle(cycle_num, metrics)
        
        # Check metadata was updated
        with open(self.logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(len(metadata['cycles']), 1)
        cycle = metadata['cycles'][0]
        self.assertEqual(cycle['cycle_number'], 1)
        self.assertIn('start_time', cycle)
        self.assertIn('end_time', cycle)
        self.assertEqual(cycle['metrics'], metrics)
    
    def test_process_tracking(self):
        """Test process tracking with context manager"""
        with track_process("Test Process", param1="value1", param2=42):
            self.logger.info("Inside process")
        
        # Check process was logged
        with open(self.logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertGreater(len(metadata['processes']), 0)
        process = metadata['processes'][-1]
        self.assertEqual(process['name'], "Test Process")
        self.assertEqual(process['details']['param1'], "value1")
        self.assertEqual(process['details']['param2'], 42)
    
    def test_nested_process_tracking(self):
        """Test nested process tracking"""
        with track_process("Outer Process"):
            self.logger.info("In outer process")
            
            with track_process("Inner Process"):
                self.logger.info("In inner process")
                self.assertEqual(len(self.logger.process_stack), 2)
            
            self.assertEqual(len(self.logger.process_stack), 1)
        
        self.assertEqual(len(self.logger.process_stack), 0)
    
    def test_function_logging_decorator(self):
        """Test function logging decorator"""
        @log_function("TestLogger")
        def test_func(x, y=10):
            return x + y
        
        result = test_func(5, y=20)
        self.assertEqual(result, 25)
        
        # Function calls should be logged
        # Check by looking at function call depth changes
        self.assertEqual(self.logger.function_call_depth, 0)
    
    def test_function_error_logging(self):
        """Test function error logging"""
        @log_function("TestLogger")
        def error_func():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            error_func()
        
        # Error should be logged
        with open(self.logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertGreater(len(metadata['errors']), 0)
        error = metadata['errors'][-1]
        self.assertIn("Test error", error['message'])
        self.assertIn('traceback', error)
    
    def test_tensor_info_logging(self):
        """Test tensor information logging"""
        # Create mock tensor
        mock_tensor = self.create_mock_image(size=(128, 128))
        
        # Log tensor info
        self.logger.log_tensor_info("test_tensor", mock_tensor, detailed=True)
        
        # Should not raise any errors
        self.assertTrue(True)
    
    def test_model_info_logging(self):
        """Test model information logging"""
        # Test with number of parameters
        self.logger.log_model_info("TestModel", num_params=1000000)
        
        # Test with mock model
        class MockModel:
            def parameters(self):
                # Return mock parameters
                class MockParam:
                    def numel(self):
                        return 1000
                    requires_grad = True
                
                return [MockParam() for _ in range(10)]
            
            def named_modules(self):
                return [
                    ('layer1', type('Conv2d', (), {})()),
                    ('layer2', type('ReLU', (), {})()),
                    ('layer3', type('Linear', (), {})())
                ]
        
        self.logger.log_model_info("MockModel", MockModel())
        
        # Should log without errors
        self.assertTrue(True)
    
    def test_training_progress_logging(self):
        """Test training progress logging"""
        # Test epoch logging
        self.logger.log_epoch_start(1, 10)
        
        # Test batch progress
        for i in range(5):
            self.logger.log_batch_progress(
                i, 5, 0.5 - i * 0.1,
                accuracy=0.8 + i * 0.04,
                precision=0.85 + i * 0.02
            )
        
        # Test epoch end
        self.logger.log_epoch_end(1, {
            'train_loss': 0.25,
            'val_loss': 0.30,
            'accuracy': 0.92
        })
        
        # Check process was logged
        with open(self.logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        epoch_processes = [p for p in metadata['processes'] if 'epoch_1' in p['name']]
        self.assertGreater(len(epoch_processes), 0)
    
    def test_similarity_check_logging(self):
        """Test similarity check logging"""
        # Test passing similarity
        self.logger.log_similarity_check(0.85, 0.7, "ref_001", {
            'structural': 0.9,
            'perceptual': 0.8,
            'pixel': 0.85
        })
        
        # Test failing similarity
        self.logger.log_similarity_check(0.65, 0.7, "ref_002")
        
        # Check warnings were logged
        with open(self.logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        warnings = metadata['warnings']
        low_sim_warnings = [w for w in warnings if 'Low similarity' in w['message']]
        self.assertGreater(len(low_sim_warnings), 0)
    
    def test_anomaly_detection_logging(self):
        """Test anomaly detection logging"""
        # Test with anomalies
        self.logger.log_anomaly_detection(
            3, 
            [(100, 150), (200, 250), (50, 75)],
            {
                'max_severity': 0.8,
                'avg_severity': 0.5,
                'types': ['scratch', 'contamination']
            }
        )
        
        # Test without anomalies
        self.logger.log_anomaly_detection(0, [])
        
        # Check warnings
        with open(self.logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        anomaly_warnings = [w for w in metadata['warnings'] if 'anomalies' in w['message']]
        self.assertGreater(len(anomaly_warnings), 0)
    
    def test_region_classification_logging(self):
        """Test region classification logging"""
        self.logger.log_region_classification(
            {'core': 0.8, 'cladding': 0.15, 'ferrule': 0.05},
            {'entropy': 0.2, 'margin': 0.65}
        )
        
        # Should log without errors
        self.assertTrue(True)
    
    def test_performance_metrics_logging(self):
        """Test performance metrics logging"""
        metrics = {
            'inference_time': 0.025,
            'memory_usage': 1024.5,
            'accuracy': 0.95,
            'throughput': 40
        }
        
        self.logger.log_performance_metrics(metrics)
        
        # Check metrics were saved
        with open(self.logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata['performance_metrics'], metrics)
    
    def test_memory_usage_logging(self):
        """Test memory usage logging"""
        # Should not raise errors even without psutil
        self.logger.log_memory_usage()
        self.assertTrue(True)
    
    def test_script_transition_logging(self):
        """Test script transition logging"""
        self.logger.log_script_transition(
            "test_logger.py",
            "test_processor.py",
            {
                'data': 'Tensor',
                'config': 'Dict',
                'results': 'List'
            }
        )
        
        # Should log without errors
        self.assertTrue(True)
    
    def test_error_logging(self):
        """Test error logging with different severity levels"""
        # Test normal error
        try:
            raise ValueError("Test error")
        except Exception as e:
            self.logger.log_error("Normal error occurred", e, critical=False)
        
        # Test critical error
        try:
            raise RuntimeError("Critical error")
        except Exception as e:
            self.logger.log_error("Critical error occurred", e, critical=True)
        
        # Check errors were logged
        with open(self.logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertGreaterEqual(len(metadata['errors']), 2)
    
    def test_run_finalization(self):
        """Test run finalization"""
        self.logger.finalize_run()
        
        # Check metadata was finalized
        with open(self.logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertIn('end_time', metadata)
        self.assertIn('duration_seconds', metadata)
        self.assertGreater(metadata['duration_seconds'], 0)
    
    def test_multiple_logger_instances(self):
        """Test multiple logger instances are cached correctly"""
        logger1 = get_logger("Logger1")
        logger2 = get_logger("Logger2")
        logger1_again = get_logger("Logger1")
        
        self.assertIs(logger1, logger1_again)
        self.assertIsNot(logger1, logger2)
        self.assertNotEqual(logger1.run_id, logger2.run_id)


class TestLoggerIntegration(FiberOpticsTestCase):
    """Test logger integration with other components"""
    
    def test_logger_with_mock_training(self):
        """Test logger during mock training scenario"""
        logger = get_logger("TrainingTest")
        
        # Start training cycle
        cycle = logger.start_cycle("Training")
        
        with track_process("Training Process", epochs=3, batch_size=16):
            for epoch in range(1, 4):
                logger.log_epoch_start(epoch, 3)
                
                # Simulate batches
                for batch in range(10):
                    loss = 1.0 - (epoch * 0.2) - (batch * 0.01)
                    logger.log_batch_progress(
                        batch, 10, loss,
                        accuracy=0.7 + (epoch * 0.05) + (batch * 0.01)
                    )
                
                # End epoch
                logger.log_epoch_end(epoch, {
                    'train_loss': 1.0 - (epoch * 0.2),
                    'val_loss': 1.1 - (epoch * 0.2),
                    'accuracy': 0.7 + (epoch * 0.1)
                })
        
        # End cycle
        logger.end_cycle(cycle, {
            'final_accuracy': 0.95,
            'total_time': 150.5
        })
        
        # Finalize
        logger.finalize_run()
        
        # Verify complete run was logged
        with open(logger.run_logger.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(len(metadata['cycles']), 1)
        self.assertIn('final_accuracy', metadata['cycles'][0]['metrics'])


if __name__ == "__main__":
    unittest.main()