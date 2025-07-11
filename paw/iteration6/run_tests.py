#!/usr/bin/env python3
import sys
import unittest
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all test modules
from tests import (
    test_config,
    test_logger,
    test_datastore,
    test_cv_module,
    test_random_pixel,
    test_intensity_reader,
    test_pattern_recognizer,
    test_anomaly_detector,
    test_batch_processor,
    test_torch_module,
    test_realtime_processor,
    test_hpc
)

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    test_modules = [
        test_config,
        test_logger,
        test_datastore,
        test_cv_module,
        test_random_pixel,
        test_intensity_reader,
        test_pattern_recognizer,
        test_anomaly_detector,
        test_batch_processor,
        test_torch_module,
        test_realtime_processor,
        test_hpc
    ]
    
    for module in test_modules:
        suite.addTests(loader.loadTestsFromModule(module))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)