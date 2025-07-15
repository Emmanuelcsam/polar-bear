
"""
test_runner.py
--------------

This script is the main entry point for the project's testing framework.

Functionality:
1.  Discovers all test files (named 'test_*.py') within the 'tests' directory.
2.  Loads and runs all the tests using Python's unittest module.
3.  Provides a summary of the test results.
"""

import unittest
import logging
from pathlib import Path

# Set up a basic logger for the test runner
logger = logging.getLogger("TestRunner")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_all_tests():
    """
    Discovers and runs all tests in the 'tests' directory.
    """
    logger.info("Starting test discovery...")
    
    # Start discovery in the 'tests' directory
    test_dir = str(Path(__file__).parent)
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern="test_*.py")
    
    if suite.countTestCases() == 0:
        logger.warning("No tests found. The testing framework is set up, but no test cases were discovered.")
        print("No tests to run.")
        return

    logger.info(f"Found {suite.countTestCases()} test cases.")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        logger.info("All tests passed successfully!")
    else:
        logger.error("Some tests failed.")
        
    return result

if __name__ == "__main__":
    print("--- Running Comprehensive Test Suite ---")
    run_all_tests()
    print("--- Test Suite Finished ---")
