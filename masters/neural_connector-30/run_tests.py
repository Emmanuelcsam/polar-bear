

import unittest
import logging
from pathlib import Path

# Configure logging for the test runner
from core.logging_config import setup_logging
setup_logging(log_level="WARNING") # Set to WARNING to keep test output clean
logger = logging.getLogger("neural_connector.test_runner")

def run_all_tests():
    """
    Discovers and runs all unit tests for the Neural Connector system.
    """
    print("\n--- Running Neural Connector Test Suite ---")
    logger.info("Starting test discovery...")

    # Define the directory where tests are located
    test_dir = Path(__file__).parent / "tests"
    
    if not test_dir.exists():
        print(f"Test directory not found at: {test_dir}")
        logger.error("Test directory not found: %s", test_dir)
        return

    # Use unittest's default test discovery
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_dir), pattern="test_*.py")

    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests
    logger.info("Executing test suite...")
    result = runner.run(suite)
    
    print("\n--- Test Suite Complete ---")
    if result.wasSuccessful():
        print(f"Result: SUCCESS ({result.testsRun} tests run)")
        logger.info("Test suite finished successfully.")
    else:
        print(f"Result: FAILED")
        print(f"  - Tests Run: {result.testsRun}")
        print(f"  - Failures: {len(result.failures)}")
        print(f"  - Errors: {len(result.errors)}")
        logger.error("Test suite finished with failures or errors.")

if __name__ == "__main__":
    run_all_tests()

