

import logging
import unittest
import importlib
import inspect
from typing import Dict, Any

logger = logging.getLogger("ZoneAnalysisConnector.Troubleshooter")

class Troubleshooter:
    """
    A class to discover and run tests from dynamically loaded modules.
    """

    def __init__(self, modules: Dict[str, Any]):
        """
        Initializes the Troubleshooter with the loaded modules.

        Args:
            modules: A dictionary of loaded modules.
        """
        self.modules = modules
        self.test_suite = unittest.TestSuite()
        self.results = None

    def discover_tests(self):
        """
        Discovers all test functions in the loaded modules.
        A function is considered a test if its name starts with 'test_'.
        """
        logger.info("Starting test discovery...")
        for module_name, module in self.modules.items():
            # Look for functions that follow the 'test_*' naming convention
            for name, func in inspect.getmembers(module, inspect.isfunction):
                if name.startswith('test_'):
                    logger.debug(f"Found test function: {name} in module {module_name}")
                    # Add the function to a new TestCase
                    test_case = unittest.FunctionTestCase(func)
                    self.test_suite.addTest(test_case)
            
            # Look for classes that are subclasses of unittest.TestCase
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, unittest.TestCase):
                    logger.debug(f"Found TestCase class: {name} in module {module_name}")
                    self.test_suite.addTest(unittest.makeSuite(cls))

        logger.info(f"Test discovery complete. Found {self.test_suite.countTestCases()} test cases.")
        return self.test_suite.countTestCases() > 0

    def run_tests(self) -> unittest.TestResult:
        """
        Runs the discovered test suite.
        """
        if self.test_suite.countTestCases() == 0:
            logger.warning("No tests were found to run.")
            return None

        logger.info(f"Running {self.test_suite.countTestCases()} tests...")
        runner = unittest.TextTestRunner(verbosity=2)
        self.results = runner.run(self.test_suite)
        
        logger.info("Test run finished.")
        return self.results

    def print_summary(self):
        """
        Prints a summary of the test results.
        """
        if not self.results:
            logger.error("No test results to summarize.")
            print("No tests were run.")
            return

        print("\n--- Test Summary ---")
        print(f"Tests Run: {self.results.testsRun}")
        print(f"Successes: {self.results.testsRun - len(self.results.failures) - len(self.results.errors)}")
        
        if self.results.failures:
            print(f"Failures: {len(self.results.failures)}")
            logger.warning(f"{len(self.results.failures)} tests failed.")
            for test, traceback in self.results.failures:
                logger.warning(f"  - FAILED: {test}\n{traceback}")
        
        if self.results.errors:
            print(f"Errors: {len(self.results.errors)}")
            logger.error(f"{len(self.results.errors)} tests encountered errors.")
            for test, traceback in self.results.errors:
                logger.error(f"  - ERROR: {test}\n{traceback}")
        
        if self.results.wasSuccessful():
            print("\nResult: ALL TESTS PASSED")
            logger.info("All tests passed successfully.")
        else:
            print("\nResult: SOME TESTS FAILED")
            logger.warning("Some tests failed or had errors.")

def run_troubleshooter(modules: Dict[str, Any]):
    """
    The main entry point for the troubleshooter.
    """
    print("\n" + "="*50)
    print("    Zone Analysis Troubleshooter")
    print("="*50)
    
    troubleshooter = Troubleshooter(modules)
    
    if not troubleshooter.discover_tests():
        print("No test functions (e.g., 'def test_...():') found in any of the modules.")
        logger.warning("Troubleshooter did not find any tests to run.")
        return
        
    results = troubleshooter.run_tests()
    
    if results:
        troubleshooter.print_summary()

if __name__ == '__main__':
    # This allows running the troubleshooter standalone for debugging
    # It requires a way to load the modules first.
    print("This script is intended to be run from the main connector.")
    # As a basic example, let's try to load a specific module and test it
    try:
        spec = importlib.util.spec_from_file_location("robust-mask-generator", "robust-mask-generator.py")
        rmg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rmg_module)
        
        logging.basicConfig(level=logging.INFO)
        run_troubleshooter({"robust-mask-generator": rmg_module})
        
    except FileNotFoundError:
        print("Could not find 'robust-mask-generator.py' to run a standalone test.")

