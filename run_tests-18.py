#!/usr/bin/env python3
"""
Test runner for all modules.
Runs all unit tests and reports results.
"""
import unittest
import sys
import os

def run_all_tests():
    """Run all unit tests in the current directory."""
    # Get all test files
    test_files = [f[:-3] for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests from each test file
    for test_file in test_files:
        try:
            module = __import__(test_file)
            suite.addTests(loader.loadTestsFromModule(module))
            print(f"Added tests from {test_file}")
        except Exception as e:
            print(f"Failed to load tests from {test_file}: {e}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
