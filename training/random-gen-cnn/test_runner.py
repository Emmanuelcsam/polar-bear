#!/usr/bin/env python3
"""
Test runner for the image categorization system
Runs all unit tests and integration tests
"""
import sys
import os
import unittest
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    """Run all test modules"""
    # List of test modules
    test_modules = [
        'test_auto_installer',
        'test_pixel_sampler',
        'test_correlation_analyzer',
        'test_batch_processor',
        'test_self_reviewer',
        'test_integration'
    ]

    print("="*60)
    print("RUNNING IMAGE CATEGORIZATION SYSTEM TESTS")
    print("="*60)

    all_passed = True
    results = {}

    for module in test_modules:
        print(f"\n{'='*20} {module.upper()} {'='*20}")

        try:
            # Import and run the test module
            test_module = __import__(module)

            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)

            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)

            # Track results
            results[module] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success': result.wasSuccessful()
            }

            if not result.wasSuccessful():
                all_passed = False

        except Exception as e:
            print(f"Error running {module}: {e}")
            results[module] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success': False
            }
            all_passed = False

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_tests = 0
    total_failures = 0
    total_errors = 0

    for module, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"{module:25} | {status:4} | Tests: {result['tests_run']:3} | "
              f"Failures: {result['failures']:2} | Errors: {result['errors']:2}")

        total_tests += result['tests_run']
        total_failures += result['failures']
        total_errors += result['errors']

    print("-" * 60)
    print(f"{'TOTAL':25} | {'':4} | Tests: {total_tests:3} | "
          f"Failures: {total_failures:2} | Errors: {total_errors:2}")

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        return 1

def run_with_coverage():
    """Run tests with coverage reporting"""
    print("Running tests with coverage...")

    # Check if coverage is available
    try:
        import coverage
        print("Coverage module found. Running with coverage...")

        # Create coverage instance
        cov = coverage.Coverage()
        cov.start()

        # Run tests
        result = run_all_tests()

        # Stop coverage and save
        cov.stop()
        cov.save()

        print("\nCoverage Report:")
        print("-" * 40)
        cov.report()

        # Generate HTML report
        html_dir = 'htmlcov'
        cov.html_report(directory=html_dir)
        print(f"\nHTML coverage report generated in {html_dir}/")

        return result

    except ImportError:
        print("Coverage module not available. Running without coverage...")
        return run_all_tests()

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")

    required_modules = [
        'numpy',
        'PIL',
        'pickle',
        'json',
        'tempfile',
        'unittest'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - MISSING")
            missing_modules.append(module)

    if missing_modules:
        print(f"\nMissing dependencies: {', '.join(missing_modules)}")
        print("Please install missing dependencies before running tests.")
        return False

    print("All dependencies available!")
    return True

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--coverage':
            return run_with_coverage()
        elif sys.argv[1] == '--check-deps':
            return 0 if check_dependencies() else 1
        elif sys.argv[1] == '--help':
            print("Usage: python test_runner.py [--coverage] [--check-deps] [--help]")
            print("  --coverage   : Run tests with coverage reporting")
            print("  --check-deps : Check if all dependencies are available")
            print("  --help       : Show this help message")
            return 0

    # Check dependencies first
    if not check_dependencies():
        return 1

    # Run tests
    return run_all_tests()

if __name__ == '__main__':
    sys.exit(main())
