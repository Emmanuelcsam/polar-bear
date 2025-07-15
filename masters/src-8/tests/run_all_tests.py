#!/usr/bin/env python3
"""
Master test runner for all geometry detection system tests
Runs all test suites and provides a comprehensive report
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all test modules
test_modules = [
    'test_integrated_geometry_system',
    'test_performance_benchmark_tool',
    'test_realtime_calibration_tool',
    'test_example_application',
    'test_setup_installer',
    'test_uv_compatible_setup',
    'test_python313_fix'
]

def run_all_tests():
    """Run all test suites and generate report"""
    print("=" * 80)
    print("GEOMETRY DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Track results
    results_summary = {}
    failed_tests = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    # Run each test module
    for module_name in test_modules:
        print(f"\n{'=' * 60}")
        print(f"Running tests from: {module_name}")
        print('=' * 60)
        
        try:
            # Import and load tests
            module = __import__(module_name)
            module_suite = loader.loadTestsFromModule(module)
            
            # Count tests
            test_count = module_suite.countTestCases()
            print(f"Found {test_count} tests")
            
            # Run tests with custom result
            stream = StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=2)
            result = runner.run(module_suite)
            
            # Print output
            output = stream.getvalue()
            print(output)
            
            # Store results
            results_summary[module_name] = {
                'tests': test_count,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'success': result.wasSuccessful()
            }
            
            # Update totals
            total_tests += test_count
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_skipped += len(result.skipped)
            
            # Collect failed tests
            for test, traceback in result.failures:
                failed_tests.append({
                    'module': module_name,
                    'test': str(test),
                    'type': 'FAILURE',
                    'traceback': traceback
                })
                
            for test, traceback in result.errors:
                failed_tests.append({
                    'module': module_name,
                    'test': str(test),
                    'type': 'ERROR',
                    'traceback': traceback
                })
                
        except ImportError as e:
            print(f"ERROR: Could not import {module_name}: {e}")
            results_summary[module_name] = {
                'tests': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False,
                'import_error': str(e)
            }
            total_errors += 1
        except Exception as e:
            print(f"ERROR: Unexpected error in {module_name}: {e}")
            results_summary[module_name] = {
                'tests': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False,
                'error': str(e)
            }
            total_errors += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal Tests Run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Skipped: {total_skipped}")
    print(f"Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%" if total_tests > 0 else "N/A")
    
    # Module breakdown
    print("\n" + "-" * 60)
    print("MODULE BREAKDOWN:")
    print("-" * 60)
    
    for module, results in results_summary.items():
        status = "✓ PASS" if results['success'] else "✗ FAIL"
        print(f"\n{module}:")
        print(f"  Status: {status}")
        print(f"  Tests: {results['tests']}")
        if results['failures'] > 0:
            print(f"  Failures: {results['failures']}")
        if results['errors'] > 0:
            print(f"  Errors: {results['errors']}")
        if results['skipped'] > 0:
            print(f"  Skipped: {results['skipped']}")
        if 'import_error' in results:
            print(f"  Import Error: {results['import_error']}")
    
    # Failed tests details
    if failed_tests:
        print("\n" + "=" * 80)
        print("FAILED TESTS DETAILS")
        print("=" * 80)
        
        for i, failure in enumerate(failed_tests, 1):
            print(f"\n{i}. {failure['type']}: {failure['test']}")
            print(f"   Module: {failure['module']}")
            print("   Traceback:")
            print("   " + "\n   ".join(failure['traceback'].split('\n')))
    
    # Overall result
    print("\n" + "=" * 80)
    overall_success = total_failures == 0 and total_errors == 0
    if overall_success:
        print("OVERALL RESULT: ✓ ALL TESTS PASSED!")
    else:
        print("OVERALL RESULT: ✗ SOME TESTS FAILED")
    print("=" * 80)
    
    return overall_success

if __name__ == "__main__":
    start_time = time.time()
    success = run_all_tests()
    elapsed_time = time.time() - start_time
    
    print(f"\nTest suite completed in {elapsed_time:.2f} seconds")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)