# run_all_tests.py
#!/usr/bin/env python3
"""
Master test runner for all modules in the image processing system.
This script runs all individual test files and provides a comprehensive test report.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_test_file(test_file):
    """Run a single test file and return the results."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout per test file
        )

        duration = time.time() - start_time

        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Return test results
        return {
            'file': test_file,
            'success': result.returncode == 0,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"❌ {test_file} TIMED OUT after {duration:.2f}s")
        return {
            'file': test_file,
            'success': False,
            'duration': duration,
            'stdout': '',
            'stderr': 'Test timed out',
            'returncode': -1
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ {test_file} FAILED with exception: {e}")
        return {
            'file': test_file,
            'success': False,
            'duration': duration,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def main():
    """Run all tests and generate a comprehensive report."""
    print("🚀 Starting comprehensive test suite for image processing system")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    # Find all test files
    test_files = []
    for i in range(11):  # 0 through 10
        test_file = f"test_{i}_*.py"
        matches = list(Path('.').glob(test_file))
        if matches:
            test_files.extend([str(f) for f in matches])

    # Sort test files for consistent order
    test_files.sort()

    print(f"\nFound {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file}")

    if not test_files:
        print("❌ No test files found! Make sure test files are in the current directory.")
        return 1

    # Run all tests
    total_start_time = time.time()
    results = []

    for test_file in test_files:
        result = run_test_file(test_file)
        results.append(result)

    total_duration = time.time() - total_start_time

    # Generate summary report
    print(f"\n{'='*80}")
    print("TEST SUMMARY REPORT")
    print(f"{'='*80}")

    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed

    print(f"Total tests run: {len(results)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Total duration: {total_duration:.2f}s")

    # Detailed results
    print(f"\nDetailed Results:")
    print(f"{'File':<35} {'Status':<10} {'Duration':<12} {'Details'}")
    print("-" * 80)

    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = f"{result['duration']:.2f}s"
        details = ""

        if not result['success']:
            if result['stderr']:
                details = result['stderr'][:40] + "..." if len(result['stderr']) > 40 else result['stderr']
            elif result['returncode'] != 0:
                details = f"Exit code: {result['returncode']}"

        print(f"{result['file']:<35} {status:<10} {duration:<12} {details}")

    # Failed tests details
    if failed > 0:
        print(f"\n{'='*80}")
        print("FAILED TESTS DETAILS")
        print(f"{'='*80}")

        for result in results:
            if not result['success']:
                print(f"\n❌ {result['file']}:")
                if result['stderr']:
                    print(f"Error: {result['stderr']}")
                if result['stdout']:
                    print(f"Output: {result['stdout']}")

    # System information
    print(f"\n{'='*80}")
    print("SYSTEM INFORMATION")
    print(f"{'='*80}")

    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")

    # Check for optional dependencies
    dependencies = ['cv2', 'torch', 'numpy']
    print(f"\nDependency Status:")
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✅ {dep} - Available")
        except ImportError:
            print(f"  ❌ {dep} - Not available")

    # Return appropriate exit code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
