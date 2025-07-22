#!/usr/bin/env python3
"""
Test runner for Fiber Optics Neural Network
Runs all unit tests and generates a comprehensive test report
"""

import unittest
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class TestResult(unittest.TestResult):
    """Custom test result class to collect detailed information"""
    
    def __init__(self):
        super().__init__()
        self.successes = []
        self.start_time = None
        self.end_time = None
        self.test_details = {}
    
    def startTest(self, test):
        """Called when a test starts"""
        self.test_details[str(test)] = {
            'start_time': time.time(),
            'status': 'running'
        }
    
    def stopTest(self, test):
        """Called when a test finishes"""
        test_name = str(test)
        if test_name in self.test_details:
            self.test_details[test_name]['end_time'] = time.time()
            self.test_details[test_name]['duration'] = (
                self.test_details[test_name]['end_time'] - 
                self.test_details[test_name]['start_time']
            )
    
    def addSuccess(self, test):
        """Called when a test passes"""
        self.successes.append(test)
        test_name = str(test)
        if test_name in self.test_details:
            self.test_details[test_name]['status'] = 'passed'
    
    def addError(self, test, err):
        """Called when a test raises an unexpected exception"""
        self.errors.append((test, err))
        test_name = str(test)
        if test_name in self.test_details:
            self.test_details[test_name]['status'] = 'error'
            self.test_details[test_name]['error'] = self._format_error(err)
    
    def addFailure(self, test, err):
        """Called when a test fails"""
        self.failures.append((test, err))
        test_name = str(test)
        if test_name in self.test_details:
            self.test_details[test_name]['status'] = 'failed'
            self.test_details[test_name]['failure'] = self._format_error(err)
    
    def addSkip(self, test, reason):
        """Called when a test is skipped"""
        self.skipped.append((test, reason))
        test_name = str(test)
        if test_name in self.test_details:
            self.test_details[test_name]['status'] = 'skipped'
            self.test_details[test_name]['reason'] = reason
    
    def _format_error(self, err):
        """Format error information"""
        exc_type, exc_value, exc_traceback = err
        return {
            'type': exc_type.__name__,
            'message': str(exc_value),
            'traceback': ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        }
    
    def get_summary(self):
        """Get test summary"""
        total = len(self.successes) + len(self.failures) + len(self.errors) + len(self.skipped)
        return {
            'total_tests': total,
            'passed': len(self.successes),
            'failed': len(self.failures),
            'errors': len(self.errors),
            'skipped': len(self.skipped),
            'success_rate': len(self.successes) / total * 100 if total > 0 else 0,
            'total_duration': self.end_time - self.start_time if self.end_time and self.start_time else 0
        }


class VerboseTestRunner:
    """Custom test runner with verbose output"""
    
    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self.result = TestResult()
    
    def run(self, test):
        """Run the test suite"""
        self.stream.write("=" * 80 + "\n")
        self.stream.write("FIBER OPTICS NEURAL NETWORK - COMPREHENSIVE TEST SUITE\n")
        self.stream.write("=" * 80 + "\n\n")
        
        self.result.start_time = time.time()
        
        # Run tests
        test(self.result)
        
        self.result.end_time = time.time()
        
        # Print results
        self._print_results()
        
        return self.result
    
    def _print_results(self):
        """Print detailed test results"""
        summary = self.result.get_summary()
        
        # Print summary
        self.stream.write("\n" + "=" * 80 + "\n")
        self.stream.write("TEST SUMMARY\n")
        self.stream.write("=" * 80 + "\n")
        self.stream.write(f"Total Tests: {summary['total_tests']}\n")
        self.stream.write(f"Passed: {summary['passed']} ✓\n")
        self.stream.write(f"Failed: {summary['failed']} ✗\n")
        self.stream.write(f"Errors: {summary['errors']} ⚠\n")
        self.stream.write(f"Skipped: {summary['skipped']} ○\n")
        self.stream.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
        self.stream.write(f"Total Duration: {summary['total_duration']:.2f} seconds\n")
        
        # Print failures
        if self.result.failures:
            self.stream.write("\n" + "=" * 80 + "\n")
            self.stream.write("FAILURES\n")
            self.stream.write("=" * 80 + "\n")
            for test, err in self.result.failures:
                self.stream.write(f"\n{test}:\n")
                self.stream.write(f"{err[0].__name__}: {err[1]}\n")
        
        # Print errors
        if self.result.errors:
            self.stream.write("\n" + "=" * 80 + "\n")
            self.stream.write("ERRORS\n")
            self.stream.write("=" * 80 + "\n")
            for test, err in self.result.errors:
                self.stream.write(f"\n{test}:\n")
                self.stream.write(f"{err[0].__name__}: {err[1]}\n")
        
        # Print detailed test timings
        self.stream.write("\n" + "=" * 80 + "\n")
        self.stream.write("DETAILED TEST RESULTS\n")
        self.stream.write("=" * 80 + "\n")
        
        # Sort tests by duration
        sorted_tests = sorted(
            self.result.test_details.items(),
            key=lambda x: x[1].get('duration', 0),
            reverse=True
        )
        
        for test_name, details in sorted_tests[:20]:  # Top 20 slowest
            status_symbol = {
                'passed': '✓',
                'failed': '✗',
                'error': '⚠',
                'skipped': '○'
            }.get(details['status'], '?')
            
            duration = details.get('duration', 0)
            self.stream.write(f"{status_symbol} {test_name}: {duration:.3f}s\n")


def discover_and_run_tests():
    """Discover and run all tests"""
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests with custom runner
    runner = VerboseTestRunner()
    result = runner.run(suite)
    
    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': result.get_summary(),
        'test_details': result.test_details,
        'failures': [
            {
                'test': str(test),
                'error': result._format_error(err)
            }
            for test, err in result.failures
        ],
        'errors': [
            {
                'test': str(test),
                'error': result._format_error(err)
            }
            for test, err in result.errors
        ]
    }
    
    # Save report to file
    report_dir = test_dir.parent / 'test_reports'
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed test report saved to: {report_file}")
    
    # Return exit code
    return 0 if len(result.failures) == 0 and len(result.errors) == 0 else 1


def run_specific_test(test_name):
    """Run a specific test module"""
    loader = unittest.TestLoader()
    
    try:
        # Try to load the specific test module
        module = __import__(test_name)
        suite = loader.loadTestsFromModule(module)
        
        # Run tests
        runner = VerboseTestRunner()
        result = runner.run(suite)
        
        return 0 if len(result.failures) == 0 and len(result.errors) == 0 else 1
        
    except ImportError as e:
        print(f"Error: Could not import test module '{test_name}': {e}")
        return 1


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        if not test_name.startswith('test_'):
            test_name = f'test_{test_name}'
        return run_specific_test(test_name)
    else:
        # Run all tests
        return discover_and_run_tests()


if __name__ == "__main__":
    sys.exit(main())