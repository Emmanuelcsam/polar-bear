#!/usr/bin/env python3
"""
Test runner script for Fiber Optics Neural Network tests
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests with coverage reporting"""
    
    # Get test directory
    test_dir = Path(__file__).parent
    project_dir = test_dir.parent
    
    # Change to project directory
    os.chdir(project_dir)
    
    print("=" * 60)
    print("FIBER OPTICS NEURAL NETWORK - TEST SUITE")
    print("=" * 60)
    print()
    
    # Install test requirements
    print("Installing test requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "test/requirements-test.txt"], check=True)
    print()
    
    # Run tests with coverage
    print("Running tests with coverage...")
    print("-" * 60)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "test/",
        "-v",  # Verbose output
        "--cov=.",  # Coverage for current directory
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal report with missing lines
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
        "--maxfail=10",  # Stop after 10 failures
    ]
    
    result = subprocess.run(cmd)
    
    print()
    print("-" * 60)
    
    if result.returncode == 0:
        print("✅ All tests passed!")
        print()
        print("Coverage report generated in htmlcov/index.html")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    run_tests()