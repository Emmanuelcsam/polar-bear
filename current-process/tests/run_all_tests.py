#!/usr/bin/env python3
"""
Run all tests for the enhanced fiber optic defect detection system
"""

import sys
import subprocess
from pathlib import Path
import time

def run_tests():
    """Run all test suites"""
    print("="*60)
    print("Running Enhanced Fiber Optic Defect Detection Tests")
    print("="*60)
    
    test_dir = Path(__file__).parent
    project_dir = test_dir.parent
    
    # Add current-process to Python path
    sys.path.insert(0, str(project_dir / "current-process"))
    
    # Test files to run
    test_files = [
        "test_enhanced_process.py",
        "test_enhanced_separation.py", 
        "test_enhanced_detection.py",
        "test_integration.py"
    ]
    
    all_passed = True
    results = {}
    
    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"Running {test_file}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run pytest for this file
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            cwd=test_dir,
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        
        # Store results
        results[test_file] = {
            'passed': result.returncode == 0,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            all_passed = False
            print(f"\n❌ {test_file} FAILED")
        else:
            print(f"\n✅ {test_file} PASSED in {duration:.2f}s")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_file, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{test_file:<30} {status} ({result['duration']:.2f}s)")
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60)
    
    return 0 if all_passed else 1


def run_single_test(test_name):
    """Run a single test function"""
    print(f"Running single test: {test_name}")
    
    test_dir = Path(__file__).parent
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-k", test_name],
        cwd=test_dir,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        return run_single_test(test_name)
    else:
        # Run all tests
        return run_tests()


if __name__ == "__main__":
    sys.exit(main())