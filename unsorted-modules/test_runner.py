#!/usr/bin/env python3
"""
Comprehensive test runner for all fiber optic inspection modules.
Runs all tests and generates a summary report.
"""

import sys
import unittest
import traceback
from pathlib import Path
import json
import datetime

# Test modules to run
TEST_MODULES = [
    'test_ai_segmenter_pytorch',
    'test_anomaly_detector_pytorch',
    'test_clustering',
    'test_dataset_builder',
    'test_defect_detection',
    'test_detection_ai',
    'test_do2mr_lei_detector',
    'test_feature_extraction',
    'test_fiber_dataset_pytorch',
    'test_gemma_fiber_analyzer',
    'test_llama_vision_finetuner',
    'test_ml_dataset_builder',
    'test_opencv_processor',
    'test_realtime_analyzer',
    'test_realtime_dashboard',
    'test_realtime_location_pipeline',
    'test_separation_ai',
    'test_tensorflow_attachment',
    'test_torch_quality_classifier',
    'test_train_anomaly',
    'test_train_segmenter',
    'test_utils',
    'test_zone_segmentation'
]

def run_all_tests():
    """Run all test modules and generate report."""
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'total_modules': len(TEST_MODULES),
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': {}
    }
    
    print("=" * 80)
    print("FIBER OPTIC INSPECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Running tests for {len(TEST_MODULES)} modules...")
    print()
    
    for module_name in TEST_MODULES:
        print(f"\n{'='*60}")
        print(f"Testing: {module_name}")
        print('='*60)
        
        try:
            # Import test module
            test_module = __import__(module_name)
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            # Record results
            module_results = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success': result.wasSuccessful(),
                'failure_details': [str(f) for f in result.failures],
                'error_details': [str(e) for e in result.errors]
            }
            
            results['details'][module_name] = module_results
            
            if result.wasSuccessful():
                results['passed'] += 1
                print(f"\n✓ {module_name}: ALL TESTS PASSED")
            else:
                results['failed'] += 1
                print(f"\n✗ {module_name}: TESTS FAILED")
                
        except ImportError as e:
            results['errors'] += 1
            results['details'][module_name] = {
                'error': f"Import error: {str(e)}",
                'traceback': traceback.format_exc()
            }
            print(f"\n✗ {module_name}: IMPORT ERROR - {str(e)}")
            
        except Exception as e:
            results['errors'] += 1
            results['details'][module_name] = {
                'error': f"Unexpected error: {str(e)}",
                'traceback': traceback.format_exc()
            }
            print(f"\n✗ {module_name}: UNEXPECTED ERROR - {str(e)}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total modules tested: {results['total_modules']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Errors: {results['errors']}")
    print(f"Success rate: {(results['passed'] / results['total_modules'] * 100):.1f}%")
    
    # Save detailed results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: test_results.json")
    
    # Return exit code
    return 0 if results['failed'] == 0 and results['errors'] == 0 else 1

if __name__ == "__main__":
    sys.exit(run_all_tests())