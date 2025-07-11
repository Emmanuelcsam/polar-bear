#!/usr/bin/env python3
"""
Comprehensive validation script for the image processing system.
This script validates that all components work correctly together.
"""

import os
import sys
import subprocess
import time
import importlib
from pathlib import Path

def validate_project_structure():
    """Validate that all required files exist."""
    print("📁 Validating project structure...")

    required_files = [
        '0_config.py',
        '1_batch_processor.py',
        '2_intensity_reader.py',
        '3_pattern_recognizer.py',
        '4_generative_learner.py',
        '5_image_generator.py',
        '6_deviation_detector.py',
        '7_geometry_recognizer.py',
        '8_real_time_anomaly.py',
        '9_gpu_example.py',
        '10_hpc_parallel_cpu.py',
        'requirements.txt',
        'README.md',
        'run_all_tests.py',
        'demo.py',
        'create_test_images.py'
    ]

    test_files = [f'test_{i}_{name}' for i in range(11) for name in ['config', 'batch_processor', 'intensity_reader', 'pattern_recognizer', 'generative_learner', 'image_generator', 'deviation_detector', 'geometry_recognizer', 'real_time_anomaly', 'gpu_example', 'hpc_parallel_cpu'] if f'test_{i}' in f'test_{i}_{name}']

    # Find actual test files
    actual_test_files = list(Path('.').glob('test_*.py'))

    missing_files = []
    found_files = []

    for file in required_files:
        if os.path.exists(file):
            found_files.append(file)
            print(f"  ✅ {file}")
        else:
            missing_files.append(file)
            print(f"  ❌ {file} (missing)")

    print(f"\nTest files found: {len(actual_test_files)}")
    for test_file in actual_test_files:
        print(f"  ✅ {test_file}")

    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required files")
        return False

    print(f"\n✅ All {len(found_files)} required files present")
    return True

def validate_imports():
    """Validate that all modules can be imported."""
    print("\n🔍 Validating module imports...")

    modules = [
        '0_config',
        '1_batch_processor',
        '2_intensity_reader',
        '3_pattern_recognizer',
        '4_generative_learner',
        '5_image_generator',
        '6_deviation_detector',
        '7_geometry_recognizer',
        '8_real_time_anomaly',
        '9_gpu_example',
        '10_hpc_parallel_cpu'
    ]

    success_count = 0

    for module_name in modules:
        try:
            # Try to import the module
            spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            print(f"  ✅ {module_name}")
            success_count += 1

        except Exception as e:
            print(f"  ❌ {module_name}: {e}")

    print(f"\n✅ {success_count}/{len(modules)} modules imported successfully")
    return success_count == len(modules)

def validate_dependencies():
    """Check dependency availability."""
    print("\n📦 Validating dependencies...")

    core_deps = ['numpy', 'os', 'sys', 'time', 'multiprocessing']
    optional_deps = ['cv2', 'torch', 'torchvision']

    available_core = 0
    available_optional = 0

    print("Core dependencies:")
    for dep in core_deps:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'torch':
                import torch
            elif dep == 'numpy':
                import numpy
            elif dep == 'multiprocessing':
                import multiprocessing
            elif dep == 'os':
                import os
            elif dep == 'sys':
                import sys
            elif dep == 'time':
                import time

            print(f"  ✅ {dep}")
            available_core += 1
        except ImportError:
            print(f"  ❌ {dep}")

    print("Optional dependencies:")
    for dep in optional_deps:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'torch':
                import torch
            elif dep == 'torchvision':
                import torchvision

            print(f"  ✅ {dep}")
            available_optional += 1
        except ImportError:
            print(f"  ⚠️  {dep} (optional)")

    print(f"\n✅ {available_core}/{len(core_deps)} core dependencies available")
    print(f"✅ {available_optional}/{len(optional_deps)} optional dependencies available")

    return available_core == len(core_deps)

def run_quick_test():
    """Run a quick functional test."""
    print("\n🧪 Running quick functional test...")

    # Test basic config
    try:
        spec = importlib.util.spec_from_file_location("config", "0_config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        print("  ✅ Config module loaded")
        print(f"     - Learning mode: {config.LEARNING_MODE}")
        print(f"     - Input dir: {config.INPUT_DIR}")
        print(f"     - Data dir: {config.DATA_DIR}")
        print(f"     - Output dir: {config.OUTPUT_DIR}")

        # Check directories exist
        for dir_name in [config.DATA_DIR, config.OUTPUT_DIR]:
            if os.path.exists(dir_name):
                print(f"     - {dir_name}: exists")
            else:
                print(f"     - {dir_name}: will be created")

        return True

    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def run_test_suite():
    """Run the comprehensive test suite."""
    print("\n🎯 Running comprehensive test suite...")

    try:
        result = subprocess.run(
            [sys.executable, 'run_all_tests.py'],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        print("Test suite output:")
        print("-" * 40)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print(f"❌ Some tests failed (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print("⏱️ Test suite timed out")
        return False
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

def main():
    """Run complete validation."""
    print("🔍 COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 60)

    validation_steps = [
        ("Project Structure", validate_project_structure),
        ("Module Imports", validate_imports),
        ("Dependencies", validate_dependencies),
        ("Quick Functional Test", run_quick_test),
        ("Test Suite", run_test_suite)
    ]

    results = []

    for step_name, step_func in validation_steps:
        print(f"\n{step_name}:")
        print("-" * 40)
        success = step_func()
        results.append((step_name, success))

        if success:
            print(f"✅ {step_name} passed")
        else:
            print(f"❌ {step_name} failed")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Validation steps: {passed}/{total} passed")

    for step_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {step_name:<25} {status}")

    if passed == total:
        print("\n🎉 SYSTEM VALIDATION SUCCESSFUL!")
        print("The image processing system is ready to use.")
        print("\nNext steps:")
        print("  1. Run demo: python demo.py")
        print("  2. Add your images to images_input/")
        print("  3. Run individual scripts as needed")
        return 0
    else:
        print(f"\n⚠️  SYSTEM VALIDATION INCOMPLETE ({total - passed} issues)")
        print("Please address the failed validation steps before using the system.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
