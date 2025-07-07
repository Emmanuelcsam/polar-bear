#!/usr/bin/env python3
"""
Quick validation script to check if tests can be imported and basic functionality works
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def validate_imports():
    """Validate that all modules can be imported"""
    print("Validating imports...")
    
    modules_to_check = [
        # Main modules
        ('src.core.integrated_geometry_system', 'Main geometry system'),
        ('src.tools.performance_benchmark_tool', 'Performance benchmarking'),
        ('src.tools.realtime_calibration_tool', 'Calibration tool'),
        ('src.applications.example_application', 'Example application'),
        ('src.tools.setup_installer', 'Setup installer'),
        ('src.tools.uv_compatible_setup', 'UV setup helper'),
        ('src.core.python313_fix', 'Python 3.13 fix'),
        
        # Test modules
        ('src.tests.test_integrated_geometry_system', 'Tests for main system'),
        ('src.tests.test_performance_benchmark_tool', 'Tests for benchmarking'),
        ('src.tests.test_realtime_calibration_tool', 'Tests for calibration'),
        ('src.tests.test_example_application', 'Tests for example app'),
        ('src.tests.test_setup_installer', 'Tests for installer'),
        ('src.tests.test_uv_compatible_setup', 'Tests for UV setup'),
        ('src.tests.test_python313_fix', 'Tests for Python 3.13 fix'),
    ]
    
    failed_imports = []
    
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name} ({description})")
        except ImportError as e:
            print(f"  ✗ {module_name} ({description}): {e}")
            failed_imports.append((module_name, str(e)))
        except Exception as e:
            print(f"  ✗ {module_name} ({description}): Unexpected error: {e}")
            failed_imports.append((module_name, str(e)))
    
    return failed_imports

def check_test_structure():
    """Check that test files have proper structure"""
    print("\nChecking test structure...")
    
    test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]
    
    for test_file in test_files:
        print(f"\n  Checking {test_file}:")
        
        with open(test_file, 'r') as f:
            content = f.read()
            
        # Check for required elements
        checks = [
            ('import unittest', 'unittest import'),
            ('class Test', 'test class definitions'),
            ('def test_', 'test methods'),
            ("if __name__ == '__main__':", 'main block'),
        ]
        
        for pattern, description in checks:
            if pattern in content:
                print(f"    ✓ Has {description}")
            else:
                print(f"    ✗ Missing {description}")

def check_dependencies():
    """Check if required dependencies are available"""
    print("\nChecking dependencies...")
    
    dependencies = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('psutil', 'psutil'),
        ('pandas', 'pandas'),
    ]
    
    missing_deps = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name} is installed")
        except ImportError:
            print(f"  ✗ {name} is NOT installed")
            missing_deps.append(name)
    
    return missing_deps

def main():
    """Run all validations"""
    print("=" * 60)
    print("GEOMETRY DETECTION SYSTEM - TEST VALIDATION")
    print("=" * 60)
    
    # Check imports
    failed_imports = validate_imports()
    
    # Check test structure
    check_test_structure()
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if not failed_imports and not missing_deps:
        print("\n✓ All validations passed!")
        print("\nYou can now run the tests with:")
        print("  python run_all_tests.py")
        return True
    else:
        print("\n✗ Some validations failed:")
        
        if failed_imports:
            print(f"\n  Failed imports: {len(failed_imports)}")
            for module, error in failed_imports[:5]:  # Show first 5
                print(f"    - {module}: {error}")
                
        if missing_deps:
            print(f"\n  Missing dependencies: {len(missing_deps)}")
            for dep in missing_deps:
                print(f"    - {dep}")
            print("\n  Install with: pip install -r requirements_test.txt")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)