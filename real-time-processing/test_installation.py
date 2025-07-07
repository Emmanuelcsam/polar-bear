#!/usr/bin/env python3
"""
Test script to verify the installation and setup.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all main modules can be imported."""
    print("Testing imports...")
    
    tests = [
        ("Core Framework", "src.core.integrated_geometry_system"),
        ("Circle Detector", "src.applications.realtime_circle_detector"),
        ("Example App", "src.applications.example_application"),
        ("Calibration Tool", "src.tools.realtime_calibration_tool"),
        ("Benchmark Tool", "src.tools.performance_benchmark_tool"),
    ]
    
    all_passed = True
    
    for name, module_path in tests:
        try:
            __import__(module_path)
            print(f"✓ {name}: OK")
        except ImportError as e:
            print(f"✗ {name}: FAILED - {e}")
            all_passed = False
    
    return all_passed

def test_dependencies():
    """Test if required dependencies are available."""
    print("\nTesting dependencies...")
    
    deps = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Matplotlib", "matplotlib"),
        ("Pillow", "PIL"),
        ("psutil", "psutil"),
    ]
    
    optional_deps = [
        ("pypylon (Basler cameras)", "pypylon"),
        ("GPUtil (GPU monitoring)", "GPUtil"),
        ("pandas (benchmarking)", "pandas"),
        ("tkinter (GUI)", "tkinter"),
    ]
    
    all_required_present = True
    
    print("\nRequired dependencies:")
    for name, module in deps:
        try:
            __import__(module)
            print(f"✓ {name}: OK")
        except ImportError:
            print(f"✗ {name}: MISSING")
            all_required_present = False
    
    print("\nOptional dependencies:")
    for name, module in optional_deps:
        try:
            __import__(module)
            print(f"✓ {name}: OK")
        except ImportError:
            print(f"- {name}: Not installed")
    
    return all_required_present

def main():
    """Run all tests."""
    print("=" * 50)
    print("Real-time Geometry Detection System")
    print("Installation Test")
    print("=" * 50)
    
    imports_ok = test_imports()
    deps_ok = test_dependencies()
    
    print("\n" + "=" * 50)
    if imports_ok and deps_ok:
        print("✓ All tests passed! The system is ready to use.")
        print("\nRun one of these commands to start:")
        print("  - ./run_linux.sh (Linux)")
        print("  - run_windows.bat (Windows)")
        print("  - python3 run_geometry_demo.py")
    else:
        print("✗ Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 50)

if __name__ == "__main__":
    main()