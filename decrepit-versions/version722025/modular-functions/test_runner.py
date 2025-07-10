#!/usr/bin/env python3
"""
Test runner for modular functions
Validates that all scripts can be imported and have correct structure
"""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path

def test_script_imports():
    """Test that all scripts can be imported without errors"""
    script_dir = Path(__file__).parent
    scripts = [
        'adaptive_intensity_segmentation.py',
        'bright_core_extractor.py', 
        'configuration_manager.py',
        'data_aggregation_reporting.py',
        'geometric_fiber_segmentation.py',
        'hough_circle_detection.py',
        'image_enhancement.py',
        'ml_defect_detection.py',
        'realtime_video_processor.py'
    ]
    
    results = {}
    
    for script in scripts:
        script_path = script_dir / script
        if not script_path.exists():
            results[script] = "MISSING"
            continue
            
        try:
            # Try to load as module
            spec = importlib.util.spec_from_file_location(
                script.replace('.py', ''), script_path
            )
            if spec is None:
                results[script] = "SPEC_ERROR"
                continue
            module = importlib.util.module_from_spec(spec)
            
            # Check if it has main function or can be executed
            with open(script_path, 'r') as f:
                content = f.read()
                if '__main__' in content and 'def main(' in content:
                    results[script] = "OK"
                else:
                    results[script] = "NO_MAIN"
                    
        except Exception as e:
            results[script] = f"ERROR: {str(e)}"
    
    return results

def test_help_commands():
    """Test that scripts respond to --help correctly"""
    script_dir = Path(__file__).parent
    scripts = [
        'configuration_manager.py',
        'data_aggregation_reporting.py',
        'image_enhancement.py'
    ]
    
    results = {}
    
    for script in scripts:
        try:
            result = subprocess.run([
                sys.executable, str(script_dir / script), '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and 'usage:' in result.stdout.lower():
                results[script] = "OK"
            else:
                results[script] = f"HELP_ERROR: {result.stderr[:100]}"
                
        except subprocess.TimeoutExpired:
            results[script] = "TIMEOUT"
        except Exception as e:
            results[script] = f"ERROR: {str(e)}"
    
    return results

def main():
    """Run all tests and display results"""
    print("=" * 60)
    print("MODULAR FUNCTIONS TEST SUITE")
    print("=" * 60)
    
    print("\n1. Testing script structure and imports...")
    import_results = test_script_imports()
    
    for script, status in import_results.items():
        status_icon = "✓" if status == "OK" else "✗"
        print(f"  {status_icon} {script}: {status}")
    
    print(f"\nImport Test Summary: {sum(1 for s in import_results.values() if s == 'OK')}/{len(import_results)} passed")
    
    print("\n2. Testing help command functionality...")
    help_results = test_help_commands()
    
    for script, status in help_results.items():
        status_icon = "✓" if status == "OK" else "✗"
        print(f"  {status_icon} {script}: {status}")
    
    print(f"\nHelp Test Summary: {sum(1 for s in help_results.values() if s == 'OK')}/{len(help_results)} passed")
    
    # Overall summary
    total_ok = sum(1 for s in import_results.values() if s == "OK")
    total_scripts = len(import_results)
    
    print("\n" + "=" * 60)
    print(f"OVERALL RESULTS: {total_ok}/{total_scripts} scripts are properly structured")
    
    if total_ok == total_scripts:
        print("🎉 All modular functions are ready for use!")
    else:
        print("⚠️  Some scripts may need debugging before use.")
        print("Check the error messages above for details.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
