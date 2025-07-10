#!/usr/bin/env python3
"""
Quick Test Suite for Modular Functions

This script performs basic import and functionality tests on all modular functions
to verify they work correctly after extraction from the legacy codebase.
"""

import sys
import os
from pathlib import Path
import importlib.util

def test_module_import(module_path):
    """Test if a module can be imported successfully."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    """Run basic tests on all modular functions."""
    print("="*60)
    print("MODULAR FUNCTIONS TEST SUITE")
    print("="*60)
    
    # Get the modular functions directory
    modular_dir = Path(__file__).parent / "modular_functions"
    
    if not modular_dir.exists():
        print("❌ ERROR: modular_functions directory not found!")
        return False
    
    # List of modules to test
    modules = [
        "image_feature_extractor.py",
        "image_transformation_engine.py", 
        "statistical_analysis_toolkit.py",
        "defect_detection_engine.py",
        "image_similarity_analyzer.py",
        "image_segmentation_toolkit.py",
        "pipeline_orchestrator.py"
    ]
    
    print(f"Testing {len(modules)} modules...\n")
    
    results = []
    for module_name in modules:
        module_path = modular_dir / module_name
        
        if not module_path.exists():
            print(f"❌ {module_name}: File not found")
            results.append((module_name, False, "File not found"))
            continue
        
        # Test import
        success, error = test_module_import(module_path)
        
        if success:
            print(f"✅ {module_name}: Import successful")
            results.append((module_name, True, None))
        else:
            print(f"❌ {module_name}: Import failed - {error}")
            results.append((module_name, False, error))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Modules tested: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    
    if successful == total:
        print("\n🎉 ALL TESTS PASSED! All modules are ready for use.")
        return True
    else:
        print(f"\n⚠️  {total - successful} modules failed testing. Check errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
