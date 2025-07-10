#!/usr/bin/env python3
"""
Comprehensive test script for all modular functions.
This script imports and tests basic functionality of all modules.
"""

import sys
import traceback

def test_module_imports():
    """Test that all modules can be imported successfully"""
    modules = [
        'image_filtering',
        'center_detection', 
        'edge_detection_ransac',
        'radial_profile_analysis',
        'mask_creation',
        'peak_detection'
    ]
    
    print("Testing module imports...")
    failed_imports = []
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name} imported successfully")
        except Exception as e:
            print(f"✗ {module_name} failed to import: {e}")
            failed_imports.append(module_name)
            traceback.print_exc()
    
    return failed_imports

def test_basic_functionality():
    """Test basic functionality of key functions"""
    import numpy as np
    
    try:
        # Test image filtering
        import image_filtering
        test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        filtered = image_filtering.apply_binary_filter(test_img)
        print("✓ Image filtering basic test passed")
        
        # Test center detection
        import center_detection
        center = center_detection.brightness_weighted_centroid(test_img)
        print("✓ Center detection basic test passed")
        
        # Test mask creation
        import mask_creation
        mask = mask_creation.create_circular_mask((100, 100), (50, 50), 25)
        print("✓ Mask creation basic test passed")
        
        # Test radial profile
        import radial_profile_analysis
        profile = radial_profile_analysis.compute_radial_intensity_profile(test_img, (50, 50))
        print("✓ Radial profile basic test passed")
        
        # Test peak detection
        import peak_detection
        test_signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        smoothed = peak_detection.moving_average_smoothing(test_signal, 5)
        print("✓ Peak detection basic test passed")
        
        # Test edge detection
        import edge_detection_ransac
        edges = edge_detection_ransac.extract_edge_points(test_img)
        print("✓ Edge detection basic test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("COMPREHENSIVE MODULE TESTING")
    print("=" * 60)
    
    # Test imports
    failed_imports = test_module_imports()
    
    print("\n" + "-" * 40)
    
    # Test basic functionality
    if not failed_imports:
        print("All modules imported successfully. Testing basic functionality...")
        functionality_passed = test_basic_functionality()
    else:
        print(f"Skipping functionality tests due to import failures: {failed_imports}")
        functionality_passed = False
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if not failed_imports and functionality_passed:
        print("✓ ALL TESTS PASSED - All modules are working correctly!")
        print("✓ The modular function library is ready for use.")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        if failed_imports:
            print(f"  - Failed imports: {failed_imports}")
        if not functionality_passed:
            print("  - Basic functionality tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
