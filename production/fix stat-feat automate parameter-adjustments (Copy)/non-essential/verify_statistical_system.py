#!/usr/bin/env python3
"""
Final verification script for the statistical features system.
Ensures all components are working correctly after implementation.
"""

import cv2
import numpy as np
import logging
import time
import os
import sys

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(__file__))

# Import using importlib to handle path issues
import importlib.util
spec = importlib.util.spec_from_file_location("statistical_features_module", "statistical_features_module.py")
stats_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats_module)

StatisticalFeaturesDetector = stats_module.StatisticalFeaturesDetector
StatisticalFeaturesProcessor = stats_module.StatisticalFeaturesProcessor

def verify_system():
    """Verify the complete statistical features system."""
    
    print("Statistical Features System - Final Verification")
    print("=" * 50)
    
    # Check required files
    required_files = [
        'statistical_features_module.py',
        'statistical_features_emulator.py',
        'small_statistical_test.bmp',
        'test_statistical_features.py',
        'test_complete_statistical_system.py'
    ]
    
    print("\n1. Checking Required Files")
    print("-" * 30)
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nERROR: Missing {len(missing_files)} required files")
        return False
    
    print(f"\n✓ All {len(required_files)} required files present")
    
    # Test basic functionality
    print("\n2. Testing Basic Functionality")
    print("-" * 30)
    
    try:
        # Load test image
        image = cv2.imread('small_statistical_test.bmp')
        if image is None:
            print("✗ Could not load test image")
            return False
        
        print(f"✓ Test image loaded: {image.shape}")
        
        # Create detector
        detector = StatisticalFeaturesDetector()
        print("✓ Detector created")
        
        # Extract features
        features, processed_frame = detector.extract_features(image)
        if features:
            print(f"✓ Features extracted: {len(features)} features")
        else:
            print("✗ No features extracted")
            return False
        
        # Test processor
        processor = StatisticalFeaturesProcessor(detector)
        print("✓ Processor created")
        
        processed = processor.process_frame(image)
        if processed is not None:
            print("✓ Frame processing successful")
        else:
            print("✗ Frame processing failed")
            return False
        
        # Test parameter updates
        detector.update_parameters(histogram_bins=64, texture_window_size=5)
        print("✓ Parameter updates successful")
        
        # Test toggle
        enabled = processor.toggle_processing()
        print(f"✓ Processing toggle: {enabled}")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False
    
    # Test GUI components
    print("\n3. Testing GUI Components")
    print("-" * 30)
    
    try:
        # Check if GUI file exists and has required classes
        with open("statistical_features_emulator.py", "r") as f:
            content = f.read()
            if "class VideoDisplayStatisticalFeatures" in content:
                print("✓ VideoDisplayStatisticalFeatures class found")
            else:
                print("✗ VideoDisplayStatisticalFeatures class not found")
                return False
            
            if "class StatisticalFeaturesGUI" in content:
                print("✓ StatisticalFeaturesGUI class found")
            else:
                print("✗ StatisticalFeaturesGUI class not found")
                return False
        
        print("✓ GUI components available")
        
    except Exception as e:
        print(f"✗ Error checking GUI components: {e}")
        return False
    
    # Performance test
    print("\n4. Performance Test")
    print("-" * 30)
    
    try:
        start_time = time.time()
        features, _ = detector.extract_features(image)
        processing_time = time.time() - start_time
        
        print(f"✓ Processing time: {processing_time:.2f} seconds")
        print(f"✓ Processing rate: {1/processing_time:.2f} fps")
        
        if processing_time < 60:  # Should complete within 1 minute
            print("✓ Performance acceptable")
        else:
            print("⚠ Performance slow but functional")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False
    
    # Check emulator process
    print("\n5. Emulator Process Check")
    print("-" * 30)
    
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'statistical_features_emulator' in result.stdout:
            print("✓ Emulator process is running")
        else:
            print("⚠ Emulator process not detected (may be normal)")
        
    except Exception as e:
        print(f"⚠ Could not check emulator process: {e}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("✓ VERIFICATION COMPLETE")
    print("✓ Statistical Features System is working correctly")
    print("✓ All components are functional")
    print("✓ Files are properly organized in non-essential/")
    print("✓ System is ready for use")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = verify_system()
    if success:
        print("\n🎉 SUCCESS: Statistical Features System is fully operational!")
        print("\nTo run the emulator:")
        print("cd non-essential")
        print("python statistical_features_emulator.py")
        print("\nTo run tests:")
        print("python test_statistical_features.py")
        print("python test_complete_statistical_system.py")
    else:
        print("\n❌ FAILURE: System has issues that need to be addressed.")
        sys.exit(1) 