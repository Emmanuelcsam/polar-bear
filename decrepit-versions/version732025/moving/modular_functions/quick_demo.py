#!/usr/bin/env python3
"""
Quick Demo of Working Modular Functions
Demonstrates the successfully modularized functions that are ready to use.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

def create_test_image():
    """Create a simple test image for demonstration"""
    # Create a synthetic fiber-like image
    img = np.zeros((300, 300), dtype=np.uint8)
    
    # Add some circular regions to simulate fiber
    cv2.circle(img, (150, 150), 80, 180, -1)  # Outer circle (cladding)
    cv2.circle(img, (150, 150), 40, 220, -1)  # Inner circle (core)
    cv2.circle(img, (150, 150), 15, 255, -1)  # Bright core
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def demo_working_modules():
    """Demonstrate the working modular functions"""
    
    print("🔬 Fiber Optic Analysis - Modular Functions Demo")
    print("=" * 55)
    
    # Create test image
    test_img = create_test_image()
    test_path = "demo_test_image.jpg"
    cv2.imwrite(test_path, test_img)
    print(f"📸 Created test image: {test_path}")
    
    # Test ImageEnhancer
    print("\n🎨 Testing ImageEnhancer...")
    try:
        from modular_functions.image_enhancer import ImageEnhancer
        enhancer = ImageEnhancer()
        enhanced = enhancer.enhance_image(test_path)
        if enhanced is not None:
            cv2.imwrite("demo_enhanced.jpg", enhanced)
            print("✅ ImageEnhancer working! Enhanced image saved as demo_enhanced.jpg")
        else:
            print("⚠️ ImageEnhancer returned None")
    except Exception as e:
        print(f"❌ ImageEnhancer error: {e}")
    
    # Test BrightCoreExtractor  
    print("\n💡 Testing BrightCoreExtractor...")
    try:
        from modular_functions.bright_core_extractor import BrightCoreExtractor
        extractor = BrightCoreExtractor()
        results = extractor.extract_bright_core(test_path)
        if results and 'bright_regions' in results:
            print(f"✅ BrightCoreExtractor working! Found {len(results['bright_regions'])} bright regions")
        else:
            print("⚠️ BrightCoreExtractor found no regions")
    except Exception as e:
        print(f"❌ BrightCoreExtractor error: {e}")
    
    # Test HoughFiberSeparator
    print("\n🔍 Testing HoughFiberSeparator...")
    try:
        from modular_functions.hough_fiber_separator import HoughFiberSeparator
        separator = HoughFiberSeparator()
        results = separator.detect_fiber_regions(test_path)
        if results and 'detected_circles' in results:
            print(f"✅ HoughFiberSeparator working! Detected {len(results['detected_circles'])} circles")
        else:
            print("⚠️ HoughFiberSeparator found no circles")
    except Exception as e:
        print(f"❌ HoughFiberSeparator error: {e}")
    
    # Test AdaptiveIntensitySegmenter
    print("\n📊 Testing AdaptiveIntensitySegmenter...")
    try:
        from modular_functions.adaptive_intensity_segmenter import AdaptiveIntensitySegmenter
        segmenter = AdaptiveIntensitySegmenter()
        results = segmenter.segment_image(test_path)
        if results and 'regions' in results:
            print(f"✅ AdaptiveIntensitySegmenter working! Found {len(results['regions'])} regions")
        else:
            print("⚠️ AdaptiveIntensitySegmenter found no regions")
    except Exception as e:
        print(f"❌ AdaptiveIntensitySegmenter error: {e}")
    
    # Test TraditionalDefectDetector
    print("\n🔍 Testing TraditionalDefectDetector...")
    try:
        from modular_functions.traditional_defect_detector import TraditionalDefectDetector
        detector = TraditionalDefectDetector()
        results = detector.detect_defects(test_path)
        if results and 'defects' in results:
            print(f"✅ TraditionalDefectDetector working! Found {len(results['defects'])} potential defects")
        else:
            print("⚠️ TraditionalDefectDetector found no defects")
    except Exception as e:
        print(f"❌ TraditionalDefectDetector error: {e}")
    
    print("\n" + "=" * 55)
    print("🎉 Demo complete! Working modules are ready for use.")
    print("📁 Check the created demo files in the current directory.")
    
    # Cleanup
    try:
        os.remove(test_path)
        print(f"🧹 Cleaned up {test_path}")
    except:
        pass

if __name__ == "__main__":
    # Add the current directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    demo_working_modules()
