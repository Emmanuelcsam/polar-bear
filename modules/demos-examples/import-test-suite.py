#!/usr/bin/env python3
"""
Basic Import Test for Modular Functions
Tests that all modules can be imported without dependencies that might not be available.
"""

import sys
from pathlib import Path

def test_imports():
    """Test basic imports of all modular functions"""
    
    print("Testing modular function imports...")
    print("=" * 50)
    
    # Test adaptive intensity segmenter
    try:
        from adaptive_intensity_segmenter import AdaptiveIntensitySegmenter
        print("✓ AdaptiveIntensitySegmenter imported successfully")
    except ImportError as e:
        print(f"✗ AdaptiveIntensitySegmenter failed: {e}")
    
    # Test bright core extractor
    try:
        from bright_core_extractor import BrightCoreExtractor
        print("✓ BrightCoreExtractor imported successfully")
    except ImportError as e:
        print(f"✗ BrightCoreExtractor failed: {e}")
    
    # Test image enhancer
    try:
        from image_enhancer import ImageEnhancer
        print("✓ ImageEnhancer imported successfully")
    except ImportError as e:
        print(f"✗ ImageEnhancer failed: {e}")
    
    # Test traditional defect detector
    try:
        from traditional_defect_detector import TraditionalDefectDetector
        print("✓ TraditionalDefectDetector imported successfully")
    except ImportError as e:
        print(f"✗ TraditionalDefectDetector failed: {e}")
    
    # Test hough fiber separator
    try:
        from hough_fiber_separator import HoughFiberSeparator
        print("✓ HoughFiberSeparator imported successfully")
    except ImportError as e:
        print(f"✗ HoughFiberSeparator failed: {e}")
    
    # Test gradient fiber segmenter (may have issues due to scipy)
    try:
        from gradient_fiber_segmenter import GradientFiberSegmenter
        print("✓ GradientFiberSegmenter imported successfully")
    except ImportError as e:
        print(f"✗ GradientFiberSegmenter failed: {e}")
    
    print("=" * 50)
    print("Import testing complete!")

if __name__ == "__main__":
    test_imports()
