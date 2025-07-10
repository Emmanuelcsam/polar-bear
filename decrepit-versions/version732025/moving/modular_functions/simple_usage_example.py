#!/usr/bin/env python3
"""
Simple Usage Example for Modular Functions
Demonstrates basic usage of each module with error handling.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def create_test_image():
    """Create a simple test image"""
    # Create a 512x512 test image
    image = np.ones((512, 512), dtype=np.uint8) * 50
    
    # Add a bright circular core
    cv2.circle(image, (256, 256), 30, (200,), -1)
    
    # Add a cladding ring
    cv2.circle(image, (256, 256), 120, (150,), 10)
    
    # Add some noise
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return image

def test_image_enhancement():
    """Test image enhancement module"""
    try:
        from image_enhancer import ImageEnhancer
        
        # Create test image
        image = create_test_image()
        
        # Enhance image
        enhancer = ImageEnhancer()
        enhanced = enhancer.auto_enhance(image)
        
        print("✓ Image Enhancement: SUCCESS")
        print(f"  Original shape: {image.shape}")
        print(f"  Enhanced shape: {enhanced.shape}")
        print(f"  Original mean: {np.mean(image):.1f}")
        print(f"  Enhanced mean: {np.mean(enhanced):.1f}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Image Enhancement: IMPORT ERROR - {e}")
        return False
    except Exception as e:
        print(f"✗ Image Enhancement: ERROR - {e}")
        return False

def test_bright_core_extraction():
    """Test bright core extraction module"""
    try:
        from bright_core_extractor import BrightCoreExtractor
        
        # Create test image and save it
        image = create_test_image()
        test_path = "temp_test_image.png"
        cv2.imwrite(test_path, image)
        
        # Extract core
        extractor = BrightCoreExtractor()
        result = extractor.extract_bright_core(test_path)
        
        # Clean up
        Path(test_path).unlink(missing_ok=True)
        
        if result['success']:
            print("✓ Bright Core Extraction: SUCCESS")
            print(f"  Center: {result['center']}")
            print(f"  Radius: {result['core_radius']}")
            print(f"  Confidence: {result['confidence']:.3f}")
        else:
            print(f"✗ Bright Core Extraction: FAILED - {result.get('error', 'Unknown error')}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Bright Core Extraction: IMPORT ERROR - {e}")
        return False
    except Exception as e:
        print(f"✗ Bright Core Extraction: ERROR - {e}")
        return False

def test_adaptive_segmentation():
    """Test adaptive intensity segmentation"""
    try:
        from adaptive_intensity_segmenter import AdaptiveIntensitySegmenter
        
        # Create test image and save it
        image = create_test_image()
        test_path = "temp_test_image.png"
        cv2.imwrite(test_path, image)
        
        # Segment image
        segmenter = AdaptiveIntensitySegmenter()
        result = segmenter.segment_image(test_path)
        
        # Clean up
        Path(test_path).unlink(missing_ok=True)
        
        if result['success']:
            print("✓ Adaptive Segmentation: SUCCESS")
            print(f"  Center: {result['center']}")
            print(f"  Core radius: {result['core_radius']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Regions found: {result['regions_found']}")
        else:
            print(f"✗ Adaptive Segmentation: FAILED - {result.get('error', 'Unknown error')}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Adaptive Segmentation: IMPORT ERROR - {e}")
        return False
    except Exception as e:
        print(f"✗ Adaptive Segmentation: ERROR - {e}")
        return False

def main():
    """Run simple usage examples"""
    print("=" * 60)
    print("SIMPLE USAGE EXAMPLES FOR MODULAR FUNCTIONS")
    print("=" * 60)
    
    tests = [
        test_image_enhancement,
        test_bright_core_extraction,
        test_adaptive_segmentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
        print()
    
    print("=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All basic tests passed!")
    elif passed > 0:
        print("⚠️  Some tests passed - modules are partially functional")
    else:
        print("❌ No tests passed - check dependencies and installation")
    print("=" * 60)

if __name__ == "__main__":
    main()
