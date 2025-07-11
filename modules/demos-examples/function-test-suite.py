#!/usr/bin/env python3
"""
Comprehensive Test Runner for Modular Functions
Tests all the extracted modular functions with sample data.
"""

import cv2
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback
import time

# Add current directory to path (we're already in modular_functions)
sys.path.append(str(Path(__file__).parent))

# Import our modular functions
try:
    from adaptive_intensity_segmenter import AdaptiveIntensitySegmenter
    from bright_core_extractor import BrightCoreExtractor
    from hough_fiber_separator import HoughFiberSeparator
    from gradient_fiber_segmenter import GradientFiberSegmenter
    from traditional_defect_detector import TraditionalDefectDetector
    from image_enhancer import ImageEnhancer
    print("✓ All modular functions imported successfully")
except ImportError as e:
    print(f"✗ Error importing modular functions: {e}")
    sys.exit(1)


class ModularFunctionTester:
    """
    Comprehensive tester for all modular functions
    """
    
    def __init__(self, test_image_path: Optional[str] = None, output_dir: str = "test_results"):
        """
        Initialize the tester
        
        Args:
            test_image_path: Path to test image (will create synthetic if None)
            output_dir: Directory to save test results
        """
        self.test_image_path = test_image_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each module
        self.module_dirs = {}
        modules = [
            'adaptive_intensity', 'bright_core', 'hough_separation',
            'gradient_segmentation', 'defect_detection', 'image_enhancement'
        ]
        for module in modules:
            module_dir = self.output_dir / module
            module_dir.mkdir(exist_ok=True)
            self.module_dirs[module] = module_dir
        
        # Test results
        self.results = {}
    
    def create_synthetic_test_image(self) -> np.ndarray:
        """
        Create a synthetic fiber optic test image
        
        Returns:
            Synthetic test image
        """
        print("Creating synthetic test image...")
        
        # Image dimensions
        width, height = 512, 512
        center_x, center_y = width // 2, height // 2
        
        # Create base image
        image = np.ones((height, width), dtype=np.uint8) * 30  # Dark background
        
        # Add ferrule (outer bright ring)
        ferrule_radius = 200
        cv2.circle(image, (center_x, center_y), ferrule_radius, 120, -1)
        
        # Add cladding
        cladding_radius = 150
        cv2.circle(image, (center_x, center_y), cladding_radius, 180, -1)
        
        # Add core (bright center)
        core_radius = 25
        cv2.circle(image, (center_x, center_y), core_radius, 250, -1)
        
        # Add some noise
        noise = np.random.normal(0, 10, (height, width))
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Add some defects
        # Scratch
        cv2.line(image, (center_x - 40, center_y - 10), (center_x + 40, center_y + 10), 80, 2)
        
        # Pit
        cv2.circle(image, (center_x + 30, center_y - 30), 5, 50, -1)
        
        # Contamination
        cv2.ellipse(image, (center_x - 50, center_y + 40), (15, 8), 30, 0, 360, 100, -1)
        
        # Save synthetic image
        synthetic_path = self.output_dir / "synthetic_test_image.png"
        cv2.imwrite(str(synthetic_path), image)
        print(f"✓ Synthetic test image saved: {synthetic_path}")
        
        return image
    
    def load_test_image(self) -> np.ndarray:
        """
        Load test image (real or synthetic)
        
        Returns:
            Test image
        """
        if self.test_image_path and Path(self.test_image_path).exists():
            print(f"Loading test image: {self.test_image_path}")
            image = cv2.imread(self.test_image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                return image
            else:
                print(f"✗ Could not load image: {self.test_image_path}")
        
        # Create synthetic image
        return self.create_synthetic_test_image()
    
    def test_adaptive_intensity_segmenter(self, image: np.ndarray) -> Dict[str, Any]:
        """Test adaptive intensity segmentation"""
        print("\n" + "="*50)
        print("Testing Adaptive Intensity Segmenter")
        print("="*50)
        
        result = {'module': 'adaptive_intensity_segmenter', 'success': False}
        
        try:
            # Save test image temporarily
            temp_image_path = self.module_dirs['adaptive_intensity'] / "test_input.png"
            cv2.imwrite(str(temp_image_path), image)
            
            # Test with default parameters
            segmenter = AdaptiveIntensitySegmenter()
            seg_result = segmenter.segment_image(str(temp_image_path), str(self.module_dirs['adaptive_intensity']))
            
            result.update(seg_result)
            result['success'] = seg_result.get('success', False)
            
            print(f"✓ Success: {result['success']}")
            if result['success']:
                print(f"  Center: {result.get('center')}")
                print(f"  Core radius: {result.get('core_radius')}")
                print(f"  Cladding radius: {result.get('cladding_radius')}")
                print(f"  Confidence: {result.get('confidence')}")
                print(f"  Regions found: {result.get('regions_found')}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            print(f"✗ Exception: {e}")
        
        return result
    
    def test_bright_core_extractor(self, image: np.ndarray) -> Dict[str, Any]:
        """Test bright core extraction"""
        print("\n" + "="*50)
        print("Testing Bright Core Extractor")
        print("="*50)
        
        result = {'module': 'bright_core_extractor', 'success': False}
        
        try:
            # Save test image temporarily
            temp_image_path = self.module_dirs['bright_core'] / "test_input.png"
            cv2.imwrite(str(temp_image_path), image)
            
            # Test with default parameters
            extractor = BrightCoreExtractor()
            extract_result = extractor.extract_bright_core(str(temp_image_path), debug_mode=True)
            
            result.update(extract_result)
            result['success'] = extract_result.get('success', False)
            
            print(f"✓ Success: {result['success']}")
            if result['success']:
                print(f"  Center: {result.get('center')}")
                print(f"  Core radius: {result.get('core_radius')}")
                print(f"  Confidence: {result.get('confidence')}")
                print(f"  Circles detected: {result.get('circles_detected')}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            print(f"✗ Exception: {e}")
        
        return result
    
    def test_hough_fiber_separator(self, image: np.ndarray) -> Dict[str, Any]:
        """Test Hough fiber separation"""
        print("\n" + "="*50)
        print("Testing Hough Fiber Separator")
        print("="*50)
        
        result = {'module': 'hough_fiber_separator', 'success': False}
        
        try:
            # Save test image temporarily
            temp_image_path = self.module_dirs['hough_separation'] / "test_input.png"
            cv2.imwrite(str(temp_image_path), image)
            
            # Test with default parameters
            separator = HoughFiberSeparator()
            sep_result = separator.separate_fiber(str(temp_image_path), str(self.module_dirs['hough_separation']))
            
            result.update(sep_result)
            result['success'] = sep_result.get('success', False)
            
            print(f"✓ Success: {result['success']}")
            if result['success']:
                print(f"  Center: {result.get('center')}")
                print(f"  Core radius: {result.get('core_radius')}")
                print(f"  Cladding radius: {result.get('cladding_radius')}")
                print(f"  Confidence: {result.get('confidence')}")
                print(f"  Core detected: {result.get('core_detected')}")
                print(f"  Cladding detected: {result.get('cladding_detected')}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            print(f"✗ Exception: {e}")
        
        return result
    
    def test_gradient_fiber_segmenter(self, image: np.ndarray) -> Dict[str, Any]:
        """Test gradient-based fiber segmentation"""
        print("\n" + "="*50)
        print("Testing Gradient Fiber Segmenter")
        print("="*50)
        
        result = {'module': 'gradient_fiber_segmenter', 'success': False}
        
        try:
            # Save test image temporarily
            temp_image_path = self.module_dirs['gradient_segmentation'] / "test_input.png"
            cv2.imwrite(str(temp_image_path), image)
            
            # Test with default parameters
            segmenter = GradientFiberSegmenter()
            seg_result = segmenter.segment_fiber(str(temp_image_path), str(self.module_dirs['gradient_segmentation']))
            
            result.update(seg_result)
            result['success'] = seg_result.get('success', False)
            
            print(f"✓ Success: {result['success']}")
            if result['success']:
                print(f"  Center: {result.get('center')}")
                print(f"  Core radius: {result.get('core_radius')}")
                print(f"  Cladding radius: {result.get('cladding_radius')}")
                print(f"  Confidence: {result.get('confidence')}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            print(f"✗ Exception: {e}")
        
        return result
    
    def test_traditional_defect_detector(self, image: np.ndarray) -> Dict[str, Any]:
        """Test traditional defect detection"""
        print("\n" + "="*50)
        print("Testing Traditional Defect Detector")
        print("="*50)
        
        result = {'module': 'traditional_defect_detector', 'success': False}
        
        try:
            # Create detector
            detector = TraditionalDefectDetector()
            
            # Use synthetic fiber parameters for testing
            h, w = image.shape
            center = (w // 2, h // 2)
            core_radius = 25
            cladding_radius = 150
            
            # Create zone masks
            zone_masks = detector.create_zone_masks(image.shape, center, core_radius, cladding_radius)
            
            # Convert to BGR for defect detection
            if len(image.shape) == 2:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                bgr_image = image
            
            # Detect defects
            all_defects = detector.detect_all_defects(bgr_image, zone_masks)
            
            # Count total defects
            total_defects = sum(len(defects) for defects in all_defects.values())
            
            result['success'] = True
            result['total_defects'] = total_defects
            result['defects_by_zone'] = {}
            
            for zone_name, defects in all_defects.items():
                result['defects_by_zone'][zone_name] = {
                    'count': len(defects),
                    'types': list(set(d.type for d in defects)) if defects else []
                }
            
            print(f"✓ Success: {result['success']}")
            print(f"  Total defects: {total_defects}")
            for zone_name, zone_info in result['defects_by_zone'].items():
                if zone_info['count'] > 0:
                    print(f"  {zone_name}: {zone_info['count']} defects ({', '.join(zone_info['types'])})")
            
            # Save visualization
            vis_image = bgr_image.copy()
            colors = {
                'scratch': (0, 255, 255),
                'pit': (255, 0, 0),
                'contamination': (0, 165, 255),
                'crack': (0, 0, 255)
            }
            
            for zone_name, defects in all_defects.items():
                for defect in defects:
                    color = colors.get(defect.type, (255, 255, 255))
                    x, y, w, h = defect.bbox
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(vis_image, f"{defect.type}", (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            vis_path = self.module_dirs['defect_detection'] / "defects_visualization.png"
            cv2.imwrite(str(vis_path), vis_image)
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            print(f"✗ Exception: {e}")
        
        return result
    
    def test_image_enhancer(self, image: np.ndarray) -> Dict[str, Any]:
        """Test image enhancement"""
        print("\n" + "="*50)
        print("Testing Image Enhancer")
        print("="*50)
        
        result = {'module': 'image_enhancer', 'success': False}
        
        try:
            # Create enhancer
            enhancer = ImageEnhancer()
            
            # Test auto enhancement
            auto_enhanced = enhancer.auto_enhance(image)
            
            # Test specific enhancement steps
            enhancement_steps = ['convert_grayscale', 'clahe', 'bilateral_filter', 'unsharp_masking']
            step_results = enhancer.enhance_image(image, enhancement_steps)
            
            result['success'] = True
            result['auto_enhancement_completed'] = auto_enhanced is not None
            result['step_enhancement_completed'] = len(step_results) > 0
            result['enhancement_steps_tested'] = enhancement_steps
            
            print(f"✓ Success: {result['success']}")
            print(f"  Auto enhancement: {'✓' if result['auto_enhancement_completed'] else '✗'}")
            print(f"  Step enhancement: {'✓' if result['step_enhancement_completed'] else '✗'}")
            print(f"  Steps tested: {len(enhancement_steps)}")
            
            # Save enhanced images
            auto_path = self.module_dirs['image_enhancement'] / "auto_enhanced.png"
            cv2.imwrite(str(auto_path), auto_enhanced)
            
            for step_name, step_image in step_results.items():
                if step_name != 'original':
                    step_path = self.module_dirs['image_enhancement'] / f"{step_name}.png"
                    cv2.imwrite(str(step_path), step_image)
            
            # Calculate enhancement statistics
            if len(image.shape) == 3:
                orig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = image
            
            if len(auto_enhanced.shape) == 3:
                enhanced_gray = cv2.cvtColor(auto_enhanced, cv2.COLOR_BGR2GRAY)
            else:
                enhanced_gray = auto_enhanced
            
            result['statistics'] = {
                'original_mean': float(np.mean(orig_gray)),
                'original_std': float(np.std(orig_gray)),
                'enhanced_mean': float(np.mean(enhanced_gray)),
                'enhanced_std': float(np.std(enhanced_gray)),
                'contrast_improvement': float(np.std(enhanced_gray) / np.std(orig_gray))
            }
            
            print(f"  Contrast improvement: {result['statistics']['contrast_improvement']:.2f}x")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            print(f"✗ Exception: {e}")
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all modular function tests
        
        Returns:
            Comprehensive test results
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE MODULAR FUNCTION TESTING")
        print("="*70)
        
        # Load test image
        test_image = self.load_test_image()
        
        # Run all tests
        test_methods = [
            ('adaptive_intensity', self.test_adaptive_intensity_segmenter),
            ('bright_core', self.test_bright_core_extractor),
            ('hough_separation', self.test_hough_fiber_separator),
            ('gradient_segmentation', self.test_gradient_fiber_segmenter),
            ('defect_detection', self.test_traditional_defect_detector),
            ('image_enhancement', self.test_image_enhancer)
        ]
        
        start_time = time.time()
        
        for test_name, test_method in test_methods:
            try:
                self.results[test_name] = test_method(test_image)
            except Exception as e:
                self.results[test_name] = {
                    'module': test_name,
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                print(f"✗ {test_name} failed: {e}")
        
        total_time = time.time() - start_time
        
        # Generate summary
        successful_tests = sum(1 for result in self.results.values() if result.get('success', False))
        total_tests = len(self.results)
        
        summary = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'success_rate': successful_tests / total_tests * 100,
            'total_time_seconds': total_time,
            'test_results': self.results
        }
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Total time: {total_time:.2f} seconds")
        
        # Save comprehensive results
        results_path = self.output_dir / "comprehensive_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        print(f"\nComprehensive results saved to: {results_path}")
        
        return summary


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Comprehensive Modular Function Tester')
    parser.add_argument('--test-image', help='Path to test image (synthetic will be created if not provided)')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for test results')
    parser.add_argument('--specific-test', choices=[
        'adaptive_intensity', 'bright_core', 'hough_separation',
        'gradient_segmentation', 'defect_detection', 'image_enhancement'
    ], help='Run only a specific test')
    
    args = parser.parse_args()
    
    # Create tester
    tester = ModularFunctionTester(test_image_path=args.test_image, output_dir=args.output_dir)
    
    if args.specific_test:
        # Run specific test
        test_image = tester.load_test_image()
        
        test_methods = {
            'adaptive_intensity': tester.test_adaptive_intensity_segmenter,
            'bright_core': tester.test_bright_core_extractor,
            'hough_separation': tester.test_hough_fiber_separator,
            'gradient_segmentation': tester.test_gradient_fiber_segmenter,
            'defect_detection': tester.test_traditional_defect_detector,
            'image_enhancement': tester.test_image_enhancer
        }
        
        if args.specific_test in test_methods:
            result = test_methods[args.specific_test](test_image)
            print(f"\nTest result: {'✓ SUCCESS' if result.get('success') else '✗ FAILED'}")
        else:
            print(f"Unknown test: {args.specific_test}")
    else:
        # Run all tests
        summary = tester.run_all_tests()
        
        if summary['success_rate'] == 100:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        elif summary['success_rate'] >= 80:
            print("\n✅ Most tests passed - good job!")
        else:
            print("\n⚠️  Some tests failed - check the results for details")


if __name__ == "__main__":
    main()
