#!/usr/bin/env python3
"""
Master Test Script for Modular Functions
Tests all the extracted modular functions to ensure they work correctly.
"""

import os
import sys
import cv2
import numpy as np
import json
import logging
from pathlib import Path

# Add the modular_functions directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all our modular functions
try:
    from numpy_json_encoder import NumpyEncoder, save_numpy_data, dumps_numpy
    from image_processing_transforms import ImageProcessor
    from hough_circle_segmenter import HoughCircleSegmenter
    from advanced_feature_extractor import AdvancedFeatureExtractor
    from defect_cluster_analyzer import DefectClusterAnalyzer
    from anomaly_detector import AnomalyDetector
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def create_test_image(size=(300, 300)):
    """Create a synthetic test image with various features."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    
    # Background
    img.fill(150)
    
    # Add circles (simulating fiber core/cladding)
    cv2.circle(img, (size[0]//2, size[1]//2), size[0]//3, (200, 200, 200), -1)  # Cladding
    cv2.circle(img, (size[0]//2, size[1]//2), size[0]//8, (250, 250, 250), -1)  # Core
    
    # Add some defects
    cv2.circle(img, (100, 100), 8, (50, 50, 50), -1)  # Dark spot
    cv2.line(img, (50, 200), (150, 250), (60, 60, 60), 2)  # Scratch
    cv2.ellipse(img, (220, 80), (15, 8), 0, 0, 360, (80, 80, 80), -1)  # Contamination
    
    # Add noise
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def test_numpy_encoder():
    """Test the NumpyEncoder module."""
    print("\n" + "="*50)
    print("Testing NumpyEncoder...")
    
    try:
        test_data = {
            'array': np.array([1, 2, 3, 4, 5]),
            'matrix': np.array([[1, 2], [3, 4]]),
            'scalar': np.float32(3.14),
            'regular': 'test_string'
        }
        
        # Test JSON serialization
        json_str = dumps_numpy(test_data)
        reconstructed = json.loads(json_str)
        
        print("✓ JSON serialization successful")
        print(f"  Original array type: {type(test_data['array'])}")
        print(f"  Reconstructed type: {type(reconstructed['array'])}")
        
        # Test file saving
        save_numpy_data(test_data, "test_numpy_output.json")
        print("✓ File saving successful")
        
        return True
        
    except Exception as e:
        print(f"✗ NumpyEncoder test failed: {e}")
        return False


def test_image_processor():
    """Test the ImageProcessor module."""
    print("\n" + "="*50)
    print("Testing ImageProcessor...")
    
    try:
        # Create test image
        test_img = create_test_image()
        cv2.imwrite("test_input.png", test_img)
        
        # Test memory-only processing
        processor = ImageProcessor(save_intermediate=False)
        results = processor.process_image_comprehensive("test_input.png")
        
        print(f"✓ Generated {len(results)} processed variations")
        print(f"  Available transforms: {list(results.keys())[:5]}...")
        
        # Test specific transform suites
        thresh_results = processor.apply_thresholding_suite(test_img)
        print(f"✓ Thresholding suite: {len(thresh_results)} variations")
        
        filter_results = processor.apply_filtering_suite(test_img)
        print(f"✓ Filtering suite: {len(filter_results)} variations")
        
        edge_results = processor.apply_edge_detection_suite(test_img)
        print(f"✓ Edge detection suite: {len(edge_results)} variations")
        
        return True
        
    except Exception as e:
        print(f"✗ ImageProcessor test failed: {e}")
        return False


def test_hough_segmenter():
    """Test the HoughCircleSegmenter module."""
    print("\n" + "="*50)
    print("Testing HoughCircleSegmenter...")
    
    try:
        # Create segmenter
        segmenter = HoughCircleSegmenter(
            min_circle_ratio=0.1,
            max_circle_ratio=0.8,
            core_to_cladding_ratio=0.2
        )
        
        # Test with synthetic fiber image
        test_img = create_test_image((400, 400))
        cv2.imwrite("test_fiber.png", test_img)
        
        # Perform segmentation
        result = segmenter.segment_from_file("test_fiber.png")
        
        if result.get('success'):
            print("✓ Segmentation successful")
            print(f"  Method: {result.get('method')}")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            print(f"  Center: {result.get('center')}")
            print(f"  Core radius: {result.get('core_radius', 0):.1f}")
            print(f"  Cladding radius: {result.get('cladding_radius', 0):.1f}")
            
            if 'masks' in result:
                print(f"  Generated masks: {list(result['masks'].keys())}")
        else:
            print(f"✗ Segmentation failed: {result.get('error')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ HoughCircleSegmenter test failed: {e}")
        return False


def test_feature_extractor():
    """Test the AdvancedFeatureExtractor module."""
    print("\n" + "="*50)
    print("Testing AdvancedFeatureExtractor...")
    
    try:
        # Create extractor
        extractor = AdvancedFeatureExtractor()
        
        # Test with synthetic image
        test_img = create_test_image((200, 200))
        
        # Extract all features
        features = extractor.extract_all_features(test_img)
        
        print(f"✓ Extracted {len(features)} features")
        
        # Test individual feature types
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        stat_features = extractor.extract_statistical_features(gray)
        print(f"  Statistical features: {len(stat_features)}")
        
        lbp_features = extractor.extract_lbp_features(gray)
        print(f"  LBP features: {len(lbp_features)}")
        
        fourier_features = extractor.extract_fourier_features(gray)
        print(f"  Fourier features: {len(fourier_features)}")
        
        matrix_features = extractor.extract_matrix_norms(gray)
        print(f"  Matrix norm features: {len(matrix_features)}")
        
        # Save features
        save_numpy_data(features, "test_features.json")
        print("✓ Features saved successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ AdvancedFeatureExtractor test failed: {e}")
        return False


def test_defect_analyzer():
    """Test the DefectClusterAnalyzer module."""
    print("\n" + "="*50)
    print("Testing DefectClusterAnalyzer...")
    
    try:
        # Create analyzer
        analyzer = DefectClusterAnalyzer(
            clustering_eps=50.0,
            min_cluster_size=2,
            image_shape=(400, 400)
        )
        
        # Create synthetic defects
        synthetic_defects = []
        
        # Cluster 1
        for i in range(8):
            x = np.random.normal(100, 15)
            y = np.random.normal(100, 15)
            defect = {
                'location_xy': [x, y],
                'defect_type': 'SCRATCH',
                'severity': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'confidence': np.random.uniform(0.5, 0.9),
                'area_px': np.random.randint(5, 50)
            }
            synthetic_defects.append(defect)
        
        # Cluster 2
        for i in range(6):
            x = np.random.normal(300, 10)
            y = np.random.normal(300, 10)
            defect = {
                'location_xy': [x, y],
                'defect_type': 'DIG',
                'severity': np.random.choice(['MEDIUM', 'HIGH']),
                'confidence': np.random.uniform(0.6, 0.9),
                'area_px': np.random.randint(10, 80)
            }
            synthetic_defects.append(defect)
        
        # Add scattered defects
        for i in range(4):
            x = np.random.uniform(50, 350)
            y = np.random.uniform(50, 350)
            defect = {
                'location_xy': [x, y],
                'defect_type': 'CONTAMINATION',
                'severity': 'LOW',
                'confidence': np.random.uniform(0.3, 0.7),
                'area_px': np.random.randint(1, 20)
            }
            synthetic_defects.append(defect)
        
        # Add defects to analyzer
        added_count = analyzer.add_defects_from_list(synthetic_defects)
        print(f"✓ Added {added_count} synthetic defects")
        
        # Perform clustering
        clusters = analyzer.cluster_defects()
        print(f"✓ Generated {len(clusters)} clusters")
        
        # Generate analysis
        spatial_analysis = analyzer.analyze_spatial_distribution()
        print(f"✓ Spatial analysis complete")
        print(f"  Center of mass: {spatial_analysis['center_of_mass']}")
        
        report = analyzer.generate_summary_report()
        print(f"✓ Summary report generated")
        print(f"  Total defects: {report['summary']['total_defects']}")
        print(f"  Clustering efficiency: {report['summary']['clustering_efficiency']:.2%}")
        
        # Create heatmap
        heatmap = analyzer.create_defect_heatmap()
        print(f"✓ Heatmap created: {heatmap.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ DefectClusterAnalyzer test failed: {e}")
        return False


def test_anomaly_detector():
    """Test the AnomalyDetector module."""
    print("\n" + "="*50)
    print("Testing AnomalyDetector...")
    
    try:
        # Create detector
        detector = AnomalyDetector(
            blackhat_threshold=25,
            morphology_kernel_size=15,
            min_defect_area=10,
            max_defect_area=2000
        )
        
        # Create test image with defects
        test_img = create_test_image((300, 300))
        
        # Add more obvious defects
        cv2.circle(test_img, (80, 80), 12, (30, 30, 30), -1)  # Large dark spot
        cv2.circle(test_img, (220, 220), 6, (40, 40, 40), -1)  # Small dark spot
        cv2.line(test_img, (50, 150), (150, 200), (50, 50, 50), 3)  # Thick scratch
        
        # Test comprehensive detection
        results = detector.comprehensive_anomaly_detection(test_img)
        
        if 'error' not in results:
            summary = results['detection_summary']
            print("✓ Comprehensive detection successful")
            print(f"  Anomaly regions: {summary['total_anomaly_regions']}")
            print(f"  Scratches: {summary['total_scratches']}")
            print(f"  Digs: {summary['total_digs']}")
            print(f"  Blobs: {summary['total_blobs']}")
            print(f"  Total defects: {summary['total_defects']}")
            print(f"  Defect density: {summary['defect_density']:.6f}")
            
            # Test individual detection methods
            defect_regions = detector.detect_defect_regions(test_img)
            print(f"✓ Defect regions: {len(defect_regions)}")
            
            scratches = detector.detect_scratches(test_img)
            print(f"✓ Scratches detected: {len(scratches)}")
            
            digs = detector.detect_digs(test_img)
            print(f"✓ Digs detected: {len(digs)}")
            
            blobs = detector.detect_blobs(test_img)
            print(f"✓ Blobs detected: {len(blobs)}")
            
            # Test inpainting
            inpainted, mask = detector.detect_and_inpaint_anomalies(test_img)
            print(f"✓ Inpainting successful, mask pixels: {np.sum(mask > 0)}")
            
        else:
            print(f"✗ Detection failed: {results['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ AnomalyDetector test failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files created during testing."""
    test_files = [
        "test_numpy_output.json",
        "test_input.png",
        "test_fiber.png",
        "test_features.json",
        "synthetic_defects.png",
        "synthetic_fiber.png",
        "synthetic_feature_test.png"
    ]
    
    for filename in test_files:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except:
            pass


def main():
    """Run all tests."""
    print("="*60)
    print("MODULAR FUNCTIONS COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during testing
    
    # Track test results
    test_results = {}
    
    # Run all tests
    tests = [
        ("NumpyEncoder", test_numpy_encoder),
        ("ImageProcessor", test_image_processor),
        ("HoughCircleSegmenter", test_hough_segmenter),
        ("AdvancedFeatureExtractor", test_feature_extractor),
        ("DefectClusterAnalyzer", test_defect_analyzer),
        ("AnomalyDetector", test_anomaly_detector),
    ]
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            test_results[test_name] = success
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            test_results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, success in test_results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("-" * 40)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All modular functions are working correctly!")
        print("The legacy codebase has been successfully modularized.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the error messages above.")
    
    # Cleanup
    print("\nCleaning up test files...")
    cleanup_test_files()
    print("✓ Cleanup complete")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
