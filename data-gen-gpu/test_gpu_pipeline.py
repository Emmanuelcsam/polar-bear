#!/usr/bin/env python3
"""
Unit tests for GPU-accelerated fiber optic analysis pipeline
Tests both GPU and CPU modes to ensure consistency
"""

import unittest
import numpy as np
import cv2
import tempfile
import os
import json
import shutil
from pathlib import Path
import logging

# Import GPU modules
from gpu_utils import GPUManager, gpu_accelerated, GPUImageProcessor
from process_gpu import ImageProcessorGPU
from separation_gpu import UnifiedSeparationGPU, ConsensusSystemGPU
from detection_gpu import OmniFiberAnalyzerGPU, OmniConfigGPU
from data_acquisition_gpu import DataAcquisitionGPU, AggregatedDefect

# Suppress logs during testing
logging.getLogger().setLevel(logging.WARNING)


class TestGPUUtils(unittest.TestCase):
    """Test GPU utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gpu_manager_gpu = GPUManager(force_cpu=False)
        self.gpu_manager_cpu = GPUManager(force_cpu=True)
        self.test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_gpu_manager_initialization(self):
        """Test GPU manager initialization"""
        # CPU mode should always work
        self.assertFalse(self.gpu_manager_cpu.use_gpu)
        
        # GPU mode depends on availability
        if self.gpu_manager_gpu.use_gpu:
            self.assertIsNotNone(self.gpu_manager_gpu.device)
    
    def test_array_transfer(self):
        """Test array transfer between CPU and GPU"""
        # Test CPU mode
        cpu_array = self.gpu_manager_cpu.array_to_gpu(self.test_array)
        self.assertIsInstance(cpu_array, np.ndarray)
        self.assertTrue(np.array_equal(cpu_array, self.test_array))
        
        # Test GPU mode (if available)
        if self.gpu_manager_gpu.use_gpu:
            gpu_array = self.gpu_manager_gpu.array_to_gpu(self.test_array)
            self.assertNotIsInstance(gpu_array, np.ndarray)  # Should be CuPy array
            
            # Transfer back
            cpu_result = self.gpu_manager_gpu.array_to_cpu(gpu_array)
            self.assertIsInstance(cpu_result, np.ndarray)
            self.assertTrue(np.array_equal(cpu_result, self.test_array))
    
    def test_gpu_accelerated_decorator(self):
        """Test GPU acceleration decorator"""
        class TestClass:
            def __init__(self):
                self.gpu_manager = GPUManager(force_cpu=True)
            
            @gpu_accelerated
            def test_function(self, array):
                xp = self.gpu_manager.get_array_module(array)
                return xp.mean(array)
        
        obj = TestClass()
        result = obj.test_function(self.test_array)
        expected = np.mean(self.test_array)
        self.assertAlmostEqual(float(result), float(expected), places=5)


class TestImageProcessorGPU(unittest.TestCase):
    """Test GPU-accelerated image processing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor_gpu = ImageProcessorGPU({}, force_cpu=False)
        self.processor_cpu = ImageProcessorGPU({}, force_cpu=True)
        
        # Create test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(self.test_image, (50, 50), 30, (255, 255, 255), -1)
        
        # Save test image
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test.png")
        cv2.imwrite(self.test_image_path, self.test_image)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.temp_dir)
    
    def test_process_single_image(self):
        """Test single image processing"""
        # Process with CPU
        cpu_results = self.processor_cpu.process_single_image(
            self.test_image_path, 
            self.temp_dir,
            return_arrays=True
        )
        
        # Should have multiple variations
        self.assertIsInstance(cpu_results, dict)
        self.assertGreater(len(cpu_results), 10)
        
        # Check some expected keys
        expected_keys = ['test_grayscale', 'test_gaussian_3', 'test_sobel', 'test_otsu']
        for key in expected_keys:
            self.assertIn(key, cpu_results)
            self.assertIsInstance(cpu_results[key], np.ndarray)
    
    def test_color_space_conversions(self):
        """Test color space conversion functions"""
        # Test RGB to grayscale
        gray = self.processor_cpu._rgb_to_grayscale_gpu(self.test_image)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape[:2], self.test_image.shape[:2])
        
        # Test RGB to HSV
        hsv = self.processor_cpu._rgb_to_hsv_gpu(self.test_image)
        self.assertEqual(hsv.shape, self.test_image.shape)
        
        # Test RGB to LAB
        lab = self.processor_cpu._rgb_to_lab_gpu(self.test_image)
        self.assertEqual(lab.shape, self.test_image.shape)
    
    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU modes produce similar results"""
        if not self.processor_gpu.gpu_manager.use_gpu:
            self.skipTest("GPU not available")
        
        # Process with both modes
        cpu_results = self.processor_cpu.process_single_image(
            self.test_image_path, self.temp_dir, return_arrays=True
        )
        
        gpu_results = self.processor_gpu.process_single_image(
            self.test_image_path, self.temp_dir, return_arrays=True
        )
        
        # Compare keys
        self.assertEqual(set(cpu_results.keys()), set(gpu_results.keys()))
        
        # Compare some results (allowing for small differences)
        for key in ['test_grayscale', 'test_gaussian_3']:
            if key in cpu_results:
                cpu_arr = cpu_results[key]
                gpu_arr = gpu_results[key]
                
                # Calculate mean absolute difference
                diff = np.mean(np.abs(cpu_arr.astype(float) - gpu_arr.astype(float)))
                self.assertLess(diff, 5.0)  # Allow small differences


class TestSeparationGPU(unittest.TestCase):
    """Test GPU-accelerated separation module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.separator_cpu = UnifiedSeparationGPU({}, force_cpu=True)
        
        # Create test image with clear regions
        self.test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Core (bright center)
        cv2.circle(self.test_image, (100, 100), 30, (255, 255, 255), -1)
        # Cladding (medium brightness)
        cv2.circle(self.test_image, (100, 100), 60, (128, 128, 128), -1)
        cv2.circle(self.test_image, (100, 100), 30, (255, 255, 255), -1)
        # Ferrule (dark outer region) - already zeros
        
        # Save test image
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_fiber.png")
        cv2.imwrite(self.test_image_path, self.test_image)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.temp_dir)
    
    def test_separation_result_structure(self):
        """Test that separation returns expected structure"""
        # Mock a simple separation (skip actual processing for unit test)
        gpu_manager = GPUManager(force_cpu=True)
        consensus_system = ConsensusSystemGPU(gpu_manager)
        
        # Create mock masks
        h, w = 200, 200
        center = (100, 100)
        core_radius = 30
        cladding_radius = 60
        
        masks = consensus_system._generate_masks_gpu(
            np.array(center), core_radius, cladding_radius, (h, w)
        )
        
        # Check mask structure
        self.assertIn('core', masks)
        self.assertIn('cladding', masks)
        self.assertIn('ferrule', masks)
        
        # Check mask properties
        for mask_name, mask in masks.items():
            self.assertEqual(mask.shape, (h, w))
            self.assertIn(mask.dtype, [np.uint8, np.bool_])
    
    def test_anomaly_detection(self):
        """Test anomaly detection and inpainting"""
        # Add some defects to test image
        defect_image = self.test_image.copy()
        cv2.circle(defect_image, (50, 50), 5, (0, 0, 0), -1)  # Black spot
        cv2.circle(defect_image, (150, 150), 3, (255, 0, 0), -1)  # Blue spot
        
        # Test anomaly detection
        inpainted, defect_mask = self.separator_cpu._detect_and_inpaint_anomalies_gpu(defect_image)
        
        # Check outputs
        self.assertEqual(inpainted.shape, defect_image.shape)
        self.assertEqual(defect_mask.shape[:2], defect_image.shape[:2])
        
        # Defect mask should have some non-zero values where defects were
        self.assertGreater(np.sum(defect_mask), 0)


class TestDetectionGPU(unittest.TestCase):
    """Test GPU-accelerated detection module"""
    
    def setUp(self):
        """Set up test fixtures"""
        config = OmniConfigGPU(
            min_defect_size=5,
            max_defect_size=1000,
            anomaly_threshold_multiplier=2.0
        )
        self.detector_cpu = OmniFiberAnalyzerGPU(config, force_cpu=True)
        
        # Create test regions
        self.test_regions = {
            'core': np.zeros((100, 100, 3), dtype=np.uint8),
            'cladding': np.zeros((100, 100, 3), dtype=np.uint8),
            'ferrule': np.zeros((100, 100, 3), dtype=np.uint8)
        }
        
        # Add normal content
        cv2.circle(self.test_regions['core'], (50, 50), 30, (200, 200, 200), -1)
        cv2.circle(self.test_regions['cladding'], (50, 50), 40, (150, 150, 150), -1)
        
        # Add defects
        cv2.circle(self.test_regions['core'], (30, 30), 5, (50, 50, 50), -1)  # Dark spot
        cv2.rectangle(self.test_regions['cladding'], (60, 60), (70, 70), (250, 250, 250), -1)  # Bright spot
    
    def test_analyze_regions(self):
        """Test region analysis"""
        result = self.detector_cpu.analyze_regions(self.test_regions)
        
        # Check result structure
        self.assertIsNotNone(result.core_result)
        self.assertIsNotNone(result.cladding_result)
        self.assertIsNotNone(result.ferrule_result)
        
        # Check quality scores
        self.assertGreaterEqual(result.overall_quality, 0)
        self.assertLessEqual(result.overall_quality, 100)
        
        # Check defect detection
        self.assertIsInstance(result.total_defects, int)
        self.assertGreaterEqual(result.total_defects, 0)
    
    def test_feature_extraction(self):
        """Test feature extraction functions"""
        test_region = self.test_regions['core']
        
        # Extract features
        features = self.detector_cpu._extract_features_gpu(test_region, 'core')
        
        # Check that features were extracted
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check some expected features
        expected_features = ['mean_intensity', 'std_intensity', 'gradient_mean']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
    
    def test_empty_region_handling(self):
        """Test handling of empty regions"""
        empty_region = np.zeros((100, 100, 3), dtype=np.uint8)
        empty_regions = {
            'core': empty_region,
            'cladding': empty_region,
            'ferrule': empty_region
        }
        
        result = self.detector_cpu.analyze_regions(empty_regions)
        
        # Should handle empty regions gracefully
        self.assertEqual(result.total_defects, 0)
        self.assertEqual(result.overall_quality, 100.0)


class TestDataAcquisitionGPU(unittest.TestCase):
    """Test GPU-accelerated data acquisition"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.acquisitor_cpu = DataAcquisitionGPU({}, force_cpu=True)
        
        # Create test defects
        self.test_defects = [
            {
                'location': (50, 50),
                'size': 25,
                'severity': 'HIGH',
                'confidence': 0.8,
                'region': 'core',
                'source_method': 'method1',
                'features': {'intensity': 100}
            },
            {
                'location': (52, 48),  # Close to first (should cluster)
                'size': 20,
                'severity': 'MEDIUM',
                'confidence': 0.7,
                'region': 'core',
                'source_method': 'method2',
                'features': {'intensity': 95}
            },
            {
                'location': (100, 100),  # Far from others
                'size': 30,
                'severity': 'LOW',
                'confidence': 0.6,
                'region': 'cladding',
                'source_method': 'method1',
                'features': {'intensity': 80}
            }
        ]
        
        # Create test image
        self.test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(self.test_image, (100, 100), 80, (128, 128, 128), -1)
    
    def test_defect_clustering(self):
        """Test defect clustering"""
        aggregated = self.acquisitor_cpu._cluster_defects_gpu(self.test_defects)
        
        # Should have fewer aggregated defects than original
        self.assertLessEqual(len(aggregated), len(self.test_defects))
        
        # Check aggregated defect properties
        for defect in aggregated:
            self.assertIsInstance(defect, AggregatedDefect)
            self.assertIsInstance(defect.center, tuple)
            self.assertGreater(defect.size, 0)
            self.assertIn(defect.severity, ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE'])
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation"""
        # Create aggregated defects
        aggregated = self.acquisitor_cpu._cluster_defects_gpu(self.test_defects)
        
        # Calculate metrics
        metrics = self.acquisitor_cpu._calculate_quality_metrics(aggregated)
        
        # Check metric structure
        self.assertIn('total_defects', metrics)
        self.assertIn('quality_score', metrics)
        self.assertIn('core_defects', metrics)
        
        # Check metric values
        self.assertEqual(metrics['total_defects'], len(aggregated))
        self.assertGreaterEqual(metrics['quality_score'], 0)
        self.assertLessEqual(metrics['quality_score'], 100)
    
    def test_visualization_generation(self):
        """Test visualization generation"""
        # Create aggregated defects
        aggregated = self.acquisitor_cpu._cluster_defects_gpu(self.test_defects)
        
        # Generate visualizations
        visualizations = self.acquisitor_cpu._generate_visualizations_gpu(
            self.test_image, aggregated, None
        )
        
        # Check visualizations
        self.assertIsInstance(visualizations, dict)
        self.assertGreater(len(visualizations), 0)
        
        # Check visualization properties
        for name, viz in visualizations.items():
            self.assertIsInstance(viz, np.ndarray)
            self.assertEqual(viz.shape[:2], self.test_image.shape[:2])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create realistic test image
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        # Add fiber structure
        cv2.circle(self.test_image, (150, 150), 40, (255, 255, 255), -1)  # Core
        cv2.circle(self.test_image, (150, 150), 80, (180, 180, 180), -1)  # Cladding
        cv2.circle(self.test_image, (150, 150), 40, (255, 255, 255), -1)  # Redraw core
        
        # Add some defects
        cv2.circle(self.test_image, (130, 130), 3, (50, 50, 50), -1)  # Dark spot in core
        cv2.rectangle(self.test_image, (180, 180), (185, 185), (250, 250, 250), -1)  # Bright spot
        
        self.test_image_path = os.path.join(self.temp_dir, "test_integration.png")
        cv2.imwrite(self.test_image_path, self.test_image)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.temp_dir)
    
    def test_cpu_pipeline_integration(self):
        """Test complete pipeline in CPU mode"""
        from app_gpu import FiberAnalysisPipelineGPU
        
        # Create pipeline in CPU mode
        pipeline = FiberAnalysisPipelineGPU(force_cpu=True)
        
        # Run analysis
        output_dir = os.path.join(self.temp_dir, "output")
        summary = pipeline.analyze_image(self.test_image_path, output_dir)
        
        # Check summary structure
        self.assertIn('quality_score', summary)
        self.assertIn('pass_fail_status', summary)
        self.assertIn('total_defects', summary)
        self.assertIn('processing_time', summary)
        
        # Check that files were created
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "analysis_summary.json")))
        
        # Check quality score is reasonable
        self.assertGreaterEqual(summary['quality_score'], 0)
        self.assertLessEqual(summary['quality_score'], 100)
    
    def test_memory_optimization(self):
        """Test that memory optimization works (passing arrays instead of files)"""
        # Initialize modules
        processor = ImageProcessorGPU({}, force_cpu=True)
        separator = UnifiedSeparationGPU({}, force_cpu=True)
        
        # Process image and return arrays
        processed_arrays = processor.process_single_image(
            self.test_image_path,
            self.temp_dir,
            return_arrays=True
        )
        
        # Check that arrays were returned
        self.assertIsInstance(processed_arrays, dict)
        self.assertGreater(len(processed_arrays), 0)
        
        # Arrays should be numpy arrays
        for name, array in processed_arrays.items():
            self.assertIsInstance(array, np.ndarray)


def run_performance_comparison():
    """Run performance comparison between GPU and CPU modes"""
    print("\n" + "="*60)
    print("Performance Comparison: GPU vs CPU")
    print("="*60)
    
    # Create test image
    test_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    temp_dir = tempfile.mkdtemp()
    test_path = os.path.join(temp_dir, "perf_test.png")
    cv2.imwrite(test_path, test_image)
    
    # Test image processing
    print("\n1. Image Processing Performance:")
    
    # CPU timing
    import time
    processor_cpu = ImageProcessorGPU({}, force_cpu=True)
    
    start = time.time()
    cpu_results = processor_cpu.process_single_image(test_path, temp_dir, return_arrays=True)
    cpu_time = time.time() - start
    print(f"   CPU Time: {cpu_time:.3f}s")
    
    # GPU timing (if available)
    processor_gpu = ImageProcessorGPU({}, force_cpu=False)
    if processor_gpu.gpu_manager.use_gpu:
        start = time.time()
        gpu_results = processor_gpu.process_single_image(test_path, temp_dir, return_arrays=True)
        gpu_time = time.time() - start
        print(f"   GPU Time: {gpu_time:.3f}s")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("   GPU not available for comparison")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print("\nPerformance comparison completed.")


if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance comparison
    run_performance_comparison()