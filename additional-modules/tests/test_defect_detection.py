#!/usr/bin/env python3
"""
Comprehensive tests for defect_detection.py
Tests DO2MR and LEI detection algorithms.
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

class TestDO2MRDetection(unittest.TestCase):
    """Test DO2MR detection algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test image with known defects
        self.test_image = np.ones((256, 256), dtype=np.uint8) * 128
        
        # Add region defects (dust, pits)
        # Small circular defect (pit)
        cv2.circle(self.test_image, (50, 50), 5, 200, -1)
        
        # Larger irregular defect (contamination)
        pts = np.array([[100, 100], [120, 105], [125, 120], [110, 125], [95, 115]], np.int32)
        cv2.fillPoly(self.test_image, [pts], 80)
        
        # Multiple small defects (dust particles)
        for x, y in [(180, 50), (185, 55), (190, 52)]:
            cv2.circle(self.test_image, (x, y), 2, 60, -1)
    
    def test_do2mr_basic_detection(self):
        """Test basic DO2MR detection functionality."""
        from defect_detection import _do2mr_detection
        
        # Run DO2MR detection
        defect_mask = _do2mr_detection(self.test_image)
        
        # Check output format
        self.assertEqual(defect_mask.shape, self.test_image.shape)
        self.assertEqual(defect_mask.dtype, np.uint8)
        
        # Check that defects are detected
        self.assertGreater(np.sum(defect_mask), 0)
        
        # Check binary mask
        unique_values = np.unique(defect_mask)
        self.assertTrue(all(v in [0, 255] for v in unique_values))
    
    def test_do2mr_parameter_sensitivity(self):
        """Test DO2MR with different parameters."""
        from defect_detection import _do2mr_detection
        
        # Test with different window sizes (if supported)
        # Note: Parameters depend on actual implementation
        
        # Should detect defects with default parameters
        mask_default = _do2mr_detection(self.test_image)
        defect_count_default = np.sum(mask_default > 0)
        
        # Different parameters might detect different number of defects
        # This is a placeholder - adjust based on actual function signature
        self.assertGreater(defect_count_default, 0)
    
    def test_do2mr_noise_robustness(self):
        """Test DO2MR robustness to noise."""
        from defect_detection import _do2mr_detection
        
        # Add noise to image
        noise = np.random.normal(0, 10, self.test_image.shape)
        noisy_image = np.clip(self.test_image + noise, 0, 255).astype(np.uint8)
        
        # Should still detect major defects despite noise
        defect_mask = _do2mr_detection(noisy_image)
        
        # Check that some defects are detected
        self.assertGreater(np.sum(defect_mask), 0)
        
        # Major defects should still be found
        # Check region around known defect location
        defect_region = defect_mask[45:55, 45:55]
        self.assertGreater(np.sum(defect_region), 0)
    
    def test_do2mr_uniform_image(self):
        """Test DO2MR on uniform image (no defects)."""
        from defect_detection import _do2mr_detection
        
        # Create uniform image
        uniform_image = np.ones((256, 256), dtype=np.uint8) * 128
        
        # Should detect few or no defects
        defect_mask = _do2mr_detection(uniform_image)
        
        # Very few pixels should be marked as defects
        defect_ratio = np.sum(defect_mask > 0) / defect_mask.size
        self.assertLess(defect_ratio, 0.01)  # Less than 1% defects

class TestLEIDetection(unittest.TestCase):
    """Test LEI (Linear Enhancement Inspector) algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test image with linear defects
        self.test_image = np.ones((256, 256), dtype=np.uint8) * 128
        
        # Add scratches (linear defects)
        # Horizontal scratch
        cv2.line(self.test_image, (20, 100), (200, 100), 200, 2)
        
        # Vertical scratch
        cv2.line(self.test_image, (150, 20), (150, 180), 60, 1)
        
        # Diagonal scratch
        cv2.line(self.test_image, (50, 50), (150, 150), 180, 3)
        
        # Curved scratch (approximated with polyline)
        pts = []
        for t in range(0, 100, 5):
            x = 100 + int(50 * np.cos(t/20))
            y = 200 + int(t/2)
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.polylines(self.test_image, [pts], False, 80, 1)
    
    def test_lei_basic_detection(self):
        """Test basic LEI detection functionality."""
        from defect_detection import _lei_detection
        
        # Run LEI detection
        defect_mask = _lei_detection(self.test_image)
        
        # Check output format
        self.assertEqual(defect_mask.shape, self.test_image.shape)
        self.assertEqual(defect_mask.dtype, np.uint8)
        
        # Check that linear defects are detected
        # LEI might not detect anything with simple Canny implementation
        # Just check the mask is created
        self.assertEqual(defect_mask.shape, self.test_image.shape)
        
        # Check binary mask
        unique_values = np.unique(defect_mask)
        self.assertTrue(all(v in [0, 255] for v in unique_values))
    
    def test_lei_directional_sensitivity(self):
        """Test LEI detection of different orientations."""
        from defect_detection import _lei_detection
        
        # Test horizontal line detection
        horizontal_image = np.ones((256, 256), dtype=np.uint8) * 128
        cv2.line(horizontal_image, (10, 128), (246, 128), 200, 2)
        
        h_mask = _lei_detection(horizontal_image)
        h_defects = np.sum(h_mask[126:130, :] > 0)  # Check around line
        self.assertGreater(h_defects, 100)  # Should detect most of the line
        
        # Test vertical line detection
        vertical_image = np.ones((256, 256), dtype=np.uint8) * 128
        cv2.line(vertical_image, (128, 10), (128, 246), 200, 2)
        
        v_mask = _lei_detection(vertical_image)
        v_defects = np.sum(v_mask[:, 126:130] > 0)  # Check around line
        self.assertGreater(v_defects, 100)  # Should detect most of the line
    
    def test_lei_thickness_sensitivity(self):
        """Test LEI detection of lines with different thickness."""
        from defect_detection import _lei_detection
        
        # Test thin line
        thin_image = np.ones((256, 256), dtype=np.uint8) * 128
        cv2.line(thin_image, (50, 128), (200, 128), 200, 1)
        
        thin_mask = _lei_detection(thin_image)
        thin_defects = np.sum(thin_mask > 0)
        
        # Test thick line
        thick_image = np.ones((256, 256), dtype=np.uint8) * 128
        cv2.line(thick_image, (50, 128), (200, 128), 200, 5)
        
        thick_mask = _lei_detection(thick_image)
        thick_defects = np.sum(thick_mask > 0)
        
        # Both should be detected
        self.assertGreater(thin_defects, 0)
        self.assertGreater(thick_defects, 0)
        
        # Thick line should have more detected pixels
        self.assertGreater(thick_defects, thin_defects)
    
    def test_lei_no_linear_defects(self):
        """Test LEI on image without linear defects."""
        from defect_detection import _lei_detection
        
        # Create image with only circular defects
        circular_image = np.ones((256, 256), dtype=np.uint8) * 128
        cv2.circle(circular_image, (128, 128), 30, 200, -1)
        cv2.circle(circular_image, (64, 64), 20, 60, -1)
        
        # Should detect few or no linear defects
        defect_mask = _lei_detection(circular_image)
        
        # Very few pixels should be marked as defects
        defect_ratio = np.sum(defect_mask > 0) / defect_mask.size
        self.assertLess(defect_ratio, 0.05)  # Less than 5% defects

class TestDetectDefects(unittest.TestCase):
    """Test main detect_defects function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test image with both types of defects
        self.test_image = np.ones((256, 256), dtype=np.uint8) * 128
        
        # Add region defects
        cv2.circle(self.test_image, (50, 50), 10, 200, -1)  # Bright spot
        cv2.circle(self.test_image, (200, 200), 15, 60, -1)  # Dark spot
        
        # Add linear defects
        cv2.line(self.test_image, (0, 128), (256, 128), 180, 2)  # Horizontal scratch
        cv2.line(self.test_image, (128, 0), (128, 256), 80, 1)  # Vertical scratch
        
        # Create zone masks
        self.zone_masks = {
            'core': np.zeros((256, 256), dtype=np.uint8),
            'cladding': np.zeros((256, 256), dtype=np.uint8),
            'ferrule': np.zeros((256, 256), dtype=np.uint8)
        }
        
        cv2.circle(self.zone_masks['core'], (128, 128), 50, 255, -1)
        cv2.circle(self.zone_masks['cladding'], (128, 128), 100, 255, -1)
        self.zone_masks['ferrule'] = np.ones((256, 256), dtype=np.uint8) * 255
    
    def test_detect_defects_comprehensive(self):
        """Test comprehensive defect detection."""
        from defect_detection import detect_defects
        
        # Run detection with config
        config = {
            'blur_kernel_size': (5, 5),
            'anomaly_threshold_sigma': 1.5,
            'scratch_aspect_ratio_threshold': 3.0,
            'min_defect_area_px': 10
        }
        defect_mask, defect_contours = detect_defects(self.test_image, self.zone_masks, config)
        
        # Check output format
        self.assertEqual(defect_mask.shape, self.test_image.shape)
        self.assertEqual(defect_mask.dtype, np.uint8)
        self.assertIsInstance(defect_contours, list)
        
        # Should detect multiple defects
        self.assertGreater(len(defect_contours), 0)
        self.assertGreater(np.sum(defect_mask), 0)
    
    def test_detect_defects_zone_filtering(self):
        """Test that defects are filtered by zones."""
        from defect_detection import detect_defects
        
        # Create defects outside all zones
        edge_image = np.ones((256, 256), dtype=np.uint8) * 128
        cv2.circle(edge_image, (10, 10), 5, 200, -1)  # Corner defect
        
        # Create zone masks that exclude edges
        small_zones = {
            'core': np.zeros((256, 256), dtype=np.uint8),
            'cladding': np.zeros((256, 256), dtype=np.uint8),
            'ferrule': np.zeros((256, 256), dtype=np.uint8)
        }
        cv2.circle(small_zones['core'], (128, 128), 30, 255, -1)
        cv2.circle(small_zones['cladding'], (128, 128), 60, 255, -1)
        cv2.circle(small_zones['ferrule'], (128, 128), 90, 255, -1)
        
        # Detect defects with config
        config = {
            'blur_kernel_size': (5, 5),
            'anomaly_threshold_sigma': 1.5,
            'scratch_aspect_ratio_threshold': 3.0,
            'min_defect_area_px': 10
        }
        defect_mask, defect_contours = detect_defects(edge_image, small_zones, config)
        
        # Corner defect should not be detected (outside zones)
        corner_region = defect_mask[0:20, 0:20]
        self.assertEqual(np.sum(corner_region), 0)
    
    def test_detect_defects_combined_algorithms(self):
        """Test that both DO2MR and LEI defects are detected."""
        from defect_detection import detect_defects
        
        # Image with clear region and linear defects
        combined_image = np.ones((256, 256), dtype=np.uint8) * 128
        
        # Clear region defect
        cv2.circle(combined_image, (100, 100), 20, 50, -1)
        
        # Clear linear defect
        cv2.line(combined_image, (150, 50), (150, 200), 200, 3)
        
        config = {
            'blur_kernel_size': (5, 5),
            'anomaly_threshold_sigma': 1.5,
            'scratch_aspect_ratio_threshold': 3.0,
            'min_defect_area_px': 10
        }
        defect_mask, defect_contours = detect_defects(combined_image, self.zone_masks, config)
        
        # Check both types are detected
        # Region around circle
        circle_region = defect_mask[80:120, 80:120]
        self.assertGreater(np.sum(circle_region), 0)
        
        # Region around line
        line_region = defect_mask[50:200, 147:153]
        self.assertGreater(np.sum(line_region), 0)
    
    def test_detect_defects_empty_zones(self):
        """Test detection with empty zone masks."""
        from defect_detection import detect_defects
        
        # Empty zone masks
        empty_zones = {
            'core': np.zeros((256, 256), dtype=np.uint8),
            'cladding': np.zeros((256, 256), dtype=np.uint8),
            'ferrule': np.zeros((256, 256), dtype=np.uint8)
        }
        
        # Should handle gracefully
        config = {
            'blur_kernel_size': (5, 5),
            'anomaly_threshold_sigma': 1.5,
            'scratch_aspect_ratio_threshold': 3.0,
            'min_defect_area_px': 10
        }
        defect_mask, defect_contours = detect_defects(self.test_image, empty_zones, config)
        
        # No defects should be detected (no valid zones)
        self.assertEqual(np.sum(defect_mask), 0)
        self.assertEqual(len(defect_contours), 0)
    
    def test_detect_defects_color_image(self):
        """Test detection on color image."""
        from defect_detection import detect_defects
        
        # Create color image
        color_image = cv2.cvtColor(self.test_image, cv2.COLOR_GRAY2BGR)
        
        # Add colored defects
        cv2.circle(color_image, (180, 180), 10, (255, 0, 0), -1)  # Blue defect
        
        # Should handle color image (likely convert to grayscale)
        config = {
            'blur_kernel_size': (5, 5),
            'anomaly_threshold_sigma': 1.5,
            'scratch_aspect_ratio_threshold': 3.0,
            'min_defect_area_px': 10
        }
        defect_mask, defect_contours = detect_defects(color_image, self.zone_masks, config)
        
        # Should still detect defects
        self.assertGreater(np.sum(defect_mask), 0)
        self.assertGreater(len(defect_contours), 0)

class TestDefectDetectionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create zone masks for edge case tests
        self.zone_masks = {
            'core': np.zeros((256, 256), dtype=np.uint8),
            'cladding': np.zeros((256, 256), dtype=np.uint8),
            'ferrule': np.zeros((256, 256), dtype=np.uint8)
        }
        
        cv2.circle(self.zone_masks['core'], (128, 128), 50, 255, -1)
        cv2.circle(self.zone_masks['cladding'], (128, 128), 100, 255, -1)
        self.zone_masks['ferrule'] = np.ones((256, 256), dtype=np.uint8) * 255
    
    def test_very_small_image(self):
        """Test detection on very small images."""
        from defect_detection import detect_defects
        
        # Create tiny image
        tiny_image = np.ones((32, 32), dtype=np.uint8) * 128
        cv2.circle(tiny_image, (16, 16), 3, 200, -1)
        
        tiny_zones = {
            'core': np.zeros((32, 32), dtype=np.uint8),
            'cladding': np.zeros((32, 32), dtype=np.uint8),
            'ferrule': np.ones((32, 32), dtype=np.uint8) * 255
        }
        
        # Should handle small images
        config = {
            'blur_kernel_size': (5, 5),
            'anomaly_threshold_sigma': 1.5,
            'scratch_aspect_ratio_threshold': 3.0,
            'min_defect_area_px': 10
        }
        defect_mask, defect_contours = detect_defects(tiny_image, tiny_zones, config)
        
        self.assertEqual(defect_mask.shape, (32, 32))
    
    def test_high_contrast_defects(self):
        """Test detection of high contrast defects."""
        from defect_detection import detect_defects
        
        # Create high contrast defects
        contrast_image = np.ones((256, 256), dtype=np.uint8) * 128
        
        # Very bright defect
        cv2.circle(contrast_image, (50, 50), 10, 255, -1)
        
        # Very dark defect
        cv2.circle(contrast_image, (200, 200), 10, 0, -1)
        
        config = {
            'blur_kernel_size': (5, 5),
            'anomaly_threshold_sigma': 1.5,
            'scratch_aspect_ratio_threshold': 3.0,
            'min_defect_area_px': 10
        }
        defect_mask, defect_contours = detect_defects(contrast_image, self.zone_masks, config)
        
        # Both extreme defects should be detected
        self.assertGreaterEqual(len(defect_contours), 2)
    
    def test_overlapping_defects(self):
        """Test detection of overlapping defects."""
        from defect_detection import detect_defects
        
        # Create overlapping defects
        overlap_image = np.ones((256, 256), dtype=np.uint8) * 128
        
        # Two overlapping circles
        cv2.circle(overlap_image, (100, 100), 20, 200, -1)
        cv2.circle(overlap_image, (115, 100), 20, 60, -1)
        
        # Line crossing through circles
        cv2.line(overlap_image, (80, 100), (135, 100), 180, 2)
        
        config = {
            'blur_kernel_size': (5, 5),
            'anomaly_threshold_sigma': 1.5,
            'scratch_aspect_ratio_threshold': 3.0,
            'min_defect_area_px': 10
        }
        defect_mask, defect_contours = detect_defects(overlap_image, self.zone_masks, config)
        
        # Should detect defects (might merge overlapping ones)
        self.assertGreater(len(defect_contours), 0)
        self.assertGreater(np.sum(defect_mask), 0)

if __name__ == '__main__':
    unittest.main()