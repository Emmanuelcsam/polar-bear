#!/usr/bin/env python3
"""
Comprehensive tests for zone_segmentation.py
Tests Hough Circle Transform based zone segmentation.
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

class TestSegmentZones(unittest.TestCase):
    """Test segment_zones function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic fiber optic image with concentric circles
        self.image_size = 512
        self.test_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Draw concentric circles (core, cladding, ferrule)
        center = (self.image_size // 2, self.image_size // 2)
        
        # Ferrule (outermost) - light gray
        cv2.circle(self.test_image, center, 200, (180, 180, 180), -1)
        
        # Cladding - medium gray  
        cv2.circle(self.test_image, center, 125, (120, 120, 120), -1)
        
        # Core (innermost) - dark gray
        cv2.circle(self.test_image, center, 50, (60, 60, 60), -1)
        
        # Create noisy version
        noise = np.random.randint(-10, 10, self.test_image.shape, dtype=np.int16)
        self.noisy_image = np.clip(self.test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    def test_basic_segmentation(self):
        """Test basic zone segmentation on synthetic image."""
        from zone_segmentation import segment_zones
        
        # Segment the perfect synthetic image
        result = segment_zones(self.test_image)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('zones', result)
        self.assertIn('masks', result)
        self.assertIn('visualization', result)
        
        # Check zones detected
        zones = result['zones']
        self.assertIn('core', zones)
        self.assertIn('cladding', zones)
        self.assertIn('ferrule', zones)
        
        # Check zone properties
        for zone_name in ['core', 'cladding', 'ferrule']:
            zone = zones[zone_name]
            self.assertIn('center', zone)
            self.assertIn('radius', zone)
            self.assertIsInstance(zone['center'], tuple)
            self.assertIsInstance(zone['radius'], (int, float))
            self.assertGreater(zone['radius'], 0)
    
    def test_zone_masks(self):
        """Test zone mask generation."""
        from zone_segmentation import segment_zones
        
        result = segment_zones(self.test_image)
        masks = result['masks']
        
        # Check mask properties
        self.assertIn('core', masks)
        self.assertIn('cladding', masks)
        self.assertIn('ferrule', masks)
        
        for zone_name, mask in masks.items():
            # Check mask shape
            self.assertEqual(mask.shape, (self.image_size, self.image_size))
            self.assertEqual(mask.dtype, np.uint8)
            
            # Check mask values (binary)
            unique_values = np.unique(mask)
            self.assertTrue(all(v in [0, 255] for v in unique_values))
            
            # Check that mask has non-zero pixels
            self.assertGreater(np.sum(mask), 0)
    
    def test_zone_hierarchy(self):
        """Test that zones follow correct hierarchy (core < cladding < ferrule)."""
        from zone_segmentation import segment_zones
        
        result = segment_zones(self.test_image)
        zones = result['zones']
        
        # Check radius hierarchy
        core_radius = zones['core']['radius']
        cladding_radius = zones['cladding']['radius']
        ferrule_radius = zones['ferrule']['radius']
        
        self.assertLess(core_radius, cladding_radius)
        self.assertLess(cladding_radius, ferrule_radius)
        
        # Check approximate radii (with some tolerance)
        self.assertAlmostEqual(core_radius, 50, delta=10)
        self.assertAlmostEqual(cladding_radius, 125, delta=10)
        self.assertAlmostEqual(ferrule_radius, 200, delta=10)
    
    def test_noisy_image_segmentation(self):
        """Test segmentation on noisy image."""
        from zone_segmentation import segment_zones
        
        result = segment_zones(self.noisy_image)
        
        # Should still detect zones despite noise
        self.assertIn('zones', result)
        zones = result['zones']
        
        # Check all zones detected
        self.assertIn('core', zones)
        self.assertIn('cladding', zones)
        self.assertIn('ferrule', zones)
        
        # Radii should be approximately correct despite noise
        self.assertAlmostEqual(zones['core']['radius'], 50, delta=15)
        self.assertAlmostEqual(zones['cladding']['radius'], 125, delta=15)
        self.assertAlmostEqual(zones['ferrule']['radius'], 200, delta=15)
    
    def test_visualization_output(self):
        """Test visualization image generation."""
        from zone_segmentation import segment_zones
        
        result = segment_zones(self.test_image)
        visualization = result['visualization']
        
        # Check visualization properties
        self.assertEqual(visualization.shape, self.test_image.shape)
        self.assertEqual(visualization.dtype, np.uint8)
        
        # Visualization should be different from input (has overlays)
        self.assertFalse(np.array_equal(visualization, self.test_image))
        
        # Should contain color overlays (check for non-grayscale pixels)
        # Assuming overlays add color
        if len(visualization.shape) == 3:
            r, g, b = visualization[:,:,0], visualization[:,:,1], visualization[:,:,2]
            has_color = not np.array_equal(r, g) or not np.array_equal(g, b)
            self.assertTrue(has_color, "Visualization should have color overlays")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        from zone_segmentation import segment_zones
        
        # Test with single channel image
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        result = segment_zones(gray_image)
        self.assertIsInstance(result, dict)
        
        # Test with very small image
        small_image = cv2.resize(self.test_image, (64, 64))
        result = segment_zones(small_image)
        # May or may not detect all circles in small image
        self.assertIsInstance(result, dict)
        
        # Test with no circles (blank image)
        blank_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result = segment_zones(blank_image)
        # Should handle gracefully
        self.assertIsInstance(result, dict)
        # May have empty or partial zones
    
    def test_parameter_sensitivity(self):
        """Test segmentation with different parameters."""
        from zone_segmentation import segment_zones
        
        # Test with different parameters (if function accepts them)
        # This tests assumes the function might accept optional parameters
        
        # Create image with different circle sizes
        test_img = np.zeros((512, 512, 3), dtype=np.uint8)
        center = (256, 256)
        cv2.circle(test_img, center, 150, (180, 180, 180), -1)
        cv2.circle(test_img, center, 100, (120, 120, 120), -1)
        cv2.circle(test_img, center, 30, (60, 60, 60), -1)
        
        result = segment_zones(test_img)
        
        # Should still detect three zones
        if 'zones' in result:
            zones = result['zones']
            if all(z in zones for z in ['core', 'cladding', 'ferrule']):
                # Check adapted to different sizes
                self.assertAlmostEqual(zones['core']['radius'], 30, delta=10)
                self.assertAlmostEqual(zones['cladding']['radius'], 100, delta=10)
                self.assertAlmostEqual(zones['ferrule']['radius'], 150, delta=10)
    
    @patch('cv2.HoughCircles')
    def test_hough_circles_failure(self, mock_hough):
        """Test handling when Hough circles detection fails."""
        from zone_segmentation import segment_zones
        
        # Mock HoughCircles to return None (no circles found)
        mock_hough.return_value = None
        
        result = segment_zones(self.test_image)
        
        # Should handle gracefully
        self.assertIsInstance(result, dict)
        # May have empty zones or default values
    
    def test_mask_overlap(self):
        """Test that zone masks don't overlap incorrectly."""
        from zone_segmentation import segment_zones
        
        result = segment_zones(self.test_image)
        masks = result['masks']
        
        # Core should be entirely within cladding
        core_mask = masks['core']
        cladding_mask = masks['cladding']
        
        # Where core is True, cladding should also be True
        core_pixels = core_mask > 0
        cladding_at_core = cladding_mask[core_pixels]
        
        # Most core pixels should be within cladding
        overlap_ratio = np.sum(cladding_at_core > 0) / np.sum(core_pixels)
        self.assertGreater(overlap_ratio, 0.9)  # At least 90% overlap
    
    def test_real_world_image_simulation(self):
        """Test with more realistic fiber optic image."""
        # Create more realistic image with gradients and imperfections
        real_image = np.zeros((512, 512, 3), dtype=np.uint8)
        center = (256, 256)
        
        # Add gradient background
        for y in range(512):
            for x in range(512):
                dist = np.sqrt((x - 256)**2 + (y - 256)**2)
                intensity = int(255 - dist / 2)
                real_image[y, x] = max(0, intensity)
        
        # Add circles with soft edges
        for r, color in [(180, 150), (120, 100), (40, 50)]:
            mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.circle(mask, center, r, 255, -1)
            
            # Gaussian blur for soft edges
            mask = cv2.GaussianBlur(mask, (21, 21), 5)
            
            # Apply to image
            for c in range(3):
                real_image[:, :, c] = np.where(mask > 128, color, real_image[:, :, c])
        
        # Add some defects
        cv2.circle(real_image, (300, 300), 5, (0, 0, 0), -1)  # Dark spot
        cv2.line(real_image, (200, 200), (250, 250), (255, 255, 255), 2)  # Scratch
        
        from zone_segmentation import segment_zones
        result = segment_zones(real_image)
        
        # Should still detect zones despite imperfections
        self.assertIn('zones', result)
        self.assertIn('masks', result)

class TestZoneSegmentationUtils(unittest.TestCase):
    """Test utility functions and edge cases."""
    
    def test_empty_image(self):
        """Test with empty/black image."""
        from zone_segmentation import segment_zones
        
        empty_image = np.zeros((256, 256, 3), dtype=np.uint8)
        result = segment_zones(empty_image)
        
        # Should return valid structure even if no zones detected
        self.assertIsInstance(result, dict)
    
    def test_single_circle_detection(self):
        """Test with image containing only one circle."""
        from zone_segmentation import segment_zones
        
        single_circle = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(single_circle, (128, 128), 80, (255, 255, 255), -1)
        
        result = segment_zones(single_circle)
        
        # Should detect at least one zone
        self.assertIsInstance(result, dict)
        if 'zones' in result:
            self.assertGreater(len(result['zones']), 0)
    
    def test_multiple_fiber_images(self):
        """Test with image containing multiple fiber optic cables."""
        from zone_segmentation import segment_zones
        
        multi_fiber = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Draw two fiber optic patterns
        centers = [(150, 150), (350, 350)]
        for center in centers:
            cv2.circle(multi_fiber, center, 80, (180, 180, 180), -1)
            cv2.circle(multi_fiber, center, 50, (120, 120, 120), -1)
            cv2.circle(multi_fiber, center, 20, (60, 60, 60), -1)
        
        result = segment_zones(multi_fiber)
        
        # Function should handle multiple fibers
        # (behavior depends on implementation - might detect largest or fail gracefully)
        self.assertIsInstance(result, dict)

if __name__ == '__main__':
    unittest.main()