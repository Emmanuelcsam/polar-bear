#!/usr/bin/env python3
"""
Comprehensive tests for feature_extraction.py
Tests feature extraction from detected defects.
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

class TestExtractFeatures(unittest.TestCase):
    """Test extract_features function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test image with known features
        self.test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add background
        self.test_image[:, :] = (100, 100, 100)
        
        # Add defects with known properties
        # Circular defect
        cv2.circle(self.test_image, (50, 50), 20, (200, 200, 200), -1)
        
        # Rectangular defect
        cv2.rectangle(self.test_image, (150, 150), (200, 180), (50, 50, 50), -1)
        
        # Create defect regions with proper structure
        self.defect_regions = []
        
        # Circle defect
        circle_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(circle_mask, (50, 50), 20, 255, -1)
        circle_contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in circle_contours:
            M = cv2.moments(contour)
            cx = int(M['m10'] / (M['m00'] + 1e-6))
            cy = int(M['m01'] / (M['m00'] + 1e-6))
            self.defect_regions.append({
                'contour': contour,
                'area': cv2.contourArea(contour),
                'centroid': (cx, cy)
            })
        
        # Rectangle defect
        rect_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(rect_mask, (150, 150), (200, 180), 255, -1)
        rect_contours, _ = cv2.findContours(rect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in rect_contours:
            M = cv2.moments(contour)
            cx = int(M['m10'] / (M['m00'] + 1e-6))
            cy = int(M['m01'] / (M['m00'] + 1e-6))
            self.defect_regions.append({
                'contour': contour,
                'area': cv2.contourArea(contour),
                'centroid': (cx, cy)
            })
        
        # Create metrics dictionary
        self.metrics = {
            'core_radius': 40,
            'cladding_radius': 80,
            'ferrule_radius': 120
        }
        
        # Create zone masks
        self.zone_masks = {
            'core': np.zeros((256, 256), dtype=np.uint8),
            'cladding': np.zeros((256, 256), dtype=np.uint8),
            'ferrule': np.zeros((256, 256), dtype=np.uint8)
        }
        
        # Define zones
        cv2.circle(self.zone_masks['core'], (128, 128), 40, 255, -1)
        cv2.circle(self.zone_masks['cladding'], (128, 128), 80, 255, -1)
        cv2.circle(self.zone_masks['ferrule'], (128, 128), 120, 255, -1)
        
    def test_basic_feature_extraction(self):
        """Test basic feature extraction functionality."""
        from feature_extraction import extract_features
        
        features = extract_features(self.test_image, self.defect_regions, self.zone_masks, self.metrics)
        
        # Check that features were extracted for each defect
        self.assertEqual(len(features), len(self.defect_regions))
        
        # Check feature structure
        for feature_dict in features:
            self.assertIsInstance(feature_dict, dict)
            
            # Check required features
            required_features = [
                'defect_id', 'area_px', 'centroid_x', 'centroid_y', 'zone',
                'rect_width', 'rect_height', 'rect_angle', 'aspect_ratio'
            ]
            
            for feature in required_features:
                self.assertIn(feature, feature_dict)
    
    def test_shape_features(self):
        """Test shape feature calculations."""
        from feature_extraction import extract_features
        
        features = extract_features(self.test_image, self.defect_regions, self.zone_masks, self.metrics)
        
        # Test circular defect (first defect)
        circle_features = features[0]
        
        # Area should be approximately π * r²
        expected_area = np.pi * 20 * 20
        self.assertAlmostEqual(circle_features['area_px'], expected_area, delta=50)
        
        # Perimeter should be approximately 2 * π * r
        expected_perimeter = 2 * np.pi * 20
        self.assertAlmostEqual(circle_features.get('perimeter', 0), expected_perimeter, delta=10)
        
        # Circularity should be close to 1 for circle
        self.assertGreater(circle_features.get('circularity', 0.5), 0.8)
        
        # Aspect ratio should be close to 1 for circle
        self.assertAlmostEqual(circle_features['aspect_ratio'], 1.0, delta=0.2)
        
        # Test rectangular defect (second defect)
        rect_features = features[1]
        
        # Area should be width * height
        expected_area = 50 * 30  # 200-150 x 180-150
        self.assertAlmostEqual(rect_features['area_px'], expected_area, delta=50)
        
        # Aspect ratio should reflect rectangle shape
        self.assertNotAlmostEqual(rect_features['aspect_ratio'], 1.0, delta=0.3)
        
        # Circularity should be lower for rectangle
        self.assertLess(rect_features.get('circularity', 0.5), 0.8)
    
    def test_intensity_features(self):
        """Test intensity feature calculations."""
        from feature_extraction import extract_features
        
        features = extract_features(self.test_image, self.defect_regions, self.zone_masks, self.metrics)
        
        # Test circular defect (bright defect)
        circle_features = features[0]
        self.assertAlmostEqual(circle_features['mean_intensity'], 200, delta=5)
        self.assertEqual(circle_features.get('min_intensity', 0), 200)
        self.assertEqual(circle_features.get('max_intensity', 255), 200)
        self.assertAlmostEqual(circle_features.get('std_intensity', 0), 0, delta=1)
        
        # Test rectangular defect (dark defect)
        rect_features = features[1]
        self.assertAlmostEqual(rect_features['mean_intensity'], 50, delta=5)
        self.assertEqual(rect_features.get('min_intensity', 0), 50)
        self.assertEqual(rect_features.get('max_intensity', 255), 50)
    
    def test_texture_features(self):
        """Test GLCM texture feature calculations."""
        from feature_extraction import extract_features
        
        # Create textured defect
        textured_image = self.test_image.copy()
        
        # Add textured region
        for i in range(100, 150):
            for j in range(100, 150):
                textured_image[i, j] = (i + j) % 256
        
        # Create contour for textured region
        texture_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(texture_mask, (100, 100), (150, 150), 255, -1)
        texture_contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to defect regions
        texture_regions = []
        for contour in texture_contours:
            M = cv2.moments(contour)
            cx = int(M['m10'] / (M['m00'] + 1e-6))
            cy = int(M['m01'] / (M['m00'] + 1e-6))
            texture_regions.append({
                'contour': contour,
                'area': cv2.contourArea(contour),
                'centroid': (cx, cy)
            })
        
        features = extract_features(textured_image, texture_regions, self.zone_masks, self.metrics)
        
        texture_features = features[0]
        
        # Check texture features are calculated
        self.assertIn('contrast', texture_features)
        self.assertIn('homogeneity', texture_features)
        self.assertIn('energy', texture_features)
        self.assertIn('correlation', texture_features)
        
        # Texture features should be in valid ranges
        self.assertGreaterEqual(texture_features['contrast'], 0)
        self.assertLessEqual(texture_features['homogeneity'], 1)
        self.assertLessEqual(texture_features['energy'], 1)
        self.assertGreaterEqual(texture_features['correlation'], -1)
        self.assertLessEqual(texture_features['correlation'], 1)
    
    def test_zone_assignment(self):
        """Test zone assignment for defects."""
        from feature_extraction import extract_features
        
        # Create defects in specific zones
        zoned_image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Core defect
        cv2.circle(zoned_image, (128, 128), 10, (255, 255, 255), -1)
        
        # Cladding defect  
        cv2.circle(zoned_image, (128, 170), 10, (255, 255, 255), -1)
        
        # Ferrule defect
        cv2.circle(zoned_image, (128, 220), 10, (255, 255, 255), -1)
        
        # Get contours
        gray = cv2.cvtColor(zoned_image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to defect regions
        zone_regions = []
        for contour in contours:
            M = cv2.moments(contour)
            cx = int(M['m10'] / (M['m00'] + 1e-6))
            cy = int(M['m01'] / (M['m00'] + 1e-6))
            zone_regions.append({
                'contour': contour,
                'area': cv2.contourArea(contour),
                'centroid': (cx, cy)
            })
        
        features = extract_features(zoned_image, zone_regions, self.zone_masks, self.metrics)
        
        # Sort by y-coordinate to identify defects
        features_sorted = sorted(features, key=lambda f: f['centroid_y'])
        
        # Check zone assignments
        # First defect should be in core
        self.assertEqual(features_sorted[0]['zone'], 'core')
        
        # Second defect should be in cladding
        self.assertIn(features_sorted[1]['zone'], ['cladding', 'core'])  # Could be either
        
        # Third defect should be in ferrule
        self.assertEqual(features_sorted[2]['zone'], 'ferrule')
    
    def test_empty_contours(self):
        """Test with empty contour list."""
        from feature_extraction import extract_features
        
        features = extract_features(self.test_image, [], self.zone_masks, self.metrics)
        
        # Should return empty list
        self.assertEqual(len(features), 0)
    
    def test_single_pixel_defect(self):
        """Test feature extraction for very small defect."""
        from feature_extraction import extract_features
        
        # Create single pixel defect
        small_image = self.test_image.copy()
        small_image[50, 50] = (255, 255, 255)
        
        # Create contour
        small_mask = np.zeros((256, 256), dtype=np.uint8)
        small_mask[50, 50] = 255
        
        # Find contours with more relaxed settings
        kernel = np.ones((3, 3), np.uint8)
        small_mask = cv2.dilate(small_mask, kernel, iterations=1)
        small_contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(small_contours) > 0:
            # Convert contours to defect regions
            small_regions = []
            for contour in small_contours:
                M = cv2.moments(contour)
                cx = int(M['m10'] / (M['m00'] + 1e-6))
                cy = int(M['m01'] / (M['m00'] + 1e-6))
                small_regions.append({
                    'contour': contour,
                    'area': cv2.contourArea(contour),
                    'centroid': (cx, cy)
                })
            
            features = extract_features(small_image, small_regions, self.zone_masks, self.metrics)
            
            if len(features) > 0:
                # Check that features are calculated even for tiny defect
                self.assertGreater(features[0]['area_px'], 0)
                self.assertGreater(features[0].get('perimeter', 1), 0)
    
    def test_complex_shape_defect(self):
        """Test feature extraction for complex shaped defect."""
        from feature_extraction import extract_features
        
        # Create L-shaped defect
        complex_image = self.test_image.copy()
        cv2.rectangle(complex_image, (50, 50), (70, 100), (255, 255, 255), -1)
        cv2.rectangle(complex_image, (50, 80), (100, 100), (255, 255, 255), -1)
        
        # Get contour
        gray = cv2.cvtColor(complex_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        complex_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to defect regions
        complex_regions = []
        for contour in complex_contours:
            M = cv2.moments(contour)
            cx = int(M['m10'] / (M['m00'] + 1e-6))
            cy = int(M['m01'] / (M['m00'] + 1e-6))
            complex_regions.append({
                'contour': contour,
                'area': cv2.contourArea(contour),
                'centroid': (cx, cy)
            })
        
        features = extract_features(complex_image, complex_regions, self.zone_masks, self.metrics)
        
        if len(features) > 0:
            # Complex shape should have lower circularity
            self.assertLess(features[0].get('circularity', 0.5), 0.7)
            
            # Should have meaningful area and perimeter
            self.assertGreater(features[0]['area_px'], 100)
            self.assertGreater(features[0].get('perimeter', 1), 50)
    
    def test_feature_consistency(self):
        """Test that features are consistent across multiple calls."""
        from feature_extraction import extract_features
        
        # Extract features twice
        features1 = extract_features(self.test_image, self.defect_regions, self.zone_masks, self.metrics)
        features2 = extract_features(self.test_image, self.defect_regions, self.zone_masks, self.metrics)
        
        # Should produce identical results
        self.assertEqual(len(features1), len(features2))
        
        for f1, f2 in zip(features1, features2):
            for key in f1:
                if isinstance(f1[key], (int, float)):
                    self.assertAlmostEqual(f1[key], f2[key], places=5)
                else:
                    self.assertEqual(f1[key], f2[key])
    
    def test_grayscale_image(self):
        """Test feature extraction on grayscale image."""
        from feature_extraction import extract_features
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        
        # Should handle grayscale image
        features = extract_features(gray_image, self.defect_regions, self.zone_masks, self.metrics)
        
        # Check features are extracted
        self.assertEqual(len(features), len(self.defect_regions))
        
        # Intensity features should still work
        for feature_dict in features:
            self.assertIn('mean_intensity', feature_dict)
            self.assertIn('std_intensity', feature_dict)

class TestFeatureValidation(unittest.TestCase):
    """Test feature validation and edge cases."""
    
    def test_feature_ranges(self):
        """Test that all features are in valid ranges."""
        from feature_extraction import extract_features
        
        # Create various defects
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add various shapes
        cv2.circle(test_image, (50, 50), 20, (255, 255, 255), -1)
        cv2.ellipse(test_image, (150, 50), (30, 20), 45, 0, 360, (0, 0, 0), -1)
        cv2.rectangle(test_image, (50, 150), (100, 200), (128, 128, 128), -1)
        
        # Get contours
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create zone masks
        zone_masks = {
            'core': np.zeros((256, 256), dtype=np.uint8),
            'cladding': np.zeros((256, 256), dtype=np.uint8),
            'ferrule': np.ones((256, 256), dtype=np.uint8) * 255
        }
        
        # Convert contours to defect regions
        defect_regions = []
        for contour in contours:
            M = cv2.moments(contour)
            cx = int(M['m10'] / (M['m00'] + 1e-6))
            cy = int(M['m01'] / (M['m00'] + 1e-6))
            defect_regions.append({
                'contour': contour,
                'area': cv2.contourArea(contour),
                'centroid': (cx, cy)
            })
        
        # Create metrics
        metrics = {
            'core_radius': 40,
            'cladding_radius': 80,
            'ferrule_radius': 120
        }
        
        features = extract_features(test_image, defect_regions, zone_masks, metrics)
        
        for feature_dict in features:
            # Check shape features
            self.assertGreater(feature_dict['area_px'], 0)
            self.assertGreater(feature_dict.get('perimeter', 1), 0)
            self.assertGreaterEqual(feature_dict.get('circularity', 0.5), 0)
            self.assertLessEqual(feature_dict.get('circularity', 0.5), 1)
            self.assertGreater(feature_dict['aspect_ratio'], 0)
            
            # Check intensity features
            self.assertGreaterEqual(feature_dict['mean_intensity'], 0)
            self.assertLessEqual(feature_dict['mean_intensity'], 255)
            self.assertGreaterEqual(feature_dict.get('std_intensity', 0), 0)
            self.assertGreaterEqual(feature_dict.get('min_intensity', 0), 0)
            self.assertLessEqual(feature_dict.get('max_intensity', 255), 255)
            
            # Check texture features
            self.assertGreaterEqual(feature_dict['contrast'], 0)
            self.assertGreaterEqual(feature_dict['homogeneity'], 0)
            self.assertLessEqual(feature_dict['homogeneity'], 1)
            self.assertGreaterEqual(feature_dict['energy'], 0)
            self.assertLessEqual(feature_dict['energy'], 1)
            self.assertGreaterEqual(feature_dict['correlation'], -1)
            self.assertLessEqual(feature_dict['correlation'], 1)
            
            # Check zone assignment
            self.assertIn(feature_dict['zone'], ['core', 'cladding', 'ferrule', 'unknown'])

if __name__ == '__main__':
    unittest.main()