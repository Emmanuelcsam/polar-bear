#!/usr/bin/env python3
"""
Comprehensive test suite for directory-crop-with-ref.py
Tests all functions and the full program functionality
"""

import unittest
import tempfile
import shutil
import json
import numpy as np
import cv2
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock
import logging

# Add the current directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after modifying path
try:
    # Import with mocked pygame to avoid display requirements
    with patch.dict(sys.modules, {'pygame': MagicMock()}):
        import directory_crop_with_ref as dcr
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure directory-crop-with-ref.py is renamed to directory_crop_with_ref.py")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDirectoryCrop(unittest.TestCase):
    """Test suite for directory crop functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.ref_dir = Path(self.test_dir) / "reference"
        self.target_dir = Path(self.test_dir) / "target"
        self.output_dir = Path(self.test_dir) / "output"
        
        # Create directories
        self.ref_dir.mkdir()
        self.target_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test images
        self.create_test_images()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
        
    def create_test_images(self):
        """Create test images for testing"""
        # Create reference images (red squares on white background)
        for i in range(3):
            img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
            cv2.imwrite(str(self.ref_dir / f"ref_{i}.png"), img)
            
        # Create target images (similar but with variations)
        for i in range(5):
            img = np.ones((250, 250, 3), dtype=np.uint8) * 255
            # Vary position and size slightly
            x1 = 60 + i * 5
            y1 = 60 + i * 5
            x2 = 160 + i * 5
            y2 = 160 + i * 5
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.imwrite(str(self.target_dir / f"target_{i}.png"), img)
            
        # Create a grayscale image
        gray_img = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.rectangle(gray_img, (50, 50), (150, 150), 128, -1)
        cv2.imwrite(str(self.target_dir / "gray.png"), gray_img)
        
        # Create an image with alpha channel
        rgba = np.ones((200, 200, 4), dtype=np.uint8) * 255
        rgba[50:150, 50:150] = [0, 0, 255, 255]
        cv2.imwrite(str(self.target_dir / "alpha.png"), rgba)
        
    def test_validate_directory(self):
        """Test directory validation"""
        # Test valid directory
        valid_dir = dcr.validate_directory(str(self.ref_dir), "Test")
        self.assertEqual(valid_dir, self.ref_dir)
        
        # Test non-existent directory
        with self.assertRaises(ValueError):
            dcr.validate_directory("/non/existent/path", "Test")
            
        # Test file instead of directory
        test_file = Path(self.test_dir) / "test.txt"
        test_file.write_text("test")
        with self.assertRaises(ValueError):
            dcr.validate_directory(str(test_file), "Test")
            
    def test_safe_divide(self):
        """Test safe division function"""
        self.assertEqual(dcr.safe_divide(10, 2), 5.0)
        self.assertEqual(dcr.safe_divide(10, 0), 0.0)
        self.assertEqual(dcr.safe_divide(10, 0, -1), -1)
        
    def test_extract_comprehensive_features(self):
        """Test feature extraction"""
        # Test with valid image
        img_path = str(self.ref_dir / "ref_0.png")
        features = dcr.extract_comprehensive_features(img_path)
        
        # Check basic features exist
        self.assertIn('r_mean', features)
        self.assertIn('g_mean', features)
        self.assertIn('b_mean', features)
        self.assertIn('hsv_h_mean', features)
        self.assertIn('contour_area', features)
        self.assertIn('aspect_ratio', features)
        self.assertIn('entropy', features)
        self.assertIn('edge_density', features)
        
        # Check feature values are reasonable
        # Note: OpenCV uses BGR format, so blue channel should be high for red rectangles
        self.assertGreater(features['b_mean'], 200)  # Blue channel in BGR (red in RGB)
        self.assertLess(features['g_mean'], 50)
        self.assertLess(features['r_mean'], 50)  # Red channel in BGR (blue in RGB)
        
        # Test with grayscale image
        gray_path = str(self.target_dir / "gray.png")
        gray_features = dcr.extract_comprehensive_features(gray_path)
        self.assertIn('r_mean', gray_features)
        
        # Test with alpha channel image
        alpha_path = str(self.target_dir / "alpha.png")
        alpha_features = dcr.extract_comprehensive_features(alpha_path)
        self.assertTrue(alpha_features['has_alpha'])
        
        # Test with non-existent image
        bad_features = dcr.extract_comprehensive_features("/non/existent.png")
        self.assertEqual(bad_features, {})
        
    def test_aggregate_features(self):
        """Test feature aggregation"""
        # Extract features from multiple images
        features_list = []
        for i in range(3):
            img_path = str(self.ref_dir / f"ref_{i}.png")
            features = dcr.extract_comprehensive_features(img_path)
            if features:
                features_list.append(features)
                
        # Aggregate features
        aggregated = dcr.aggregate_features(features_list)
        
        # Check aggregated features
        self.assertIn('r_mean_mean', aggregated)
        self.assertIn('r_mean_std', aggregated)
        self.assertIn('r_mean_min', aggregated)
        self.assertIn('r_mean_max', aggregated)
        
        # Test empty list
        empty_agg = dcr.aggregate_features([])
        self.assertEqual(empty_agg, {})
        
    def test_analyze_reference_directory(self):
        """Test reference directory analysis"""
        # Analyze directory
        ref_features = dcr.analyze_reference_directory(self.ref_dir)
        
        # Check features were extracted
        self.assertIn('num_references', ref_features)
        self.assertEqual(ref_features['num_references'], 3)
        self.assertIn('r_mean_mean', ref_features)
        
        # Check cache file was created
        cache_file = Path('ref_features.json')
        self.assertTrue(cache_file.exists())
        
        # Clean up cache
        cache_file.unlink()
        
        # Test with empty directory
        empty_dir = Path(self.test_dir) / "empty"
        empty_dir.mkdir()
        with self.assertRaises(ValueError):
            dcr.analyze_reference_directory(empty_dir)
            
    def test_generate_mask(self):
        """Test mask generation"""
        # Load test image and extract features
        img = cv2.imread(str(self.target_dir / "target_0.png"))
        ref_features = dcr.analyze_reference_directory(self.ref_dir)
        
        params = {
            'color_multiplier': 1.5,
            'morph_kernel': 5,
            'area_tolerance': 0.5,
            'aspect_tolerance': 0.5
        }
        
        # Generate mask
        mask = dcr.generate_mask(img, ref_features, params)
        
        # Check mask properties
        self.assertEqual(mask.shape, img.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        # Check if mask has any foreground pixels
        self.assertGreater(np.sum(mask > 0), 0)  # Should have some foreground pixels
        
        # Clean up cache
        cache_file = Path('ref_features.json')
        if cache_file.exists():
            cache_file.unlink()
            
    def test_apply_crop(self):
        """Test crop application"""
        # Create test image and mask
        img = cv2.imread(str(self.target_dir / "target_0.png"))
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
        
        # Apply crop
        cropped = dcr.apply_crop(img, mask)
        
        # Check output
        self.assertEqual(cropped.shape[2], 4)  # Should have alpha channel
        self.assertEqual(cropped.shape[:2], img.shape[:2])
        np.testing.assert_array_equal(cropped[:, :, 3], mask)
        
    def test_adjust_parameters(self):
        """Test parameter adjustment"""
        params = {
            'color_multiplier': 1.0,
            'morph_kernel': 5,
            'area_tolerance': 0.5,
            'aspect_tolerance': 0.5
        }
        
        # Test adjustment
        adjusted = dcr.adjust_parameters(params.copy(), 1)
        self.assertGreater(adjusted['color_multiplier'], params['color_multiplier'])
        self.assertGreater(adjusted['area_tolerance'], params['area_tolerance'])
        
        # Test multiple iterations
        for i in range(5):
            adjusted = dcr.adjust_parameters(adjusted, i)
            
        # Check limits
        self.assertLessEqual(adjusted['color_multiplier'], 3.0)
        self.assertLessEqual(adjusted['morph_kernel'], 15)
        
    def test_process_single_image(self):
        """Test single image processing"""
        # Analyze references
        ref_features = dcr.analyze_reference_directory(self.ref_dir)
        
        params = {
            'color_multiplier': 1.5,
            'morph_kernel': 5,
            'area_tolerance': 0.5,
            'aspect_tolerance': 0.5
        }
        
        # Process single image
        img_path = self.target_dir / "target_0.png"
        success = dcr.process_single_image(img_path, self.output_dir, ref_features, params)
        
        # Check result
        self.assertTrue(success)
        output_path = self.output_dir / "target_0_cropped.png"
        self.assertTrue(output_path.exists())
        
        # Check output image
        output_img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        self.assertEqual(output_img.shape[2], 4)  # Has alpha channel
        
        # Clean up cache
        cache_file = Path('ref_features.json')
        if cache_file.exists():
            cache_file.unlink()
            
    def test_process_directory(self):
        """Test directory processing"""
        # Analyze references
        ref_features = dcr.analyze_reference_directory(self.ref_dir)
        
        params = {
            'color_multiplier': 1.5,
            'morph_kernel': 5,
            'area_tolerance': 0.5,
            'aspect_tolerance': 0.5
        }
        
        # Process directory
        dcr.process_directory(self.target_dir, self.output_dir, ref_features, params)
        
        # Check outputs
        output_files = list(self.output_dir.glob("*.png"))
        self.assertGreater(len(output_files), 0)
        
        # Check each output
        for output_file in output_files:
            img = cv2.imread(str(output_file), cv2.IMREAD_UNCHANGED)
            self.assertEqual(img.shape[2], 4)  # Has alpha channel
            
        # Clean up cache
        cache_file = Path('ref_features.json')
        if cache_file.exists():
            cache_file.unlink()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_image(self):
        """Test handling of empty/black images"""
        # Create empty image
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        temp_path = "temp_empty.png"
        cv2.imwrite(temp_path, empty_img)
        
        # Extract features
        features = dcr.extract_comprehensive_features(temp_path)
        
        # Should handle gracefully
        self.assertIsInstance(features, dict)
        
        # Clean up
        os.unlink(temp_path)
        
    def test_tiny_image(self):
        """Test handling of very small images"""
        # Create tiny image
        tiny_img = np.ones((5, 5, 3), dtype=np.uint8) * 255
        temp_path = "temp_tiny.png"
        cv2.imwrite(temp_path, tiny_img)
        
        # Extract features
        features = dcr.extract_comprehensive_features(temp_path)
        
        # Should return empty dict for too small images
        self.assertEqual(features, {})
        
        # Clean up
        os.unlink(temp_path)
        
    def test_huge_image(self):
        """Test handling of very large images"""
        # Create large image (but not too large to avoid memory issues)
        large_img = np.ones((5000, 5000, 3), dtype=np.uint8) * 255
        temp_path = "temp_large.png"
        cv2.imwrite(temp_path, large_img)
        
        # Extract features
        features = dcr.extract_comprehensive_features(temp_path)
        
        # Should handle by resizing
        self.assertIsInstance(features, dict)
        if features:
            self.assertIn('width', features)
            self.assertLessEqual(features['width'], dcr.MAX_IMAGE_SIZE)
            
        # Clean up
        os.unlink(temp_path)
        
    def test_corrupted_features(self):
        """Test handling of corrupted feature data"""
        # Test with NaN values
        features_list = [
            {'value': float('nan')},
            {'value': 1.0},
            {'value': 2.0}
        ]
        
        # Should handle gracefully
        aggregated = dcr.aggregate_features(features_list)
        self.assertIsInstance(aggregated, dict)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full workflow"""
    
    @patch('builtins.input')
    @patch('directory_crop_with_ref.CropPreviewUI')
    def test_full_workflow(self, mock_ui_class, mock_input):
        """Test the complete workflow with mocked UI"""
        # Create test directories
        with tempfile.TemporaryDirectory() as test_dir:
            ref_dir = Path(test_dir) / "ref"
            target_dir = Path(test_dir) / "target"
            output_dir = Path(test_dir) / "output"
            
            ref_dir.mkdir()
            target_dir.mkdir()
            
            # Create test images
            for i in range(2):
                # Reference images
                ref_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                cv2.rectangle(ref_img, (25, 25), (75, 75), (0, 0, 255), -1)
                cv2.imwrite(str(ref_dir / f"ref_{i}.png"), ref_img)
                
                # Target images
                target_img = np.ones((120, 120, 3), dtype=np.uint8) * 255
                cv2.rectangle(target_img, (30, 30), (90, 90), (0, 0, 255), -1)
                cv2.imwrite(str(target_dir / f"target_{i}.png"), target_img)
            
            # Mock user inputs
            mock_input.side_effect = [
                str(ref_dir),
                str(target_dir),
                str(output_dir)
            ]
            
            # Mock UI to auto-confirm
            mock_ui = MagicMock()
            mock_ui.run_preview.return_value = True  # Confirm
            mock_ui_class.return_value = mock_ui
            
            # Run main with patched sys.exit
            with patch('sys.exit'):
                dcr.main()
            
            # Check outputs were created
            output_files = list(output_dir.glob("*.png"))
            self.assertGreater(len(output_files), 0)
            
            # Clean up cache
            cache_file = Path('ref_features.json')
            if cache_file.exists():
                cache_file.unlink()


def run_tests():
    """Run all tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDirectoryCrop))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*60)
    print("Running comprehensive tests for directory-crop-with-ref.py")
    print("="*60)
    
    success = run_tests()
    
    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*60)
    
    sys.exit(0 if success else 1)