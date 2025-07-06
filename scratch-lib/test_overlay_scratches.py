#!/usr/bin/env python3
"""
Comprehensive test suite for overlay_scratches.py
Tests all functions and modules in the script.
"""

import unittest
import numpy as np
import cv2
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib.util

# Import the module to test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to mock the dependency check during import for testing
with patch('builtins.input', return_value='no'):
    with patch('sys.exit'):
        try:
            import overlay_scratches
        except SystemExit:
            pass

class TestDependencyCheck(unittest.TestCase):
    """Test the dependency checking and installation functionality."""
    
    def test_check_dependencies_all_present(self):
        """Test when all dependencies are already installed."""
        # This test assumes cv2 and numpy are installed
        # The function should not prompt for installation
        with patch('builtins.input') as mock_input:
            overlay_scratches.check_and_install_dependencies()
            mock_input.assert_not_called()
    
    def test_check_dependencies_missing(self):
        """Test when dependencies are missing."""
        with patch('importlib.util.find_spec') as mock_find_spec:
            # Simulate missing cv2
            mock_find_spec.side_effect = lambda x: None if x == 'cv2' else MagicMock()
            
            with patch('builtins.input', return_value='no'):
                with patch('sys.exit') as mock_exit:
                    overlay_scratches.check_and_install_dependencies()
                    mock_exit.assert_called_once_with(1)

class TestImageProcessing(unittest.TestCase):
    """Test image processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test images
        # Dark image with some light scratches (simulating BMP scratch file)
        self.scratch_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some scratches (light lines on dark background)
        cv2.line(self.scratch_img, (10, 10), (90, 90), (50, 50, 50), 2)
        cv2.line(self.scratch_img, (20, 80), (80, 20), (60, 60, 60), 1)
        
        # Clean fiber optic image
        self.clean_img = np.ones((100, 100, 3), dtype=np.uint8) * 180
        cv2.circle(self.clean_img, (50, 50), 30, (100, 100, 100), -1)
        cv2.circle(self.clean_img, (50, 50), 20, (200, 200, 200), -1)
        
        # Save test images
        self.scratch_path = os.path.join(self.test_dir, 'test_scratch.bmp')
        self.clean_path = os.path.join(self.test_dir, 'test_clean.png')
        cv2.imwrite(self.scratch_path, self.scratch_img)
        cv2.imwrite(self.clean_path, self.clean_img)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_extract_scratches(self):
        """Test scratch extraction from BMP file."""
        mask, image = overlay_scratches.extract_scratches(self.scratch_path, threshold=30)
        
        # Check that mask is binary
        self.assertTrue(np.all(np.isin(mask, [0, 255])))
        
        # Check dimensions match
        self.assertEqual(mask.shape[:2], self.scratch_img.shape[:2])
        self.assertEqual(image.shape, self.scratch_img.shape)
        
        # Check that some scratches were detected
        self.assertGreater(np.sum(mask), 0)
    
    def test_extract_scratches_invalid_path(self):
        """Test extract_scratches with invalid file path."""
        with self.assertRaises(ValueError):
            overlay_scratches.extract_scratches('nonexistent.bmp')
    
    def test_overlay_scratches(self):
        """Test overlaying scratches onto clean image."""
        mask, scratch_image = overlay_scratches.extract_scratches(self.scratch_path)
        result = overlay_scratches.overlay_scratches(self.clean_path, mask, scratch_image, opacity=0.5)
        
        # Check output dimensions
        self.assertEqual(result.shape, self.clean_img.shape)
        
        # Check that result is different from original (scratches were added)
        self.assertFalse(np.array_equal(result, self.clean_img))
        
        # Check value ranges
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))
    
    def test_overlay_scratches_invalid_background(self):
        """Test overlay_scratches with invalid background path."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        scratch_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            overlay_scratches.overlay_scratches('nonexistent.png', mask, scratch_image)
    
    def test_overlay_scratches_size_mismatch(self):
        """Test overlay with different sized images."""
        # Create different sized mask and scratch image
        large_mask = np.zeros((200, 200), dtype=np.uint8)
        large_scratch = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Should resize automatically
        result = overlay_scratches.overlay_scratches(self.clean_path, large_mask, large_scratch)
        self.assertEqual(result.shape, self.clean_img.shape)

class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality."""
    
    def setUp(self):
        """Set up test directory with multiple files."""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, 'output')
        
        # Create multiple test files
        for i in range(3):
            # BMP files
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.line(img, (0, i*10), (50, i*10+20), (40+i*10, 40+i*10, 40+i*10), 1)
            cv2.imwrite(os.path.join(self.test_dir, f'scratch_{i}.bmp'), img)
            
            # Clean PNG files
            clean = np.ones((50, 50, 3), dtype=np.uint8) * 150
            cv2.circle(clean, (25, 25), 15, (100, 100, 100), -1)
            cv2.imwrite(os.path.join(self.test_dir, f'clean_{i}.png'), clean)
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)
    
    def test_process_batch(self):
        """Test batch processing of multiple files."""
        overlay_scratches.process_batch(self.test_dir, self.output_dir, threshold=30, opacity=0.7)
        
        # Check that output directory was created
        self.assertTrue(os.path.exists(self.output_dir))
        
        # Check that output files were created
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0)
        
        # Should have 3 BMP x 3 clean = 9 output files
        self.assertEqual(len(output_files), 9)

class TestUserInput(unittest.TestCase):
    """Test user input functions."""
    
    def test_get_user_input_default(self):
        """Test get_user_input with default value."""
        with patch('builtins.input', return_value=''):
            result = overlay_scratches.get_user_input('Test prompt', 'default_value')
            self.assertEqual(result, 'default_value')
    
    def test_get_user_input_custom(self):
        """Test get_user_input with custom value."""
        with patch('builtins.input', return_value='custom_value'):
            result = overlay_scratches.get_user_input('Test prompt', 'default_value')
            self.assertEqual(result, 'custom_value')
    
    def test_get_user_input_with_validation(self):
        """Test get_user_input with validation function."""
        def validate_positive(value):
            if int(value) < 0:
                raise ValueError("Must be positive")
        
        with patch('builtins.input', side_effect=['-5', '10']):
            result = overlay_scratches.get_user_input('Test prompt', '5', validate_positive)
            self.assertEqual(result, '10')
    
    def test_validate_threshold_valid(self):
        """Test threshold validation with valid values."""
        # Should not raise exception
        overlay_scratches.validate_threshold('0')
        overlay_scratches.validate_threshold('128')
        overlay_scratches.validate_threshold('255')
    
    def test_validate_threshold_invalid(self):
        """Test threshold validation with invalid values."""
        with self.assertRaises(ValueError):
            overlay_scratches.validate_threshold('-1')
        
        with self.assertRaises(ValueError):
            overlay_scratches.validate_threshold('256')
        
        with self.assertRaises(ValueError):
            overlay_scratches.validate_threshold('not_a_number')
    
    def test_validate_opacity_valid(self):
        """Test opacity validation with valid values."""
        # Should not raise exception
        overlay_scratches.validate_opacity('0.0')
        overlay_scratches.validate_opacity('0.5')
        overlay_scratches.validate_opacity('1.0')
    
    def test_validate_opacity_invalid(self):
        """Test opacity validation with invalid values."""
        with self.assertRaises(ValueError):
            overlay_scratches.validate_opacity('-0.1')
        
        with self.assertRaises(ValueError):
            overlay_scratches.validate_opacity('1.1')
        
        with self.assertRaises(ValueError):
            overlay_scratches.validate_opacity('not_a_number')

class TestMainFunction(unittest.TestCase):
    """Test main function and program flow."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        bmp_img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.line(bmp_img, (10, 10), (40, 40), (50, 50, 50), 2)
        cv2.imwrite(os.path.join(self.test_dir, 'test.bmp'), bmp_img)
        
        clean_img = np.ones((50, 50, 3), dtype=np.uint8) * 150
        cv2.imwrite(os.path.join(self.test_dir, 'test_clean.png'), clean_img)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_main_demo_mode(self):
        """Test main function in demo mode."""
        inputs = [
            self.test_dir,  # input directory
            'output',       # output directory
            '30',          # threshold
            '0.7',         # opacity
            'demo',        # mode
            'yes'          # proceed
        ]
        
        with patch('builtins.input', side_effect=inputs):
            with patch('sys.exit'):
                overlay_scratches.main()
        
        # Check that output was created
        output_path = os.path.join(self.test_dir, 'output', 'demo_output.jpg')
        self.assertTrue(os.path.exists(output_path))
    
    def test_main_cancel_demo(self):
        """Test cancelling demo mode."""
        inputs = [
            self.test_dir,  # input directory
            'output',       # output directory
            '30',          # threshold
            '0.7',         # opacity
            'demo',        # mode
            'no'           # don't proceed
        ]
        
        with patch('builtins.input', side_effect=inputs):
            overlay_scratches.main()
        
        # Output should not be created
        output_path = os.path.join(self.test_dir, 'output', 'demo_output.jpg')
        self.assertFalse(os.path.exists(output_path))
    
    def test_main_invalid_directory(self):
        """Test main function with invalid directory."""
        inputs = [
            'nonexistent_directory',  # invalid input directory
        ]
        
        with patch('builtins.input', side_effect=inputs):
            with self.assertRaises(SystemExit):
                overlay_scratches.main()

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_directory(self):
        """Test processing empty directory."""
        empty_dir = tempfile.mkdtemp()
        
        try:
            inputs = [
                empty_dir,     # input directory
                'output',      # output directory
                '30',         # threshold
                '0.7',        # opacity
                'demo',       # mode
            ]
            
            with patch('builtins.input', side_effect=inputs):
                with self.assertRaises(SystemExit):
                    overlay_scratches.main()
        finally:
            shutil.rmtree(empty_dir)
    
    def test_extreme_threshold_values(self):
        """Test with extreme threshold values."""
        test_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        test_path = 'test_extreme.bmp'
        cv2.imwrite(test_path, test_img)
        
        try:
            # Test with threshold 0 (everything is scratch)
            mask, _ = overlay_scratches.extract_scratches(test_path, threshold=0)
            self.assertGreater(np.sum(mask), 0)
            
            # Test with threshold 255 (nothing is scratch)
            mask, _ = overlay_scratches.extract_scratches(test_path, threshold=255)
            self.assertEqual(np.sum(mask), 0)
        finally:
            os.remove(test_path)
    
    def test_extreme_opacity_values(self):
        """Test with extreme opacity values."""
        # Create simple test images
        scratch_mask = np.ones((50, 50), dtype=np.uint8) * 255
        scratch_image = np.ones((50, 50, 3), dtype=np.uint8) * 100
        background = np.ones((50, 50, 3), dtype=np.uint8) * 200
        
        bg_path = 'test_bg.png'
        cv2.imwrite(bg_path, background)
        
        try:
            # Test with opacity 0.0 (no scratch visible)
            result = overlay_scratches.overlay_scratches(bg_path, scratch_mask, scratch_image, opacity=0.0)
            np.testing.assert_array_almost_equal(result, background)
            
            # Test with opacity 1.0 (full scratch)
            result = overlay_scratches.overlay_scratches(bg_path, scratch_mask, scratch_image, opacity=1.0)
            # Result should be different from background
            self.assertFalse(np.array_equal(result, background))
        finally:
            os.remove(bg_path)

def run_tests():
    """Run all tests and generate report."""
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()