#!/usr/bin/env python3
"""
Simple function-by-function tests for overlay_scratches.py
This script tests each function individually with minimal setup.
"""

import numpy as np
import cv2
import os
import tempfile
from pathlib import Path

# Mock the dependency check for testing
import sys
from unittest.mock import patch
with patch('builtins.input', return_value='no'):
    with patch('sys.exit'):
        try:
            import overlay_scratches
        except SystemExit:
            pass

def test_dependency_check():
    """Test 1: Check dependency installation function."""
    print("Test 1: Testing dependency check...")
    try:
        # This should pass if cv2 and numpy are installed
        overlay_scratches.check_and_install_dependencies()
        print("✓ Dependencies are installed")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()

def test_extract_scratches():
    """Test 2: Test scratch extraction from BMP."""
    print("Test 2: Testing scratch extraction...")
    
    # Create a test BMP image
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add some light scratches
    cv2.line(test_img, (20, 20), (80, 80), (60, 60, 60), 2)
    cv2.line(test_img, (30, 70), (70, 30), (50, 50, 50), 1)
    
    # Save temporarily
    temp_path = "test_scratch.bmp"
    cv2.imwrite(temp_path, test_img)
    
    try:
        # Test extraction
        mask, image = overlay_scratches.extract_scratches(temp_path, threshold=40)
        
        # Verify results
        assert mask is not None, "Mask should not be None"
        assert image is not None, "Image should not be None"
        assert mask.shape == (100, 100), f"Mask shape incorrect: {mask.shape}"
        assert image.shape == (100, 100, 3), f"Image shape incorrect: {image.shape}"
        assert np.sum(mask) > 0, "No scratches detected"
        
        print("✓ Scratch extraction works correctly")
        print(f"  - Detected {np.sum(mask > 0)} scratch pixels")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    print()

def test_overlay_scratches_function():
    """Test 3: Test overlay function."""
    print("Test 3: Testing scratch overlay...")
    
    # Create test images
    background = np.ones((100, 100, 3), dtype=np.uint8) * 150
    cv2.circle(background, (50, 50), 30, (100, 100, 100), -1)
    
    scratch_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(scratch_mask, (10, 10), (90, 90), 255, 3)
    
    scratch_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(scratch_image, (10, 10), (90, 90), (80, 80, 80), 3)
    
    # Save background temporarily
    bg_path = "test_background.png"
    cv2.imwrite(bg_path, background)
    
    try:
        # Test overlay
        result = overlay_scratches.overlay_scratches(bg_path, scratch_mask, scratch_image, opacity=0.5)
        
        # Verify results
        assert result is not None, "Result should not be None"
        assert result.shape == background.shape, f"Result shape incorrect: {result.shape}"
        assert not np.array_equal(result, background), "Result should be different from background"
        
        print("✓ Overlay function works correctly")
        print(f"  - Output shape: {result.shape}")
        print(f"  - Pixels modified: {np.sum(result != background) // 3}")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        if os.path.exists(bg_path):
            os.remove(bg_path)
    print()

def test_user_input_validation():
    """Test 4: Test input validation functions."""
    print("Test 4: Testing input validation...")
    
    # Test threshold validation
    try:
        overlay_scratches.validate_threshold("50")
        print("✓ Valid threshold accepted")
    except:
        print("✗ Valid threshold rejected")
    
    try:
        overlay_scratches.validate_threshold("300")
        print("✗ Invalid threshold accepted")
    except ValueError:
        print("✓ Invalid threshold rejected")
    
    # Test opacity validation
    try:
        overlay_scratches.validate_opacity("0.5")
        print("✓ Valid opacity accepted")
    except:
        print("✗ Valid opacity rejected")
    
    try:
        overlay_scratches.validate_opacity("1.5")
        print("✗ Invalid opacity accepted")
    except ValueError:
        print("✓ Invalid opacity rejected")
    
    print()

def test_process_batch():
    """Test 5: Test batch processing."""
    print("Test 5: Testing batch processing...")
    
    # Create temporary directory
    test_dir = tempfile.mkdtemp()
    output_dir = os.path.join(test_dir, "output")
    
    try:
        # Create test files
        for i in range(2):
            # BMP files
            scratch_img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.line(scratch_img, (10*i, 10*i), (40, 40), (50, 50, 50), 1)
            cv2.imwrite(os.path.join(test_dir, f"scratch_{i}.bmp"), scratch_img)
            
            # Clean files
            clean_img = np.ones((50, 50, 3), dtype=np.uint8) * 180
            cv2.circle(clean_img, (25, 25), 15, (100, 100, 100), -1)
            cv2.imwrite(os.path.join(test_dir, f"clean_{i}.png"), clean_img)
        
        # Run batch processing
        overlay_scratches.process_batch(test_dir, output_dir, threshold=30, opacity=0.7)
        
        # Check results
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            expected_files = 4  # 2 BMPs x 2 cleans
            
            if len(output_files) == expected_files:
                print("✓ Batch processing completed successfully")
                print(f"  - Created {len(output_files)} output files")
            else:
                print(f"✗ Expected {expected_files} files, got {len(output_files)}")
        else:
            print("✗ Output directory not created")
    
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        # Cleanup
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
    print()

def test_get_user_input():
    """Test 6: Test user input function."""
    print("Test 6: Testing user input function...")
    
    from unittest.mock import patch
    
    # Test default value
    with patch('builtins.input', return_value=''):
        result = overlay_scratches.get_user_input("Test prompt", "default")
        if result == "default":
            print("✓ Default value works")
        else:
            print(f"✗ Expected 'default', got '{result}'")
    
    # Test custom value
    with patch('builtins.input', return_value='custom'):
        result = overlay_scratches.get_user_input("Test prompt", "default")
        if result == "custom":
            print("✓ Custom value works")
        else:
            print(f"✗ Expected 'custom', got '{result}'")
    
    print()

def test_edge_cases():
    """Test 7: Test edge cases."""
    print("Test 7: Testing edge cases...")
    
    # Test with very small image
    tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
    tiny_path = "tiny.bmp"
    cv2.imwrite(tiny_path, tiny_img)
    
    try:
        mask, image = overlay_scratches.extract_scratches(tiny_path, threshold=30)
        print("✓ Handles small images")
    except Exception as e:
        print(f"✗ Failed with small image: {e}")
    finally:
        if os.path.exists(tiny_path):
            os.remove(tiny_path)
    
    # Test with nonexistent file
    try:
        overlay_scratches.extract_scratches("nonexistent.bmp")
        print("✗ Should have raised error for nonexistent file")
    except ValueError:
        print("✓ Correctly handles nonexistent files")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print()

def run_all_tests():
    """Run all function tests."""
    print("="*60)
    print("FUNCTION-BY-FUNCTION TESTS FOR overlay_scratches.py")
    print("="*60)
    print()
    
    test_dependency_check()
    test_extract_scratches()
    test_overlay_scratches_function()
    test_user_input_validation()
    test_process_batch()
    test_get_user_input()
    test_edge_cases()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()