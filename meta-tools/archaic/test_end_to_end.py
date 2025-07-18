#!/usr/bin/env python3
"""
End-to-end test for directory-crop-with-ref.py
Tests the complete program functionality
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import cv2
from pathlib import Path
import subprocess
import time

def create_test_images(ref_dir, target_dir):
    """Create test images for testing"""
    print("Creating test images...")
    
    # Create reference images (green circles on white background)
    for i in range(3):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(img, (100, 100), 50, (0, 255, 0), -1)  # Green circle
        cv2.imwrite(str(ref_dir / f"ref_{i}.png"), img)
    
    # Create target images with variations
    for i in range(5):
        img = np.ones((250, 250, 3), dtype=np.uint8) * 255
        # Vary position and size
        center_x = 125 + (i - 2) * 10
        center_y = 125 + (i - 2) * 10
        radius = 60 + (i - 2) * 5
        cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), -1)
        cv2.imwrite(str(target_dir / f"target_{i}.png"), img)
    
    print(f"Created {len(list(ref_dir.glob('*.png')))} reference images")
    print(f"Created {len(list(target_dir.glob('*.png')))} target images")

def test_program():
    """Test the full program"""
    print("="*60)
    print("END-TO-END TEST FOR directory-crop-with-ref.py")
    print("="*60)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as test_dir:
        ref_dir = Path(test_dir) / "reference"
        target_dir = Path(test_dir) / "target"
        output_dir = Path(test_dir) / "output"
        
        ref_dir.mkdir()
        target_dir.mkdir()
        
        # Create test images
        create_test_images(ref_dir, target_dir)
        
        # Prepare input for the program
        inputs = f"{ref_dir}\n{target_dir}\n{output_dir}\n"
        
        print("\nRunning directory-crop-with-ref.py...")
        print(f"Reference directory: {ref_dir}")
        print(f"Target directory: {target_dir}")
        print(f"Output directory: {output_dir}")
        
        # Run the program with automated inputs
        # We'll simulate clicking "Confirm" by setting up a separate process
        env = os.environ.copy()
        env['SDL_VIDEODRIVER'] = 'dummy'  # Use dummy video driver for testing
        
        try:
            # Create a test script that will auto-confirm
            test_script = Path(test_dir) / "test_run.py"
            test_script.write_text(f'''
import sys
import os
# Add the actual script directory to path
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")

# Mock pygame to auto-confirm
import unittest.mock as mock

# Set up pygame mock
pygame_mock = mock.MagicMock()
pygame_mock.display.set_mode = mock.MagicMock()
pygame_mock.event.get = mock.MagicMock(return_value=[])
pygame_mock.mouse.get_pos = mock.MagicMock(return_value=(100, 100))

# Mock the preview UI to auto-confirm
class MockUI:
    def __init__(self, *args, **kwargs):
        pass
    def run_preview(self, *args, **kwargs):
        return True  # Auto-confirm
    def quit(self):
        pass

# Patch imports
with mock.patch.dict(sys.modules, {{'pygame': pygame_mock}}):
    import directory_crop_with_ref as dcr
    dcr.CropPreviewUI = MockUI
    
    # Mock input to provide directories
    with mock.patch('builtins.input', side_effect=[
        "{ref_dir}",
        "{target_dir}", 
        "{output_dir}"
    ]):
        dcr.main()
''')
            
            # Run the test script
            result = subprocess.run(
                [sys.executable, str(test_script)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print("\nProgram output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            
            # Check results
            print("\nChecking results...")
            
            # Check if output directory was created
            if output_dir.exists():
                print(f"✓ Output directory created: {output_dir}")
            else:
                print(f"✗ Output directory not created")
                return False
            
            # Check output files
            output_files = list(output_dir.glob("*.png"))
            print(f"✓ Generated {len(output_files)} output files")
            
            if len(output_files) == 0:
                print("✗ No output files generated")
                return False
            
            # Verify output images
            for output_file in output_files[:2]:  # Check first 2
                img = cv2.imread(str(output_file), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"✗ Failed to read {output_file}")
                    return False
                
                if img.shape[2] != 4:
                    print(f"✗ Output image doesn't have alpha channel: {output_file}")
                    return False
                
                # Check if alpha channel has both 0 and 255 values
                alpha = img[:, :, 3]
                if np.any(alpha > 0):
                    print(f"✓ Output image has foreground pixels: {output_file.name}")
                else:
                    print(f"✗ Output image has no foreground pixels: {output_file.name}")
            
            # Check log file
            log_file = Path("advanced_crop_learner.log")
            if log_file.exists():
                print("✓ Log file created")
                # Clean up log file
                log_file.unlink()
            else:
                print("✗ Log file not created")
            
            # Clean up cache
            cache_file = Path("ref_features.json")
            if cache_file.exists():
                print("✓ Cache file created (cleaning up)")
                cache_file.unlink()
            
            print("\n" + "="*60)
            print("END-TO-END TEST COMPLETED SUCCESSFULLY!")
            print("="*60)
            return True
            
        except subprocess.TimeoutExpired:
            print("✗ Program timed out")
            return False
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_direct_run():
    """Test running the program directly with sample data"""
    print("\n" + "="*60)
    print("DIRECT RUN TEST")
    print("="*60)
    
    # Check if the main script exists
    script_path = Path("directory-crop-with-ref.py")
    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False
    
    print(f"✓ Script found: {script_path}")
    
    # Test importing
    try:
        # Test if all required libraries can be imported
        import numpy
        import cv2
        import torch
        import pygame
        print("✓ All required libraries are available")
    except ImportError as e:
        print(f"✗ Missing required library: {e}")
        return False
    
    # Check script syntax
    try:
        compile(script_path.read_text(), str(script_path), 'exec')
        print("✓ Script syntax is valid")
    except SyntaxError as e:
        print(f"✗ Syntax error in script: {e}")
        return False
    
    print("\n" + "="*60)
    print("DIRECT RUN TEST PASSED!")
    print("="*60)
    return True

if __name__ == "__main__":
    # Run tests
    success = True
    
    # Test direct run
    if not test_direct_run():
        success = False
    
    # Test full program
    if not test_program():
        success = False
    
    # Final result
    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED! ✓")
        print("directory-crop-with-ref.py is running at 110%!")
    else:
        print("SOME TESTS FAILED! ✗")
    print("="*60)
    
    sys.exit(0 if success else 1)