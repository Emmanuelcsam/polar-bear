#!/usr/bin/env python3
"""
Manual test to verify directory-crop-with-ref.py functionality
Creates test data and provides instructions for testing
"""

import os
import numpy as np
import cv2
from pathlib import Path
import shutil

def create_test_data():
    """Create test data for manual testing"""
    print("="*60)
    print("CREATING TEST DATA FOR directory-crop-with-ref.py")
    print("="*60)
    
    # Create test directories
    test_root = Path("test_crop_data")
    if test_root.exists():
        shutil.rmtree(test_root)
    
    ref_dir = test_root / "reference_images"
    target_dir = test_root / "target_images"
    output_dir = test_root / "output"
    
    ref_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    
    print(f"\nCreating test directories:")
    print(f"Reference: {ref_dir.absolute()}")
    print(f"Target: {target_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    
    # Create reference images (objects to be cropped)
    print("\nCreating reference images (pre-cropped examples)...")
    
    # Type 1: Red squares
    for i in range(3):
        # Create transparent background
        img = np.zeros((150, 150, 4), dtype=np.uint8)
        # Draw red square
        cv2.rectangle(img, (25, 25), (125, 125), (0, 0, 255, 255), -1)
        cv2.imwrite(str(ref_dir / f"ref_square_{i}.png"), img)
    
    # Type 2: Green circles
    for i in range(3):
        img = np.zeros((150, 150, 4), dtype=np.uint8)
        cv2.circle(img, (75, 75), 50, (0, 255, 0, 255), -1)
        cv2.imwrite(str(ref_dir / f"ref_circle_{i}.png"), img)
    
    print(f"Created {len(list(ref_dir.glob('*.png')))} reference images")
    
    # Create target images (images to be cropped)
    print("\nCreating target images (images to be cropped)...")
    
    # Target images with red squares on various backgrounds
    backgrounds = [
        (255, 255, 255),  # White
        (200, 200, 200),  # Light gray
        (100, 100, 100),  # Dark gray
        (255, 200, 200),  # Light red
        (200, 255, 200),  # Light green
    ]
    
    for i, bg_color in enumerate(backgrounds):
        # Create background
        img = np.ones((300, 300, 3), dtype=np.uint8)
        img[:, :] = bg_color
        # Add red square at varying positions
        x = 50 + i * 20
        y = 50 + i * 20
        size = 100 + i * 10
        cv2.rectangle(img, (x, y), (x + size, y + size), (0, 0, 255), -1)
        # Add some noise
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite(str(target_dir / f"target_square_{i}.png"), img)
    
    # Target images with green circles
    for i, bg_color in enumerate(backgrounds):
        img = np.ones((300, 300, 3), dtype=np.uint8)
        img[:, :] = bg_color
        # Add green circle
        center_x = 150 + (i - 2) * 20
        center_y = 150 + (i - 2) * 20
        radius = 60 + i * 5
        cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), -1)
        # Add some texture
        for _ in range(50):
            pt1 = (np.random.randint(0, 300), np.random.randint(0, 300))
            pt2 = (np.random.randint(0, 300), np.random.randint(0, 300))
            cv2.line(img, pt1, pt2, bg_color, 1)
        cv2.imwrite(str(target_dir / f"target_circle_{i}.png"), img)
    
    print(f"Created {len(list(target_dir.glob('*.png')))} target images")
    
    # Create a mixed difficult case
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    # Multiple shapes
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
    cv2.circle(img, (300, 100), 40, (0, 255, 0), -1)
    cv2.rectangle(img, (250, 250), (350, 350), (255, 0, 0), -1)
    cv2.imwrite(str(target_dir / "target_mixed.png"), img)
    
    print("\n" + "="*60)
    print("TEST DATA CREATED SUCCESSFULLY!")
    print("="*60)
    
    return ref_dir, target_dir, output_dir

def test_individual_functions():
    """Test individual functions of the module"""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL FUNCTIONS")
    print("="*60)
    
    try:
        import directory_crop_with_ref as dcr
        
        # Test 1: Directory validation
        print("\n1. Testing directory validation...")
        test_dir = Path("test_crop_data/reference_images")
        if test_dir.exists():
            validated = dcr.validate_directory(str(test_dir), "Test")
            print(f"✓ Directory validation: {validated}")
        
        # Test 2: Feature extraction
        print("\n2. Testing feature extraction...")
        test_images = list(test_dir.glob("*.png"))
        if test_images:
            features = dcr.extract_comprehensive_features(str(test_images[0]))
            print(f"✓ Extracted {len(features)} features from {test_images[0].name}")
            print(f"  Sample features: {list(features.keys())[:5]}...")
        
        # Test 3: Safe division
        print("\n3. Testing safe division...")
        result1 = dcr.safe_divide(10, 2)
        result2 = dcr.safe_divide(10, 0)
        print(f"✓ Safe divide 10/2 = {result1}")
        print(f"✓ Safe divide 10/0 = {result2}")
        
        # Test 4: Parameter adjustment
        print("\n4. Testing parameter adjustment...")
        params = {'color_multiplier': 1.0, 'morph_kernel': 5, 
                 'area_tolerance': 0.5, 'aspect_tolerance': 0.5}
        adjusted = dcr.adjust_parameters(params.copy(), 1)
        print(f"✓ Original params: {params}")
        print(f"✓ Adjusted params: {adjusted}")
        
        print("\n✓ All function tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_instructions():
    """Print instructions for manual testing"""
    print("\n" + "="*60)
    print("MANUAL TESTING INSTRUCTIONS")
    print("="*60)
    
    print("\nTo test the program manually:")
    print("\n1. Run the main program:")
    print("   python directory-crop-with-ref.py")
    
    print("\n2. When prompted, enter these directories:")
    ref_dir = Path("test_crop_data/reference_images").absolute()
    target_dir = Path("test_crop_data/target_images").absolute()
    output_dir = Path("test_crop_data/output").absolute()
    
    print(f"   Reference: {ref_dir}")
    print(f"   Target: {target_dir}")
    print(f"   Output: {output_dir}")
    
    print("\n3. In the preview window:")
    print("   - Review the cropping results")
    print("   - Click 'Confirm' if satisfied")
    print("   - Click 'Adjust' to try different parameters")
    print("   - Press ESC to cancel")
    
    print("\n4. Check the output directory for results")
    
    print("\n" + "="*60)
    print("PROGRAM FEATURES TO TEST:")
    print("="*60)
    print("✓ Automatic library installation")
    print("✓ Reference image analysis with caching")
    print("✓ Multiple image format support")
    print("✓ Interactive preview with parameter adjustment")
    print("✓ Batch processing with progress bars")
    print("✓ Comprehensive logging")
    print("✓ Error handling and recovery")
    print("✓ Memory optimization for large images")
    print("✓ Multi-threaded processing")
    print("✓ Keyboard shortcuts in UI")

if __name__ == "__main__":
    # Create test data
    ref_dir, target_dir, output_dir = create_test_data()
    
    # Test individual functions
    test_individual_functions()
    
    # Print instructions
    print_instructions()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE - Ready for manual testing!")
    print("="*60)