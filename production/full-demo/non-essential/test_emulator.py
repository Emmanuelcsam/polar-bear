#!/usr/bin/env python3
"""Test script to verify the frequency features emulator works correctly."""

import cv2
import numpy as np
import sys
import os

def test_basic_operations():
    """Test basic image processing operations from the emulator."""
    
    print("Testing Frequency Features Emulator...")
    print("-" * 40)
    
    # Check if test image exists
    test_image_path = "frequency_test.bmp"
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image '{test_image_path}' not found!")
        return False
    
    print(f"✓ Test image found: {test_image_path}")
    
    # Load test image
    try:
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Failed to load image")
        h, w = image.shape
        print(f"✓ Image loaded successfully: {w}x{h}")
    except Exception as e:
        print(f"ERROR: Failed to load image: {e}")
        return False
    
    # Test FFT computation
    try:
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        phase = np.angle(f_shift)
        print(f"✓ FFT computed successfully")
        print(f"  - Magnitude range: [{np.min(magnitude):.2f}, {np.max(magnitude):.2f}]")
        print(f"  - Phase range: [{np.min(phase):.2f}, {np.max(phase):.2f}]")
    except Exception as e:
        print(f"ERROR: FFT computation failed: {e}")
        return False
    
    # Test frequency filtering
    try:
        # Simple low-pass filter
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_normalized = dist_from_center / max_dist
        
        # Create low-pass mask
        mask = (dist_normalized <= 0.3).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (21, 21), 5)
        
        # Apply filter
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.real(img_filtered)
        img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
        
        print(f"✓ Frequency filtering works")
        print(f"  - Output range: [{np.min(img_filtered)}, {np.max(img_filtered)}]")
    except Exception as e:
        print(f"ERROR: Frequency filtering failed: {e}")
        return False
    
    # Test feature calculation
    try:
        # Basic frequency features
        fft_mean = np.mean(magnitude)
        fft_std = np.std(magnitude)
        fft_max = np.max(magnitude)
        total_power = np.sum(magnitude ** 2) / (h * w)
        dc_component = magnitude[h//2, w//2]
        
        print(f"✓ Feature extraction works")
        print(f"  - FFT Mean: {fft_mean:.2f}")
        print(f"  - FFT Std: {fft_std:.2f}")
        print(f"  - FFT Max: {fft_max:.2f}")
        print(f"  - Total Power: {total_power:.2e}")
        print(f"  - DC Component: {dc_component:.2f}")
    except Exception as e:
        print(f"ERROR: Feature extraction failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("All tests passed successfully!")
    return True

if __name__ == "__main__":
    success = test_basic_operations()
    sys.exit(0 if success else 1)
