#!/usr/bin/env python3
"""
Test script for frequency features emulator.
"""

import cv2
import numpy as np
import sys
import os

# Add the current directory to path
sys.path.append('.')

def test_frequency_filtering():
    """Test frequency filtering functionality."""
    print("Testing frequency filtering...")
    
    # Create a test image with known frequency components
    img = np.zeros((400, 400), dtype=np.uint8)
    
    # Add different frequency patterns
    x = np.arange(400)
    y = np.arange(400)
    X, Y = np.meshgrid(x, y)
    
    # Low frequency component
    low_freq = np.sin(2 * np.pi * X / 100) * 50
    
    # High frequency component
    high_freq = np.sin(2 * np.pi * Y / 10) * 30
    
    # Diagonal pattern
    diagonal = np.sin(2 * np.pi * (X + Y) / 50) * 40
    
    # Combine patterns
    img = low_freq + high_freq + diagonal + 128
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Test frequency filtering
    def apply_frequency_filter(gray, filter_type, cutoff_freq):
        """Apply frequency domain filter to grayscale image."""
        h, w = gray.shape
        
        # Compute FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)

        # Create frequency grid
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Normalize distance to [0, 1]
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_normalized = dist_from_center / max_dist

        # Create filter mask
        if filter_type == 'lowpass':
            mask = dist_normalized <= cutoff_freq
        elif filter_type == 'highpass':
            mask = dist_normalized >= cutoff_freq
        elif filter_type == 'bandpass':
            inner_cutoff = cutoff_freq / 2
            mask = (dist_normalized >= inner_cutoff) & (dist_normalized <= cutoff_freq)
        else:
            mask = np.ones((h, w), dtype=bool)

        # Apply filter
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.real(img_filtered)

        # Normalize and clip
        img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
        
        return img_filtered

    # Test different filters
    filters = [
        ('lowpass', 0.3),
        ('highpass', 0.1),
        ('bandpass', 0.3)
    ]
    
    for filter_type, cutoff in filters:
        print(f"Testing {filter_type} filter with cutoff {cutoff}...")
        filtered = apply_frequency_filter(img, filter_type, cutoff)
        
        # Save results
        cv2.imwrite(f"test_frequency_{filter_type}_{cutoff}.bmp", filtered)
        print(f"Saved: test_frequency_{filter_type}_{cutoff}.bmp")
    
    # Save original
    cv2.imwrite("test_frequency_original.bmp", img_bgr)
    print("Saved: test_frequency_original.bmp")
    
    print("Frequency filtering test completed!")

def test_image_loading():
    """Test image loading functionality."""
    print("Testing image loading...")
    
    # Check if test images exist
    test_images = [
        "pictures/good.bmp",
        "pictures/frequency_test.bmp",
        "pictures/morphological_test.bmp"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"Found test image: {img_path}")
            img = cv2.imread(img_path)
            if img is not None:
                print(f"  - Loaded successfully, size: {img.shape}")
            else:
                print(f"  - Failed to load")
        else:
            print(f"Test image not found: {img_path}")

if __name__ == "__main__":
    print("Testing Frequency Features Emulator...")
    print("=" * 50)
    
    test_image_loading()
    print()
    test_frequency_filtering()
    
    print("\nTest completed!") 