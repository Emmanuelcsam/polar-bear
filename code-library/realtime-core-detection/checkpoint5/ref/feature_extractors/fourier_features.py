#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import sys
import os


def extract_fourier_features(gray):
    """Extract 2D Fourier Transform features."""
    # Compute 2D FFT
    f = np.fft.fft2(gray)
    # Shift zero frequency to center
    fshift = np.fft.fftshift(f)
    # Compute magnitude spectrum
    magnitude = np.abs(fshift)
    # Compute power spectrum
    power = magnitude**2
    # Compute phase spectrum
    phase = np.angle(fshift)
    
    # Calculate center coordinates
    center = np.array(power.shape) // 2
    # Create coordinate grids
    y, x = np.ogrid[:power.shape[0], :power.shape[1]]
    # Compute distance from center for each pixel
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    # Compute radial profile (average power at each radius)
    radial_prof = []
    for radius in range(1, min(center)):
        # Create ring mask
        mask = (r >= radius - 1) & (r < radius)
        # Average power in ring
        if mask.any():
            radial_prof.append(np.mean(power[mask]))
    
    # Convert to array
    radial_prof = np.array(radial_prof)
    
    # Compute spectral centroid and spread if profile exists
    if len(radial_prof) > 0:
        # Weighted average of frequencies
        spectral_centroid = float(np.sum(np.arange(len(radial_prof)) * radial_prof) / 
                                 (np.sum(radial_prof) + 1e-10))
        # Weighted standard deviation of frequencies
        spectral_spread = float(np.sqrt(np.sum((np.arange(len(radial_prof)) - 
                                               spectral_centroid)**2 * radial_prof) / 
                                       (np.sum(radial_prof) + 1e-10)))
    else:
        spectral_centroid = 0.0
        spectral_spread = 0.0
    
    return {
        'fft_mean_magnitude': float(np.mean(magnitude)),
        'fft_std_magnitude': float(np.std(magnitude)),
        'fft_max_magnitude': float(np.max(magnitude)),
        'fft_total_power': float(np.sum(power)),
        'fft_dc_component': float(magnitude[center[0], center[1]]),
        'fft_mean_phase': float(np.mean(phase)),
        'fft_std_phase': float(np.std(phase)),
        'fft_spectral_centroid': spectral_centroid,
        'fft_spectral_spread': spectral_spread,
    }


def main():
    """Standalone script to test Fourier feature extraction."""
    parser = argparse.ArgumentParser(description='Extract Fourier features from image')
    parser.add_argument('--test', action='store_true', help='Run test with sample image')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Output file for features (JSON)')
    
    args = parser.parse_args()
    
    if args.test:
        # Generate test image
        print("Testing Fourier feature extraction...")
        
        # Create test image with different patterns
        test_img = np.zeros((100, 100), dtype=np.uint8)
        
        # Add some patterns
        test_img[20:80, 20:80] = 128  # Gray square
        test_img[30:70, 30:70] = 255  # White square inside
        test_img[40:60, 40:60] = 0    # Black square inside
        
        # Add some noise
        noise = np.random.normal(0, 20, test_img.shape).astype(np.uint8)
        test_img = np.clip(test_img + noise, 0, 255)
        
        # Extract features
        features = extract_fourier_features(test_img)
        
        print("Extracted Fourier features:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
        
        print("✓ Fourier feature extraction test completed!")
        
    elif args.image:
        # Load and process image
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        
        # Load image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image: {args.image}")
            sys.exit(1)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Extract features
        features = extract_fourier_features(gray)
        
        print(f"Fourier features for {args.image}:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
        
        # Save to file if output specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(features, f, indent=2)
            print(f"Features saved to: {args.output}")
        
    else:
        print("Usage:")
        print("  python fourier_features.py --test")
        print("  python fourier_features.py --image input.png")
        print("  python fourier_features.py --image input.png --output features.json")
        sys.exit(1)


if __name__ == "__main__":
    main() 