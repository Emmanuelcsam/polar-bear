#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import sys
import os


def extract_morphological_features(gray):
    """Extract morphological features."""
    # Initialize feature dictionary
    features = {}
    
    # Multi-scale morphological operations
    for size in [3, 5, 7, 11]:
        # Create circular structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        
        # White tophat: bright features smaller than kernel
        wth = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        # Black tophat: dark features smaller than kernel
        bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Statistics of tophat transforms
        features[f'morph_wth_{size}_mean'] = float(np.mean(wth))
        features[f'morph_wth_{size}_max'] = float(np.max(wth))
        features[f'morph_wth_{size}_sum'] = float(np.sum(wth))
        features[f'morph_bth_{size}_mean'] = float(np.mean(bth))
        features[f'morph_bth_{size}_max'] = float(np.max(bth))
        features[f'morph_bth_{size}_sum'] = float(np.sum(bth))
    
    # Binary morphology analysis
    # Otsu's threshold for automatic binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Define 5x5 square kernel
    kernel = np.ones((5, 5), np.uint8)
    # Erosion: shrinks white regions
    erosion = cv2.erode(binary, kernel, iterations=1)
    # Dilation: expands white regions
    dilation = cv2.dilate(binary, kernel, iterations=1)
    # Morphological gradient: difference between dilation and erosion
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    
    # Compute morphological statistics
    features['morph_binary_area_ratio'] = float(np.sum(binary) / binary.size)
    features['morph_gradient_sum'] = float(np.sum(gradient))
    features['morph_erosion_ratio'] = float(np.sum(erosion) / (np.sum(binary) + 1e-10))
    features['morph_dilation_ratio'] = float(np.sum(dilation) / (np.sum(binary) + 1e-10))
    
    return features


def main():
    """Standalone script to test morphological feature extraction."""
    parser = argparse.ArgumentParser(description='Extract morphological features from image')
    parser.add_argument('--test', action='store_true', help='Run test with sample image')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Output file for features (JSON)')
    
    args = parser.parse_args()
    
    if args.test:
        # Generate test image
        print("Testing morphological feature extraction...")
        
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
        features = extract_morphological_features(test_img)
        
        print("Extracted morphological features:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
        
        print("✓ Morphological feature extraction test completed!")
        
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
        features = extract_morphological_features(gray)
        
        print(f"Morphological features for {args.image}:")
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
        print("  python morphological_features.py --test")
        print("  python morphological_features.py --image input.png")
        print("  python morphological_features.py --image input.png --output features.json")
        sys.exit(1)


if __name__ == "__main__":
    main() 