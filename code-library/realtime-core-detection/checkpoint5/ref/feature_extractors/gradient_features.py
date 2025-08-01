#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import sys
import os


def extract_gradient_features(gray):
    """Extract gradient-based features."""
    # Compute Sobel gradients (first derivatives)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude (edge strength)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # Compute gradient orientation (edge direction)
    grad_orient = np.arctan2(grad_y, grad_x)
    
    # Compute Laplacian (second derivative)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Detect edges using Canny algorithm
    edges = cv2.Canny(gray, 50, 150)
    # Calculate edge density
    edge_density = np.sum(edges) / edges.size
    
    return {
        'gradient_magnitude_mean': float(np.mean(grad_mag)),
        'gradient_magnitude_std': float(np.std(grad_mag)),
        'gradient_magnitude_max': float(np.max(grad_mag)),
        'gradient_magnitude_sum': float(np.sum(grad_mag)),
        'gradient_orientation_mean': float(np.mean(grad_orient)),
        'gradient_orientation_std': float(np.std(grad_orient)),
        'laplacian_mean': float(np.mean(np.abs(laplacian))),
        'laplacian_std': float(np.std(laplacian)),
        'laplacian_sum': float(np.sum(np.abs(laplacian))),
        'edge_density': float(edge_density),
        'edge_count': float(np.sum(edges > 0)),
    }


def main():
    """Standalone script to test gradient feature extraction."""
    parser = argparse.ArgumentParser(description='Extract gradient features from image')
    parser.add_argument('--test', action='store_true', help='Run test with sample image')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Output file for features (JSON)')
    
    args = parser.parse_args()
    
    if args.test:
        # Generate test image
        print("Testing gradient feature extraction...")
        
        # Create test image with edges
        test_img = np.zeros((100, 100), dtype=np.uint8)
        
        # Add some edges
        cv2.line(test_img, (20, 20), (80, 20), 255, 2)  # Horizontal line
        cv2.line(test_img, (20, 40), (20, 80), 255, 2)  # Vertical line
        cv2.circle(test_img, (70, 70), 15, 255, 2)      # Circle
        
        # Add some noise
        noise = np.random.normal(0, 20, test_img.shape).astype(np.uint8)
        test_img = np.clip(test_img + noise, 0, 255)
        
        # Extract features
        features = extract_gradient_features(test_img)
        
        print("Extracted gradient features:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
        
        print("✓ Gradient feature extraction test completed!")
        
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
        features = extract_gradient_features(gray)
        
        print(f"Gradient features for {args.image}:")
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
        print("  python gradient_features.py --test")
        print("  python gradient_features.py --image input.png")
        print("  python gradient_features.py --image input.png --output features.json")
        sys.exit(1)


if __name__ == "__main__":
    main() 