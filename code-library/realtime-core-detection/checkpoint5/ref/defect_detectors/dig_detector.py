#!/usr/bin/env python3

import cv2
import numpy as np
from typing import List, Dict


def detect_digs(gray: np.ndarray, min_size: int = 10, 
               max_size: int = 5000) -> List[Dict]:
    """Detect digs using morphological black-hat transform."""
    # Create circular kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Apply black-hat transform to extract dark spots
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold at 95th percentile to get significant dark spots
    threshold_value = np.percentile(blackhat, 95)
    _, dig_mask = cv2.threshold(blackhat, threshold_value, 255, 
                               cv2.THRESH_BINARY)
    
    # Find contours of dark spots
    contours, _ = cv2.findContours(dig_mask.astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digs = []
    
    # Process each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size
        if min_size < area < max_size:
            # Calculate moments for centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])  # X centroid
                cy = int(M["m01"] / M["m00"])  # Y centroid
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                
                # Calculate bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                dig = {
                    'center': (cx, cy),         # Center point
                    'area': float(area),        # Area in pixels
                    'contour': contour,         # Contour points
                    'circularity': float(circularity),  # Shape metric
                    'radius': float(radius),    # Bounding circle radius
                    'confidence': 0.8,          # Fixed confidence for digs
                }
                
                digs.append(dig)
    
    return digs


def detect_digs_advanced(gray: np.ndarray, min_size: int = 10,
                        max_size: int = 5000) -> List[Dict]:
    """Advanced dig detection with multiple methods."""
    digs = []
    
    # Method 1: Morphological black-hat
    morph_digs = detect_digs(gray, min_size, max_size)
    digs.extend(morph_digs)
    
    # Method 2: Adaptive thresholding
    adaptive_digs = detect_digs_adaptive(gray, min_size, max_size)
    digs.extend(adaptive_digs)
    
    # Method 3: Laplacian of Gaussian
    log_digs = detect_digs_log(gray, min_size, max_size)
    digs.extend(log_digs)
    
    return digs


def detect_digs_adaptive(gray: np.ndarray, min_size: int = 10,
                        max_size: int = 5000) -> List[Dict]:
    """Detect digs using adaptive thresholding."""
    digs = []
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 5)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_size < area < max_size:
            # Calculate moments for centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                
                # Only consider relatively circular shapes
                if circularity > 0.3:
                    dig = {
                        'center': (cx, cy),
                        'area': float(area),
                        'contour': contour,
                        'circularity': float(circularity),
                        'confidence': 0.6,
                        'method': 'adaptive'
                    }
                    digs.append(dig)
    
    return digs


def detect_digs_log(gray: np.ndarray, min_size: int = 10,
                   max_size: int = 5000) -> List[Dict]:
    """Detect digs using Laplacian of Gaussian."""
    digs = []
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Find negative peaks (dark spots)
    threshold = np.percentile(laplacian, 5)  # Bottom 5%
    dark_spots = laplacian < threshold
    
    # Find connected components
    num_features, labeled = cv2.connectedComponents(dark_spots.astype(np.uint8))
    
    for i in range(1, int(num_features)):  # Skip background (0)
        # Get component mask
        component = (labeled == i).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_size < area < max_size:
                # Calculate moments for centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                    
                    # Calculate bounding circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    dig = {
                        'center': (cx, cy),
                        'area': float(area),
                        'contour': contour,
                        'circularity': float(circularity),
                        'radius': float(radius),
                        'confidence': 0.7,
                        'method': 'log'
                    }
                    digs.append(dig)
    
    return digs


def test_dig_detection():
    """Test function for dig detection."""
    # Create test image with digs
    test_image = np.zeros((200, 200), dtype=np.uint8)
    
    # Add some digs (dark circles)
    cv2.circle(test_image, (50, 50), 8, 50, -1)   # Small dig
    cv2.circle(test_image, (150, 100), 12, 30, -1)  # Medium dig
    cv2.circle(test_image, (100, 150), 15, 20, -1)  # Large dig
    
    # Add noise
    noise = np.random.normal(0, 15, test_image.shape).astype(np.uint8)
    test_image = np.clip(test_image + noise, 0, 255)
    
    # Detect digs
    digs = detect_digs_advanced(test_image)
    
    # Print results
    print(f"Detected {len(digs)} digs:")
    for i, dig in enumerate(digs):
        print(f"  Dig {i+1}:")
        print(f"    Center: {dig['center']}")
        print(f"    Area: {dig['area']:.1f} pixels")
        print(f"    Circularity: {dig['circularity']:.3f}")
        print(f"    Confidence: {dig['confidence']:.2f}")
        if 'method' in dig:
            print(f"    Method: {dig['method']}")
        if 'radius' in dig:
            print(f"    Radius: {dig['radius']:.1f} pixels")
    
    return digs


if __name__ == "__main__":
    test_dig_detection() 