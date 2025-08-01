#!/usr/bin/env python3

import cv2
import numpy as np
from typing import List, Dict


def detect_blobs(gray: np.ndarray, min_size: int = 100,
                max_size: int = 10000) -> List[Dict]:
    """Detect blobs using adaptive thresholding."""
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 5)
    
    # Morphological operations to clean up blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove noise
    
    # Find blob contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    blobs = []
    
    # Process each blob
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size
        if min_size < area < max_size:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Compute circularity (perfect circle = 1.0)
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
            
            # Calculate aspect ratio
            aspect_ratio = w / (h + 1e-10)  # Width/height ratio
            
            # Calculate moments for centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])  # X centroid
                cy = int(M["m01"] / M["m00"])  # Y centroid
                
                blob = {
                    'contour': contour,                     # Contour points
                    'bbox': (x, y, w, h),                   # Bounding box
                    'area': float(area),                    # Area in pixels
                    'circularity': float(circularity),      # Shape metric
                    'aspect_ratio': float(aspect_ratio),    # Shape ratio
                    'center': (cx, cy),                     # Center point
                    'perimeter': float(perimeter),          # Perimeter
                    'confidence': 0.6,                      # Fixed confidence
                }
                
                blobs.append(blob)
    
    return blobs


def detect_blobs_advanced(gray: np.ndarray, min_size: int = 100,
                         max_size: int = 10000) -> List[Dict]:
    """Advanced blob detection with multiple methods."""
    blobs = []
    
    # Method 1: Adaptive thresholding
    adaptive_blobs = detect_blobs(gray, min_size, max_size)
    blobs.extend(adaptive_blobs)
    
    # Method 2: Watershed segmentation
    watershed_blobs = detect_blobs_watershed(gray, min_size, max_size)
    blobs.extend(watershed_blobs)
    
    # Method 3: MSER (Maximally Stable Extremal Regions)
    mser_blobs = detect_blobs_mser(gray, min_size, max_size)
    blobs.extend(mser_blobs)
    
    return blobs


def detect_blobs_watershed(gray: np.ndarray, min_size: int = 100,
                          max_size: int = 10000) -> List[Dict]:
    """Detect blobs using watershed segmentation."""
    blobs = []
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Sure foreground area
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(),
                              255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    
    # Process watershed regions
    for marker in range(2, markers.max() + 1):  # Skip background and border
        mask = (markers == marker).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_size < area < max_size:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate properties
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                aspect_ratio = w / (h + 1e-10)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    blob = {
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': float(area),
                        'circularity': float(circularity),
                        'aspect_ratio': float(aspect_ratio),
                        'center': (cx, cy),
                        'perimeter': float(perimeter),
                        'confidence': 0.7,
                        'method': 'watershed'
                    }
                    blobs.append(blob)
    
    return blobs


def detect_blobs_mser(gray: np.ndarray, min_size: int = 100,
                     max_size: int = 10000) -> List[Dict]:
    """Detect blobs using MSER (Maximally Stable Extremal Regions)."""
    blobs = []
    
    # Create MSER detector
    mser = cv2.MSER_create(
        _min_area=min_size,
        _max_area=max_size,
        _delta=5
    )
    
    # Detect regions
    regions, _ = mser.detectRegions(gray)
    
    for region in regions:
        # Convert region to contour
        contour = region.reshape(-1, 1, 2).astype(np.int32)
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        if min_size < area < max_size:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
            aspect_ratio = w / (h + 1e-10)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                blob = {
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': float(area),
                    'circularity': float(circularity),
                    'aspect_ratio': float(aspect_ratio),
                    'center': (cx, cy),
                    'perimeter': float(perimeter),
                    'confidence': 0.8,
                    'method': 'mser'
                }
                blobs.append(blob)
    
    return blobs


def test_blob_detection():
    """Test function for blob detection."""
    # Create test image with blobs
    test_image = np.zeros((200, 200), dtype=np.uint8)
    
    # Add some blobs
    cv2.circle(test_image, (50, 50), 15, 200, -1)    # Large blob
    cv2.circle(test_image, (150, 100), 10, 180, -1)   # Medium blob
    cv2.ellipse(test_image, (100, 150), (20, 10), 45, 0, 360, 160, -1)  # Elliptical blob
    
    # Add noise
    noise = np.random.normal(0, 20, test_image.shape).astype(np.uint8)
    test_image = np.clip(test_image + noise, 0, 255)
    
    # Detect blobs
    blobs = detect_blobs_advanced(test_image)
    
    # Print results
    print(f"Detected {len(blobs)} blobs:")
    for i, blob in enumerate(blobs):
        print(f"  Blob {i+1}:")
        print(f"    Center: {blob['center']}")
        print(f"    Area: {blob['area']:.1f} pixels")
        print(f"    Circularity: {blob['circularity']:.3f}")
        print(f"    Aspect Ratio: {blob['aspect_ratio']:.2f}")
        print(f"    Confidence: {blob['confidence']:.2f}")
        if 'method' in blob:
            print(f"    Method: {blob['method']}")
    
    return blobs


if __name__ == "__main__":
    test_blob_detection() 