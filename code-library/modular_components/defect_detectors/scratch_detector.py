#!/usr/bin/env python3

import cv2
import numpy as np
from typing import List, Dict, Tuple


def detect_scratches(gray: np.ndarray, min_length: int = 20, 
                    max_gap: int = 5, threshold: int = 40) -> List[Dict]:
    """Detect scratches using Hough line transform."""
    # Edge detection using Canny
    edges = cv2.Canny(gray, 30, 100)
    
    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold,
                           minLineLength=min_length, maxLineGap=max_gap)
    
    scratches = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Extract endpoints
            
            # Calculate line length
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Calculate angle in degrees
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            # Create scratch object
            scratch = {
                'line': (x1, y1, x2, y2),  # Line endpoints
                'length': float(length),     # Line length in pixels
                'angle': float(angle),       # Angle in degrees
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),  # Center point
                'confidence': 0.7,           # Fixed confidence for scratches
            }
            
            scratches.append(scratch)
    
    return scratches


def detect_scratches_advanced(gray: np.ndarray, min_length: int = 20,
                            max_gap: int = 5, threshold: int = 40) -> List[Dict]:
    """Advanced scratch detection with multiple methods."""
    scratches = []
    
    # Method 1: Hough line transform
    hough_scratches = detect_scratches(gray, min_length, max_gap, threshold)
    scratches.extend(hough_scratches)
    
    # Method 2: Morphological operations
    morph_scratches = detect_scratches_morphological(gray)
    scratches.extend(morph_scratches)
    
    # Method 3: Gradient-based detection
    gradient_scratches = detect_scratches_gradient(gray)
    scratches.extend(gradient_scratches)
    
    return scratches


def detect_scratches_morphological(gray: np.ndarray) -> List[Dict]:
    """Detect scratches using morphological operations."""
    scratches = []
    
    # Create structuring elements for different orientations
    kernels = []
    for angle in [0, 45, 90, 135]:
        # Create line kernel
        kernel = np.zeros((15, 15), dtype=np.uint8)
        center = 7
        
        if angle == 0:  # Horizontal
            kernel[center, :] = 1
        elif angle == 45:  # Diagonal
            for i in range(15):
                kernel[i, i] = 1
        elif angle == 90:  # Vertical
            kernel[:, center] = 1
        else:  # 135 degrees
            for i in range(15):
                kernel[i, 14-i] = 1
        
        kernels.append((kernel, angle))
    
    # Apply morphological operations
    for kernel, angle in kernels:
        # Erode and dilate to find line-like structures
        eroded = cv2.erode(gray, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area and aspect ratio
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-10)
                
                # Check if it's line-like (high aspect ratio)
                if aspect_ratio > 3:
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        scratch = {
                            'line': (x, y, x + w, y + h),  # Bounding box as line
                            'length': float(max(w, h)),     # Length
                            'angle': float(angle),          # Angle
                            'center': (cx, cy),             # Center
                            'confidence': 0.6,              # Lower confidence
                            'method': 'morphological'
                        }
                        scratches.append(scratch)
    
    return scratches


def detect_scratches_gradient(gray: np.ndarray) -> List[Dict]:
    """Detect scratches using gradient analysis."""
    scratches = []
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    # Threshold magnitude to find strong edges
    threshold = np.percentile(magnitude, 90)
    strong_edges = magnitude > threshold
    
    # Find connected components
    labeled, num_features = cv2.connectedComponents(strong_edges.astype(np.uint8))
    
    for i in range(1, num_features):  # Skip background (0)
        # Get component mask
        component = (labeled == i).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-10)
                
                # Check if it's line-like
                if aspect_ratio > 2.5:
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calculate average direction
                        mask = component[y:y+h, x:x+w]
                        dir_mask = direction[y:y+h, x:x+w]
                        avg_direction = np.mean(dir_mask[mask > 0])
                        
                        scratch = {
                            'line': (x, y, x + w, y + h),
                            'length': float(max(w, h)),
                            'angle': float(avg_direction * 180 / np.pi),
                            'center': (cx, cy),
                            'confidence': 0.5,
                            'method': 'gradient'
                        }
                        scratches.append(scratch)
    
    return scratches


def test_scratch_detection():
    """Test function for scratch detection."""
    # Create test image with scratches
    test_image = np.zeros((200, 200), dtype=np.uint8)
    
    # Add some scratches
    # Horizontal scratch
    cv2.line(test_image, (20, 50), (180, 50), 255, 2)
    
    # Diagonal scratch
    cv2.line(test_image, (30, 30), (170, 170), 255, 2)
    
    # Vertical scratch
    cv2.line(test_image, (100, 20), (100, 180), 255, 2)
    
    # Add noise
    noise = np.random.normal(0, 20, test_image.shape).astype(np.uint8)
    test_image = np.clip(test_image + noise, 0, 255)
    
    # Detect scratches
    scratches = detect_scratches_advanced(test_image)
    
    # Print results
    print(f"Detected {len(scratches)} scratches:")
    for i, scratch in enumerate(scratches):
        print(f"  Scratch {i+1}:")
        print(f"    Length: {scratch['length']:.1f} pixels")
        print(f"    Angle: {scratch['angle']:.1f} degrees")
        print(f"    Center: {scratch['center']}")
        print(f"    Confidence: {scratch['confidence']:.2f}")
        if 'method' in scratch:
            print(f"    Method: {scratch['method']}")
    
    return scratches


if __name__ == "__main__":
    test_scratch_detection() 