#!/usr/bin/env python3
"""
Center Detection Functions
Extracted from multiple fiber optic analysis scripts

This module contains various methods for detecting the center of fiber optic images:
- Hough Circle Transform center detection
- Brightness-weighted centroid
- Morphological center detection
- Gradient-based center finding
- Multi-method center fusion
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.optimize import minimize
from scipy.ndimage import center_of_mass


def detect_center_hough_circles(image: np.ndarray, blur_kernel: int = 9, 
                                dp: float = 1.2, min_dist: Optional[int] = None,
                                param1: int = 50, param2: int = 30) -> Optional[Tuple[int, int]]:
    """
    Detect fiber center using Hough Circle Transform.
    
    From: sergio.py, separation_old2.py, fiber_optic_segmentation.py
    
    Args:
        image: Input grayscale image
        blur_kernel: Gaussian blur kernel size
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Higher threshold for edge detection
        param2: Accumulator threshold for center detection
        
    Returns:
        (x, y) center coordinates or None if not found
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply blur
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 2)
    
    # Set default min_dist if not provided
    if min_dist is None:
        min_dist = max(100, min(gray.shape) // 4)
    
    # Calculate radius range
    height, width = gray.shape
    min_radius = 5
    max_radius = int(min(height, width) / 3)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        center_x, center_y, _ = circles[0, 0]
        return (center_x, center_y)
    
    return None


def brightness_weighted_centroid(image: np.ndarray, percentile: float = 95) -> Tuple[int, int]:
    """
    Find center using brightness-weighted centroid of brightest pixels.
    
    From: segmentation.py, separation_linux.py, gradient_approach.py
    
    Args:
        image: Input grayscale image
        percentile: Percentile threshold for bright pixels
        
    Returns:
        (x, y) center coordinates
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Smooth the image
    smoothed = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Find bright pixels
    threshold = np.percentile(smoothed, percentile)
    bright_mask = smoothed > threshold
    
    if np.sum(bright_mask) == 0:
        # Fallback to image center
        h, w = gray.shape
        return (w // 2, h // 2)
    
    # Calculate weighted centroid
    y_coords, x_coords = np.where(bright_mask)
    weights = smoothed[bright_mask].astype(np.float64)
    
    center_x = int(np.average(x_coords, weights=weights))
    center_y = int(np.average(y_coords, weights=weights))
    
    return (center_x, center_y)


def morphological_center(image: np.ndarray) -> Tuple[int, int]:
    """
    Find center using morphological operations.
    
    From: gradient_approach.py
    
    Args:
        image: Input grayscale image
        
    Returns:
        (x, y) center coordinates
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply morphological operations to find the central structure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Opening to remove noise
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # Find the largest connected component
    _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate moments
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return (center_x, center_y)
    
    # Fallback to image center
    h, w = gray.shape
    return (w // 2, h // 2)


def gradient_based_center(image: np.ndarray, iterations: int = 10) -> Tuple[int, int]:
    """
    Find center by optimizing radial gradient alignment.
    
    From: gradient_approach.py
    
    Args:
        image: Input grayscale image
        iterations: Number of optimization iterations
        
    Returns:
        (x, y) center coordinates
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    def objective(center):
        cx, cy = center
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return 1e6  # Large penalty for out-of-bounds
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[:h, :w]
        
        # Calculate radial directions
        dx_radial = (x_grid - cx).astype(np.float64)
        dy_radial = (y_grid - cy).astype(np.float64)
        
        # Normalize radial directions
        radial_magnitude = np.sqrt(dx_radial**2 + dy_radial**2)
        radial_magnitude[radial_magnitude == 0] = 1  # Avoid division by zero
        
        dx_radial_norm = dx_radial / radial_magnitude
        dy_radial_norm = dy_radial / radial_magnitude
        
        # Calculate alignment score
        alignment = grad_x * dx_radial_norm + grad_y * dy_radial_norm
        
        # Return negative alignment (minimize negative = maximize positive)
        return -np.mean(np.abs(alignment))
    
    # Initial guess at image center
    initial_center = [w // 2, h // 2]
    
    # Optimize
    try:
        result = minimize(objective, initial_center, method='Powell', 
                         options={'maxiter': iterations})
        center_x, center_y = result.x
        return (int(center_x), int(center_y))
    except:
        # Fallback to initial guess
        return (w // 2, h // 2)


def edge_based_center(image: np.ndarray, sample_size: int = 100) -> Tuple[int, int]:
    """
    Find center using edge points and geometric fitting.
    
    From: gradient_approach.py, computational_separation.py
    
    Args:
        image: Input grayscale image
        sample_size: Number of edge points to sample
        
    Returns:
        (x, y) center coordinates
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150)
    edge_points = np.column_stack(np.where(edges)[::-1])  # (x, y) format
    
    if len(edge_points) < 3:
        h, w = gray.shape
        return (w // 2, h // 2)
    
    # Sample edge points if too many
    if len(edge_points) > sample_size:
        indices = np.random.choice(len(edge_points), sample_size, replace=False)
        edge_points = edge_points[indices]
    
    # Fit circle to edge points using least squares
    def fit_circle(points):
        x, y = points[:, 0], points[:, 1]
        
        # Set up the system Ax = b
        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        b = x**2 + y**2
        
        try:
            # Solve using least squares
            params = np.linalg.lstsq(A, b, rcond=None)[0]
            center_x, center_y = params[0], params[1]
            return (int(center_x), int(center_y))
        except:
            # Fallback to centroid
            return (int(np.mean(x)), int(np.mean(y)))
    
    return fit_circle(edge_points)


def multi_method_center_fusion(image: np.ndarray, methods: Optional[List[str]] = None) -> Tuple[int, int]:
    """
    Find center using multiple methods and fuse results.
    
    From: segmentation.py, separation_linux.py
    
    Args:
        image: Input grayscale image
        methods: List of methods to use (None = all methods)
        
    Returns:
        (x, y) fused center coordinates
    """
    if methods is None:
        methods = ['hough', 'brightness', 'morphological', 'gradient', 'edge']
    
    centers = []
    weights = []
    
    # Hough circles (most reliable for good images)
    if 'hough' in methods:
        hough_center = detect_center_hough_circles(image)
        if hough_center is not None:
            centers.append(hough_center)
            weights.append(3.0)  # High weight
    
    # Brightness centroid (good for bright cores)
    if 'brightness' in methods:
        bright_center = brightness_weighted_centroid(image)
        centers.append(bright_center)
        weights.append(2.0)  # Medium-high weight
    
    # Morphological center (robust to noise)
    if 'morphological' in methods:
        morph_center = morphological_center(image)
        centers.append(morph_center)
        weights.append(1.5)  # Medium weight
    
    # Gradient-based center (good for clear boundaries)
    if 'gradient' in methods:
        grad_center = gradient_based_center(image)
        centers.append(grad_center)
        weights.append(2.0)  # Medium-high weight
    
    # Edge-based center (geometric approach)
    if 'edge' in methods:
        edge_center = edge_based_center(image)
        centers.append(edge_center)
        weights.append(1.0)  # Lower weight (can be noisy)
    
    if not centers:
        # Fallback to image center
        h, w = image.shape[:2]
        return (w // 2, h // 2)
    
    # Weighted average of all centers
    centers = np.array(centers)
    weights = np.array(weights)
    
    # Remove outliers (centers more than 2 std deviations away)
    if len(centers) > 2:
        center_mean = np.mean(centers, axis=0)
        distances = np.linalg.norm(centers - center_mean, axis=1)
        std_dist = np.std(distances)
        
        if std_dist > 0:
            mask = distances <= (np.mean(distances) + 2 * std_dist)
            centers = centers[mask]
            weights = weights[mask]
    
    # Calculate weighted average
    weighted_center = np.average(centers, axis=0, weights=weights)
    
    return (int(weighted_center[0]), int(weighted_center[1]))


def validate_center(image: np.ndarray, center: Tuple[int, int], 
                   min_brightness_ratio: float = 1.2) -> bool:
    """
    Validate if a center point is reasonable for fiber optic analysis.
    
    From: bright_core_extractor.py
    
    Args:
        image: Input grayscale image
        center: (x, y) center coordinates to validate
        min_brightness_ratio: Minimum brightness ratio for validation
        
    Returns:
        True if center is valid, False otherwise
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    center_x, center_y = center
    
    # Check if center is within image bounds
    if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
        return False
    
    # Check local contrast around the center
    radius = 20
    outer_ring_width = 15
    
    # Create inner and outer masks
    inner_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(inner_mask, (center_x, center_y), radius, 255, -1)
    
    outer_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(outer_mask, (center_x, center_y), radius + outer_ring_width, 255, -1)
    outer_mask = cv2.subtract(outer_mask, inner_mask)
    
    # Calculate mean intensities
    inner_pixels = gray[inner_mask > 0]
    outer_pixels = gray[outer_mask > 0]
    
    if len(inner_pixels) == 0 or len(outer_pixels) == 0:
        return False
    
    inner_mean = np.mean(inner_pixels)
    outer_mean = np.mean(outer_pixels)
    
    # Check if inner region is sufficiently brighter
    return inner_mean > outer_mean * min_brightness_ratio


def main():
    """Test the center detection functions"""
    # Create a test image with a bright circular center
    test_image = np.zeros((400, 400), dtype=np.uint8)
    
    # Add background noise
    test_image = test_image + np.random.randint(0, 50, test_image.shape).astype(np.uint8)
    
    # Add bright circular center
    center = (200, 200)
    cv2.circle(test_image, center, 80, 200, -1)  # Bright core
    cv2.circle(test_image, center, 120, 150, 15)  # Ring structure
    
    print("Testing Center Detection Functions...")
    print(f"True center: {center}")
    
    # Test Hough circles
    hough_center = detect_center_hough_circles(test_image)
    print(f"✓ Hough circles: {hough_center}")
    
    # Test brightness centroid
    bright_center = brightness_weighted_centroid(test_image)
    print(f"✓ Brightness centroid: {bright_center}")
    
    # Test morphological center
    morph_center = morphological_center(test_image)
    print(f"✓ Morphological center: {morph_center}")
    
    # Test gradient-based center
    grad_center = gradient_based_center(test_image)
    print(f"✓ Gradient-based center: {grad_center}")
    
    # Test edge-based center
    edge_center = edge_based_center(test_image)
    print(f"✓ Edge-based center: {edge_center}")
    
    # Test multi-method fusion
    fused_center = multi_method_center_fusion(test_image)
    print(f"✓ Multi-method fusion: {fused_center}")
    
    # Test validation
    is_valid = validate_center(test_image, fused_center)
    print(f"✓ Center validation: {is_valid}")
    
    print("\nAll center detection functions tested successfully!")


if __name__ == "__main__":
    main()
