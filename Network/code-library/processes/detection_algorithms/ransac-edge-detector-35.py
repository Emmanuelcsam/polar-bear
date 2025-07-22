#!/usr/bin/env python3
"""
Edge Detection and RANSAC Functions
Extracted from multiple fiber optic analysis scripts

This module contains edge detection and geometric fitting functions:
- Canny edge detection with parameter optimization
- RANSAC circle fitting for robust parameter estimation
- Edge point extraction and preprocessing
- Geometric circle parameter refinement
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
from scipy.optimize import least_squares


def extract_edge_points(image: np.ndarray, sigma: float = 1.5, 
                       low_threshold: int = 30, high_threshold: int = 100) -> np.ndarray:
    """
    Extract edge points using Canny edge detection.
    
    From: computational_separation.py
    
    Args:
        image: Input grayscale image
        sigma: Gaussian blur sigma for preprocessing
        low_threshold: Lower threshold for Canny
        high_threshold: Upper threshold for Canny
        
    Returns:
        Array of edge points as (x, y) coordinates
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur if sigma > 0
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Return edge points as (x, y)
    return np.column_stack(np.where(edges)[::-1])


def adaptive_canny_thresholds(image: np.ndarray) -> Tuple[int, int]:
    """
    Calculate adaptive Canny thresholds based on image statistics.
    
    Args:
        image: Input grayscale image
        
    Returns:
        (low_threshold, high_threshold) for Canny edge detection
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate image statistics
    median = np.median(gray)
    sigma = 0.33
    
    # Calculate thresholds
    low_threshold = int(max(0, (1.0 - sigma) * median))
    high_threshold = int(min(255, (1.0 + sigma) * median))
    
    return low_threshold, high_threshold


def ransac_circle_fitting(points: np.ndarray, max_iterations: int = 1000, 
                         distance_threshold: float = 2.0) -> Optional[Tuple[float, float, float]]:
    """
    Fit a single circle to points using RANSAC.
    
    From: computational_separation.py (modified)
    
    Args:
        points: Array of (x, y) points
        max_iterations: Maximum RANSAC iterations
        distance_threshold: Maximum distance for inliers
        
    Returns:
        (center_x, center_y, radius) or None if fitting fails
    """
    if len(points) < 3:
        return None
    
    best_inliers = 0
    best_circle = None
    
    for _ in range(max_iterations):
        # Sample 3 random points
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]
        
        # Fit circle to 3 points
        circle = fit_circle_to_three_points(sample_points)
        if circle is None:
            continue
        
        center_x, center_y, radius = circle
        
        # Check for reasonable radius
        if radius <= 0 or radius > 1000:
            continue
        
        # Calculate distances from all points to circle
        distances = np.abs(np.sqrt((points[:, 0] - center_x)**2 + 
                                 (points[:, 1] - center_y)**2) - radius)
        
        # Count inliers
        inliers = np.sum(distances < distance_threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_circle = circle
    
    return best_circle


def fit_circle_to_three_points(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Fit circle to exactly three points using circumcenter calculation.
    
    From: computational_separation.py
    
    Args:
        points: Array of 3 (x, y) points
        
    Returns:
        (center_x, center_y, radius) or None if points are collinear
    """
    if len(points) != 3:
        return None
    
    p1, p2, p3 = points
    
    # Calculate circumcenter
    d = 2 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    
    if abs(d) < 1e-6:  # Points are collinear
        return None
    
    cx = ((p1[0]**2+p1[1]**2)*(p2[1]-p3[1]) + (p2[0]**2+p2[1]**2)*(p3[1]-p1[1]) + 
          (p3[0]**2+p3[1]**2)*(p1[1]-p2[1])) / d
    cy = ((p1[0]**2+p1[1]**2)*(p3[0]-p2[0]) + (p2[0]**2+p2[1]**2)*(p1[0]-p3[0]) + 
          (p3[0]**2+p3[1]**2)*(p2[0]-p1[0])) / d
    
    # Calculate radius
    radius = np.sqrt((p1[0] - cx)**2 + (p1[1] - cy)**2)
    
    return (cx, cy, radius)


def ransac_two_circles_fitting(points: np.ndarray, max_iterations: int = 1000) -> Optional[List[float]]:
    """
    Fit two concentric circles using RANSAC and radial histogram analysis.
    
    From: computational_separation.py
    
    Args:
        points: Array of (x, y) edge points
        max_iterations: Maximum RANSAC iterations
        
    Returns:
        [center_x, center_y, radius1, radius2] or None if fitting fails
    """
    if len(points) < 6:
        return None
    
    best_score = -1
    best_params = None
    img_bounds = points.max(axis=0)
    
    for _ in range(max_iterations):
        # Sample 3 points to find circle center
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]
        
        # Calculate circumcenter
        circle = fit_circle_to_three_points(sample_points)
        if circle is None:
            continue
        
        cx, cy, _ = circle
        
        # Skip if center is far outside image bounds
        if (cx < -img_bounds[0] or cx > 2*img_bounds[0] or 
            cy < -img_bounds[1] or cy > 2*img_bounds[1]):
            continue
        
        # Score by radial histogram
        distances = np.linalg.norm(points - [cx, cy], axis=1)
        max_r = min(distances.max(), max(img_bounds))
        
        hist, bins = np.histogram(distances, bins=50, range=(0, max_r))
        peaks = np.argsort(hist)[-2:]  # Two highest bins
        score = hist[peaks].sum()
        
        if score > best_score:
            best_score = score
            r1, r2 = bins[peaks] + (bins[1]-bins[0])/2
            best_params = [cx, cy, min(r1, r2), max(r1, r2)]
    
    return best_params


def refine_circle_parameters(points: np.ndarray, initial_params: List[float], 
                           max_iterations: int = 50) -> List[float]:
    """
    Refine circle parameters using iterative least squares.
    
    From: computational_separation.py
    
    Args:
        points: Array of (x, y) edge points
        initial_params: [center_x, center_y, radius1, radius2]
        max_iterations: Maximum refinement iterations
        
    Returns:
        Refined [center_x, center_y, radius1, radius2]
    """
    if len(initial_params) != 4:
        return initial_params
    
    cx, cy, r1, r2 = initial_params
    img_bounds = points.max(axis=0)
    
    for _ in range(max_iterations):
        # Compute distances to center
        distances = np.linalg.norm(points - [cx, cy], axis=1)
        
        # Assign points to nearest circle
        mask1 = np.abs(distances - r1) < np.abs(distances - r2)
        pts1, pts2 = points[mask1], points[~mask1]
        
        # Update radii as mean distances (with bounds)
        if len(pts1) > 0:
            r1 = np.mean(np.linalg.norm(pts1 - [cx, cy], axis=1))
            r1 = min(r1, max(img_bounds))
        if len(pts2) > 0:
            r2 = np.mean(np.linalg.norm(pts2 - [cx, cy], axis=1))
            r2 = min(r2, max(img_bounds))
        
        # Update center as weighted mean
        if len(pts1) + len(pts2) > 0:
            all_pts = np.vstack([pts1, pts2]) if len(pts1) > 0 and len(pts2) > 0 else pts1 if len(pts1) > 0 else pts2
            cx, cy = np.mean(all_pts, axis=0)
    
    return [cx, cy, min(r1, r2), max(r1, r2)]


def least_squares_circle_fit(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Fit circle using least squares optimization.
    
    Args:
        points: Array of (x, y) points
        
    Returns:
        (center_x, center_y, radius) or None if fitting fails
    """
    if len(points) < 3:
        return None
    
    def residuals(params, points_arg):
        cx, cy, r = params
        distances = np.sqrt((points_arg[:, 0] - cx)**2 + (points_arg[:, 1] - cy)**2)
        return distances - r
    
    # Initial guess (centroid and mean distance)
    centroid = np.mean(points, axis=0)
    mean_distance = np.mean(np.linalg.norm(points - centroid, axis=1))
    initial_params = [centroid[0], centroid[1], mean_distance]
    
    try:
        result = least_squares(residuals, initial_params, args=(points,))
        cx, cy, r = result.x
        return (cx, cy, abs(r))
    except:
        return None


def filter_edge_points_by_gradient(image: np.ndarray, edge_points: np.ndarray, 
                                  min_gradient: float = 10.0) -> np.ndarray:
    """
    Filter edge points by gradient magnitude to keep only strong edges.
    
    Args:
        image: Input grayscale image
        edge_points: Array of (x, y) edge points
        min_gradient: Minimum gradient magnitude threshold
        
    Returns:
        Filtered edge points
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Filter points by gradient magnitude
    filtered_points = []
    for x, y in edge_points:
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            if gradient_magnitude[y, x] >= min_gradient:
                filtered_points.append([x, y])
    
    return np.array(filtered_points) if filtered_points else np.array([]).reshape(0, 2)


def extract_circular_edge_points(image: np.ndarray, center: Tuple[int, int], 
                                radius: float, thickness: int = 5) -> np.ndarray:
    """
    Extract edge points within a circular region around expected boundary.
    
    Args:
        image: Input grayscale image
        center: (x, y) center coordinates
        radius: Expected radius
        thickness: Thickness of the extraction band
        
    Returns:
        Array of edge points within the circular region
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Get all edge points
    low_thresh, high_thresh = adaptive_canny_thresholds(gray)
    edge_points = extract_edge_points(gray, sigma=1.0, 
                                    low_threshold=low_thresh, 
                                    high_threshold=high_thresh)
    
    if len(edge_points) == 0:
        return np.array([]).reshape(0, 2)
    
    # Calculate distances from center
    distances = np.linalg.norm(edge_points - np.array(center), axis=1)
    
    # Filter by distance (within thickness band around expected radius)
    mask = np.abs(distances - radius) <= thickness
    
    return edge_points[mask]


def main():
    """Test the edge detection and RANSAC functions"""
    # Create a test image with circular structures
    test_image = np.zeros((400, 400), dtype=np.uint8)
    
    # Add some background noise
    test_image = test_image + np.random.randint(0, 30, test_image.shape).astype(np.uint8)
    
    # Add circular structures
    center = (200, 200)
    cv2.circle(test_image, center, 80, 255, 3)   # Inner circle
    cv2.circle(test_image, center, 120, 255, 3)  # Outer circle
    
    print("Testing Edge Detection and RANSAC Functions...")
    
    # Test edge point extraction
    edge_points = extract_edge_points(test_image)
    print(f"✓ Extracted {len(edge_points)} edge points")
    
    # Test adaptive Canny thresholds
    low_thresh, high_thresh = adaptive_canny_thresholds(test_image)
    print(f"✓ Adaptive Canny thresholds: {low_thresh}, {high_thresh}")
    
    # Test single circle RANSAC
    single_circle = ransac_circle_fitting(edge_points)
    print(f"✓ Single circle RANSAC: {single_circle}")
    
    # Test two circles RANSAC
    two_circles = ransac_two_circles_fitting(edge_points)
    print(f"✓ Two circles RANSAC: {two_circles}")
    
    # Test least squares circle fit
    if len(edge_points) > 10:
        sample_points = edge_points[np.random.choice(len(edge_points), 10, replace=False)]
        ls_circle = least_squares_circle_fit(sample_points)
        print(f"✓ Least squares circle: {ls_circle}")
    
    # Test gradient filtering
    filtered_points = filter_edge_points_by_gradient(test_image, edge_points)
    print(f"✓ Gradient filtered points: {len(filtered_points)}")
    
    # Test circular edge extraction
    circular_points = extract_circular_edge_points(test_image, center, 80)
    print(f"✓ Circular edge points: {len(circular_points)}")
    
    # Test parameter refinement
    if two_circles is not None:
        refined_params = refine_circle_parameters(edge_points, two_circles)
        print(f"✓ Refined parameters: {refined_params}")
    
    print("\nAll edge detection and RANSAC functions tested successfully!")


if __name__ == "__main__":
    main()
