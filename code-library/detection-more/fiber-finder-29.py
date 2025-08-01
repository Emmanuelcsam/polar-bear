#!/usr/bin/env python3
"""
Fiber Core and Cladding Detection Module
=======================================
Standalone module for detecting fiber core and cladding boundaries
using multiple detection methods.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Any, NamedTuple
from scipy.signal import find_peaks
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionResult(NamedTuple):
    """Result structure for fiber detection."""
    method_name: str
    center: Tuple[float, float]
    core_radius: float
    cladding_radius: float
    confidence: float
    details: Dict[str, Any]

def hough_circle_detection(image: np.ndarray, min_radius_factor: float = 0.08, 
                          max_radius_factor: float = 0.45) -> Optional[DetectionResult]:
    """
    Detect fiber core and cladding using Hough Circle Transform.
    
    Args:
        image: Input grayscale image
        min_radius_factor: Minimum radius as fraction of image size
        max_radius_factor: Maximum radius as fraction of image size
        
    Returns:
        Detection result or None
    """
    logger.info("Running Hough circle detection...")
    
    # Ensure uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    h, w = image.shape
    min_radius = int(min(h, w) * min_radius_factor)
    max_radius = int(min(h, w) * max_radius_factor)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, 
        minDist=int(min(h, w) * 0.15),
        param1=70, param2=35,
        minRadius=min_radius, maxRadius=max_radius
    )
    
    if circles is not None and len(circles[0]) >= 2:
        circles = np.round(circles[0, :]).astype("int")
        
        # Sort by radius (smaller first for core, larger for cladding)
        circles = sorted(circles, key=lambda x: x[2])
        
        core_circle = circles[0]
        cladding_circle = circles[-1]  # Largest circle
        
        # Calculate confidence based on circularity and concentricity
        confidence = calculate_circle_confidence(image, core_circle, cladding_circle)
        
        return DetectionResult(
            method_name='hough_circles',
            center=((core_circle[0] + cladding_circle[0]) / 2, 
                   (core_circle[1] + cladding_circle[1]) / 2),
            core_radius=float(core_circle[2]),
            cladding_radius=float(cladding_circle[2]),
            confidence=confidence,
            details={'num_circles_found': len(circles)}
        )
    
    logger.warning("Hough circle detection failed")
    return None

def radial_profile_detection(image: np.ndarray) -> Optional[DetectionResult]:
    """
    Detect fiber boundaries using radial intensity profile analysis.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Detection result or None
    """
    logger.info("Running radial profile detection...")
    
    h, w = image.shape
    center_x, center_y = w // 2, h // 2
    max_radius = min(h, w) // 2 - 10
    
    # Calculate radial profile
    radial_profile = []
    for r in range(max_radius):
        # Create circular mask
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= r**2
        inner_mask = (x - center_x)**2 + (y - center_y)**2 <= (r-1)**2 if r > 0 else np.zeros_like(mask)
        
        # Ring mask
        ring_mask = mask & ~inner_mask
        
        if np.any(ring_mask):
            radial_profile.append(np.mean(image[ring_mask]))
        else:
            radial_profile.append(0)
    
    radial_profile = np.array(radial_profile)
    
    if len(radial_profile) < 10:
        return None
    
    # Smooth the profile
    from scipy.ndimage import gaussian_filter1d
    smoothed_profile = gaussian_filter1d(radial_profile, sigma=2)
    
    # Find boundaries using gradient peaks
    gradient = np.gradient(smoothed_profile)
    peaks, properties = find_peaks(np.abs(gradient), prominence=np.std(gradient) * 0.5)
    
    if len(peaks) >= 2:
        # Take the two most prominent peaks
        prominences = properties['prominences']
        sorted_indices = np.argsort(prominences)[-2:]
        core_radius = float(peaks[sorted_indices[0]])
        cladding_radius = float(peaks[sorted_indices[1]])
        
        # Ensure proper ordering
        if core_radius > cladding_radius:
            core_radius, cladding_radius = cladding_radius, core_radius
        
        confidence = 0.7  # Base confidence for radial method
        
        return DetectionResult(
            method_name='radial_profile',
            center=(float(center_x), float(center_y)),
            core_radius=core_radius,
            cladding_radius=cladding_radius,
            confidence=confidence,
            details={'num_peaks': len(peaks)}
        )
    
    logger.warning("Radial profile detection failed")
    return None

def edge_based_detection(image: np.ndarray) -> Optional[DetectionResult]:
    """
    Detect fiber boundaries using edge detection and circle fitting.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Detection result or None
    """
    logger.info("Running edge-based detection...")
    
    # Ensure uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # Get edge points
    edge_points = np.column_stack(np.where(edges.T))
    
    if len(edge_points) < 100:
        return None
    
    # RANSAC-like circle fitting
    best_circles = []
    
    for _ in range(50):  # Random trials
        if len(edge_points) < 3:
            continue
            
        # Sample 3 points randomly
        indices = np.random.choice(len(edge_points), 3, replace=False)
        p1, p2, p3 = edge_points[indices]
        
        # Calculate circle from 3 points
        circle = circle_from_three_points(p1, p2, p3)
        if circle is None:
            continue
            
        cx, cy, r = circle
        
        # Count inliers
        distances = np.sqrt((edge_points[:, 0] - cx)**2 + (edge_points[:, 1] - cy)**2)
        inliers = np.abs(distances - r) < 3  # 3 pixel tolerance
        inlier_count = np.sum(inliers)
        
        if inlier_count > len(edge_points) * 0.1:  # At least 10% inliers
            best_circles.append((cx, cy, r, inlier_count))
    
    if len(best_circles) >= 2:
        # Sort by inlier count and take top 2
        best_circles.sort(key=lambda x: x[3], reverse=True)
        best_circles = sorted(best_circles[:2], key=lambda x: x[2])  # Sort by radius
        
        inner = best_circles[0]
        outer = best_circles[1]
        
        # Average centers
        center_x = (inner[0] + outer[0]) / 2
        center_y = (inner[1] + outer[1]) / 2
        
        return DetectionResult(
            method_name='edge_based',
            center=(center_x, center_y),
            core_radius=float(inner[2]),
            cladding_radius=float(outer[2]),
            confidence=0.7,
            details={'num_circles_found': len(best_circles)}
        )
    
    logger.warning("Edge-based detection failed")
    return None

def intensity_based_detection(image: np.ndarray, core_percentile: float = 75, 
                             cladding_percentile: float = 50) -> Optional[DetectionResult]:
    """
    Detect fiber boundaries using intensity thresholding.
    
    Args:
        image: Input grayscale image
        core_percentile: Percentile for core threshold
        cladding_percentile: Percentile for cladding threshold
        
    Returns:
        Detection result or None
    """
    logger.info("Running intensity-based detection...")
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    h, w = image.shape
    
    # Multi-level thresholding
    core_thresh = np.percentile(image, core_percentile)
    clad_thresh = np.percentile(image, cladding_percentile)
    
    # Create masks
    core_mask = image > core_thresh
    clad_mask = image > clad_thresh
    
    # Clean masks with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core_mask = cv2.morphologyEx(core_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    clad_mask = cv2.morphologyEx(clad_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    core_contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clad_contours, _ = cv2.findContours(clad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if core_contours and clad_contours:
        # Get largest contours
        core_contour = max(core_contours, key=cv2.contourArea)
        clad_contour = max(clad_contours, key=cv2.contourArea)
        
        # Fit circles
        (cx1, cy1), r1 = cv2.minEnclosingCircle(core_contour)
        (cx2, cy2), r2 = cv2.minEnclosingCircle(clad_contour)
        
        # Average centers
        center_x = (cx1 + cx2) / 2
        center_y = (cy1 + cy2) / 2
        
        return DetectionResult(
            method_name='intensity_based',
            center=(center_x, center_y),
            core_radius=float(r1),
            cladding_radius=float(r2),
            confidence=0.6,
            details={'core_area': cv2.contourArea(core_contour), 'clad_area': cv2.contourArea(clad_contour)}
        )
    
    logger.warning("Intensity-based detection failed")
    return None

def morphological_detection(image: np.ndarray) -> Optional[DetectionResult]:
    """
    Detect fiber boundaries using morphological operations.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Detection result or None
    """
    logger.info("Running morphological detection...")
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Apply morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    
    # Threshold gradient
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply closing to connect edges
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find circular regions
    circular_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
            
        # Check circularity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            
            if solidity > 0.8:  # High solidity indicates circular shape
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circular_regions.append((x, y, radius, solidity))
    
    if len(circular_regions) >= 2:
        # Sort by radius
        circular_regions.sort(key=lambda x: x[2])
        
        inner = circular_regions[0]
        outer = circular_regions[-1]
        
        center_x = (inner[0] + outer[0]) / 2
        center_y = (inner[1] + outer[1]) / 2
        
        return DetectionResult(
            method_name='morphological',
            center=(center_x, center_y),
            core_radius=float(inner[2]),
            cladding_radius=float(outer[2]),
            confidence=0.6,
            details={'num_circular_regions': len(circular_regions)}
        )
    
    logger.warning("Morphological detection failed")
    return None

def ensemble_detection(image: np.ndarray, methods: List[str] = None, 
                      weights: Dict[str, float] = None) -> Optional[DetectionResult]:
    """
    Ensemble detection combining multiple methods.
    
    Args:
        image: Input grayscale image
        methods: List of methods to use
        weights: Weights for each method
        
    Returns:
        Best detection result
    """
    if methods is None:
        methods = ['hough', 'radial', 'edge', 'intensity', 'morphological']
    
    if weights is None:
        weights = {
            'hough': 1.0,
            'radial': 0.8,
            'edge': 0.7,
            'intensity': 0.6,
            'morphological': 0.5
        }
    
    logger.info(f"Running ensemble detection with methods: {methods}")
    
    # Method registry
    method_functions = {
        'hough': hough_circle_detection,
        'radial': radial_profile_detection,
        'edge': edge_based_detection,
        'intensity': intensity_based_detection,
        'morphological': morphological_detection
    }
    
    results = []
    
    # Run all methods
    for method in methods:
        if method in method_functions:
            try:
                result = method_functions[method](image)
                if result is not None:
                    # Weight the confidence
                    weighted_confidence = result.confidence * weights.get(method, 1.0)
                    results.append(result._replace(confidence=weighted_confidence))
                    logger.info(f"{method} detection successful (confidence: {weighted_confidence:.3f})")
                else:
                    logger.warning(f"{method} detection failed")
            except Exception as e:
                logger.error(f"Error in {method} detection: {e}")
    
    if not results:
        logger.error("All detection methods failed")
        return None
    
    # Select best result based on confidence
    best_result = max(results, key=lambda x: x.confidence)
    
    logger.info(f"Best method: {best_result.method_name} (confidence: {best_result.confidence:.3f})")
    
    return best_result

# Helper functions
def circle_from_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Calculate circle parameters from three points."""
    try:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Calculate circle center
        D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        
        if abs(D) < 1e-7:  # Points are collinear
            return None
        
        ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
        uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
        
        # Calculate radius
        radius = np.sqrt((x1 - ux)**2 + (y1 - uy)**2)
        
        return (ux, uy, radius)
        
    except Exception:
        return None

def calculate_circle_confidence(image: np.ndarray, core_circle: np.ndarray, 
                               cladding_circle: np.ndarray) -> float:
    """Calculate confidence score for detected circles."""
    try:
        # Circularity measure
        core_mask = np.zeros_like(image)
        cv2.circle(core_mask, (core_circle[0], core_circle[1]), core_circle[2], 255, -1)
        
        clad_mask = np.zeros_like(image)
        cv2.circle(clad_mask, (cladding_circle[0], cladding_circle[1]), cladding_circle[2], 255, -1)
        
        # Calculate concentricity
        core_center = np.array([core_circle[0], core_circle[1]])
        clad_center = np.array([cladding_circle[0], cladding_circle[1]])
        concentricity = 1.0 / (1.0 + np.linalg.norm(core_center - clad_center))
        
        # Calculate radius ratio (should be around 0.1-0.2 for typical fibers)
        radius_ratio = core_circle[2] / cladding_circle[2] if cladding_circle[2] > 0 else 0
        ratio_score = 1.0 - abs(radius_ratio - 0.15) / 0.15 if radius_ratio > 0 else 0
        
        # Combine scores
        confidence = 0.5 * concentricity + 0.5 * max(0, ratio_score)
        
        return min(1.0, max(0.0, confidence))
        
    except Exception:
        return 0.5

def test_fiber_detection():
    """Test fiber detection functions."""
    logger.info("Testing fiber core and cladding detection...")
    
    # Create synthetic fiber image
    test_image = np.zeros((256, 256), dtype=np.uint8)
    
    # Add cladding (outer circle)
    cv2.circle(test_image, (128, 128), 80, 120, -1)
    
    # Add core (inner circle)
    cv2.circle(test_image, (128, 128), 15, 200, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_image.shape)
    test_image = np.clip(test_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    logger.info(f"Test image shape: {test_image.shape}")
    
    # Test individual methods
    methods = ['hough', 'radial', 'edge', 'intensity', 'morphological']
    results = {}
    
    for method in methods:
        try:
            if method == 'hough':
                result = hough_circle_detection(test_image)
            elif method == 'radial':
                result = radial_profile_detection(test_image)
            elif method == 'edge':
                result = edge_based_detection(test_image)
            elif method == 'intensity':
                result = intensity_based_detection(test_image)
            elif method == 'morphological':
                result = morphological_detection(test_image)
            
            results[method] = result
            
            if result:
                logger.info(f"{method}: Core radius={result.core_radius:.1f}, "
                           f"Cladding radius={result.cladding_radius:.1f}, "
                           f"Confidence={result.confidence:.3f}")
            else:
                logger.info(f"{method}: Detection failed")
                
        except Exception as e:
            logger.error(f"Error testing {method}: {e}")
    
    # Test ensemble method
    ensemble_result = ensemble_detection(test_image, methods)
    
    if ensemble_result:
        logger.info(f"Ensemble: Best method={ensemble_result.method_name}, "
                   f"Core radius={ensemble_result.core_radius:.1f}, "
                   f"Cladding radius={ensemble_result.cladding_radius:.1f}, "
                   f"Confidence={ensemble_result.confidence:.3f}")
    
    logger.info("All fiber detection tests completed!")
    
    return {
        'test_image': test_image,
        'individual_results': results,
        'ensemble_result': ensemble_result
    }

if __name__ == "__main__":
    # Run tests
    test_results = test_fiber_detection()
    logger.info("Fiber detection module is ready for use!")
