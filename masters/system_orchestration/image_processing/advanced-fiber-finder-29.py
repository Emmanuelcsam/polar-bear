#!/usr/bin/env python3
"""
Advanced Fiber Structure Detection Module
========================================
Sophisticated algorithms for detecting fiber optic structures including
cladding and core detection with multiple fallback methods.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from scipy import ndimage

try:
    import circle_fit as cf
    CIRCLE_FIT_AVAILABLE = True
except ImportError:
    CIRCLE_FIT_AVAILABLE = False

def detect_core_improved(image: np.ndarray, 
                        cladding_center: Tuple[float, float], 
                        cladding_radius: float,
                        core_diameter_hint: Optional[float] = None) -> Tuple[Tuple[float, float], float]:
    """
    Enhanced core detection with intensity-based analysis and multiple fallback methods.
    
    Args:
        image: Input image (grayscale)
        cladding_center: Center of cladding (x, y)
        cladding_radius: Radius of cladding
        core_diameter_hint: Optional hint for core diameter
        
    Returns:
        Tuple of (core_center, core_diameter)
    """
    # Create tighter cladding mask focusing on actual core region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Use smaller search radius to focus on core area
    cv2.circle(mask, tuple(map(int, cladding_center)), int(cladding_radius * 0.3), 255, -1)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    gray_smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Intensity-based core detection
    masked_region = cv2.bitwise_and(gray_smooth, mask)
    
    # Calculate radial intensity profile
    cy, cx = int(cladding_center[1]), int(cladding_center[0])
    Y, X = np.ogrid[:gray.shape[0], :gray.shape[1]]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Sample intensity values at different radii
    max_radius = int(cladding_radius * 0.3)
    intensity_profile = []
    
    for r in range(0, max_radius, 2):
        ring_mask = (dist_from_center >= r) & (dist_from_center < r + 2) & (mask > 0)
        if np.any(ring_mask):
            mean_intensity = np.mean(gray_smooth[ring_mask])
            intensity_profile.append((r, mean_intensity))
    
    if len(intensity_profile) > 5:
        # Find the radius where intensity changes significantly
        radii = np.array([p[0] for p in intensity_profile])
        intensities = np.array([p[1] for p in intensity_profile])
        
        # Calculate intensity gradient
        gradient = np.gradient(intensities)
        
        # Find the maximum gradient location (core-cladding boundary)
        max_gradient_idx = np.argmax(np.abs(gradient[1:-1])) + 1
        
        if max_gradient_idx < len(radii) - 1:
            core_radius = radii[max_gradient_idx]
            
            # Validate the detected radius
            if 3 < core_radius < cladding_radius * 0.15:  # Reasonable core size
                return tuple(cladding_center), core_radius * 2
    
    # Method 1: Adaptive thresholding
    try:
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        adaptive_thresh = cv2.bitwise_and(adaptive_thresh, mask)
        
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area and circularity
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.3 < circularity < 1.2 and area > 100:  # Reasonable circularity and size
                        valid_contours.append(contour)
            
            if valid_contours:
                # Find the most central contour
                best_contour = None
                min_distance = float('inf')
                
                for contour in valid_contours:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    distance = np.sqrt((x - cladding_center[0])**2 + (y - cladding_center[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_contour = contour
                
                if best_contour is not None:
                    (x, y), radius = cv2.minEnclosingCircle(best_contour)
                    return (x, y), radius * 2  # Return diameter
    except Exception as e:
        logging.debug(f"Adaptive threshold method failed: {e}")
    
    # Method 2: Edge-based detection
    try:
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, mask)
        
        # Dilate to connect edge fragments
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Similar filtering as above
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum area threshold
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    distance = np.sqrt((x - cladding_center[0])**2 + (y - cladding_center[1])**2)
                    if distance < cladding_radius * 0.3:  # Core should be near center
                        return (x, y), radius * 2
    except Exception as e:
        logging.debug(f"Edge-based method failed: {e}")
    
    # Method 3: Improved fallback based on cladding
    logging.debug("Using improved fallback method for core detection")
    if core_diameter_hint:
        core_radius = core_diameter_hint / 2
    else:
        # Better estimation for single-mode fibers (typically 9µm core in 125µm cladding)
        core_radius = cladding_radius * 0.072
    
    return tuple(cladding_center), core_radius * 2

def locate_fiber_structure_advanced(processed_image: np.ndarray,
                                   original_gray_image: Optional[np.ndarray] = None,
                                   hough_params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Advanced fiber structure localization using multiple detection methods.
    
    Args:
        processed_image: Preprocessed grayscale image
        original_gray_image: Original grayscale image for better core detection
        hough_params: Parameters for Hough circle detection
        
    Returns:
        Dictionary containing localization results or None if failed
    """
    h, w = processed_image.shape[:2]
    min_img_dim = min(h, w)
    
    # Default Hough parameters
    if hough_params is None:
        hough_params = {
            "dp": 1.2,
            "min_dist_factor": 0.15,
            "param1": 70,
            "param2": 35,
            "min_radius_factor": 0.08,
            "max_radius_factor": 0.45
        }
    
    # Calculate parameters
    min_dist_circles = int(min_img_dim * hough_params["min_dist_factor"])
    min_radius_hough = int(min_img_dim * hough_params["min_radius_factor"])
    max_radius_hough = int(min_img_dim * hough_params["max_radius_factor"])
    
    localization_result = {}
    
    # Method 1: HoughCircles detection
    logging.info("Attempting cladding detection using HoughCircles...")
    circles = cv2.HoughCircles(
        processed_image,
        cv2.HOUGH_GRADIENT,
        dp=hough_params["dp"],
        minDist=min_dist_circles,
        param1=hough_params["param1"],
        param2=hough_params["param2"],
        minRadius=min_radius_hough,
        maxRadius=max_radius_hough
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) > 0:
            x_cl, y_cl, r_cl = circles[0]
            localization_result['cladding_center_xy'] = (x_cl, y_cl)
            localization_result['cladding_radius_px'] = float(r_cl)
            localization_result['localization_method'] = 'HoughCircles'
            logging.info(f"Cladding (HoughCircles): Center=({x_cl},{y_cl}), Radius={r_cl}px")
    
    # Method 2: Contour-based detection
    if 'cladding_center_xy' not in localization_result:
        logging.info("HoughCircles failed, attempting contour-based detection...")
        try:
            edges = cv2.Canny(processed_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (likely to be the cladding)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Fit ellipse to the largest contour
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    (x_cl, y_cl), (minor_axis, major_axis), angle = ellipse
                    
                    # Check if it's reasonably circular
                    axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
                    if axis_ratio > 0.7:  # Reasonably circular
                        avg_radius = (minor_axis + major_axis) / 4.0
                        localization_result['cladding_center_xy'] = (int(x_cl), int(y_cl))
                        localization_result['cladding_radius_px'] = float(avg_radius)
                        localization_result['cladding_ellipse_params'] = ellipse
                        localization_result['localization_method'] = 'ContourFitCircle'
                        logging.info(f"Cladding (Contour): Center=({int(x_cl)},{int(y_cl)}), Radius={avg_radius:.1f}px")
        except Exception as e:
            logging.error(f"Contour-based detection failed: {e}")
    
    # Method 3: Circle-fit library fallback
    if 'cladding_center_xy' not in localization_result and CIRCLE_FIT_AVAILABLE:
        logging.info("Attempting circle-fit library method...")
        try:
            edges = cv2.Canny(processed_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Combine all contour points
                all_points = np.vstack(contours).reshape(-1, 2)
                
                if len(all_points) > 10:
                    # Use circle_fit library for robust fitting
                    xc_cf, yc_cf, r_cf, residual_cf = cf.hyper_fit(all_points)
                    
                    # Validate the result
                    if 0 < r_cf < min_img_dim * 0.5 and residual_cf < 50:
                        localization_result['cladding_center_xy'] = (int(xc_cf), int(yc_cf))
                        localization_result['cladding_radius_px'] = float(r_cf)
                        localization_result['localization_method'] = 'CircleFitLib'
                        localization_result['fit_residual'] = residual_cf
                        logging.info(f"Cladding (CircleFitLib): Center=({int(xc_cf)},{int(yc_cf)}), Radius={r_cf:.1f}px")
        except Exception as e:
            logging.error(f"Circle-fit library method failed: {e}")
    
    # Check if cladding was found
    if 'cladding_center_xy' not in localization_result:
        logging.error("Failed to localize fiber cladding by any method.")
        return None
    
    # Core Detection
    logging.info("Starting enhanced core detection...")
    
    image_for_core_detect = original_gray_image if original_gray_image is not None else processed_image
    cladding_center = localization_result['cladding_center_xy']
    cladding_radius = localization_result['cladding_radius_px']
    
    try:
        core_center, core_diameter = detect_core_improved(
            image_for_core_detect, 
            cladding_center, 
            cladding_radius
        )
        
        localization_result['core_center_xy'] = tuple(map(int, core_center))
        localization_result['core_radius_px'] = float(core_diameter / 2)
        logging.info(f"Core detected: Center=({int(core_center[0])},{int(core_center[1])}), Radius={core_diameter/2:.1f}px")
        
    except Exception as e:
        logging.error(f"Enhanced core detection failed: {e}")
        # Final fallback
        localization_result['core_center_xy'] = localization_result['cladding_center_xy']
        localization_result['core_radius_px'] = localization_result['cladding_radius_px'] * 0.072
        logging.warning("Core detection failed, using fallback estimation")
    
    return localization_result

def generate_fiber_zone_masks(image_shape: Tuple[int, int],
                             cladding_center: Tuple[float, float],
                             cladding_radius: float,
                             core_center: Optional[Tuple[float, float]] = None,
                             core_radius: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Generate binary masks for fiber zones (Core and Cladding).
    
    Args:
        image_shape: Shape of the image (height, width)
        cladding_center: Center of cladding (x, y)
        cladding_radius: Radius of cladding
        core_center: Center of core (x, y), defaults to cladding center
        core_radius: Radius of core, defaults to 0.072 * cladding_radius
        
    Returns:
        Dictionary of zone masks
    """
    masks = {}
    h, w = image_shape[:2]
    Y, X = np.ogrid[:h, :w]
    
    # Use defaults if not provided
    if core_center is None:
        core_center = cladding_center
    if core_radius is None:
        core_radius = cladding_radius * 0.072  # Typical single-mode ratio
    
    cx, cy = cladding_center
    core_cx, core_cy = core_center
    
    # Distance calculations
    dist_sq_from_cladding = (X - cx)**2 + (Y - cy)**2
    dist_sq_from_core = (X - core_cx)**2 + (Y - core_cy)**2
    
    # Create Core mask
    masks["Core"] = (dist_sq_from_core <= core_radius**2).astype(np.uint8) * 255
    
    # Create Cladding mask (excluding core)
    cladding_mask = (dist_sq_from_cladding <= cladding_radius**2).astype(np.uint8)
    core_mask = (dist_sq_from_core <= core_radius**2).astype(np.uint8)
    masks["Cladding"] = (cladding_mask - core_mask) * 255
    
    logging.info(f"Generated zone masks - Core radius: {core_radius:.1f}px, Cladding radius: {cladding_radius:.1f}px")
    
    return masks

def validate_fiber_detection(localization_data: Dict[str, Any],
                            image_shape: Tuple[int, int]) -> bool:
    """
    Validate the detected fiber structure parameters.
    
    Args:
        localization_data: Fiber localization results
        image_shape: Shape of the image
        
    Returns:
        True if validation passes, False otherwise
    """
    h, w = image_shape[:2]
    min_dim = min(h, w)
    
    # Check if required data exists
    if 'cladding_center_xy' not in localization_data or 'cladding_radius_px' not in localization_data:
        return False
    
    center_x, center_y = localization_data['cladding_center_xy']
    radius = localization_data['cladding_radius_px']
    
    # Validate center is within image bounds
    if not (0 <= center_x < w and 0 <= center_y < h):
        logging.warning(f"Cladding center ({center_x}, {center_y}) is outside image bounds")
        return False
    
    # Validate radius is reasonable
    if not (min_dim * 0.05 <= radius <= min_dim * 0.5):
        logging.warning(f"Cladding radius {radius} is not within reasonable bounds")
        return False
    
    # Check if fiber fits within image
    if (center_x - radius < 0 or center_x + radius >= w or 
        center_y - radius < 0 or center_y + radius >= h):
        logging.warning("Detected fiber extends beyond image bounds")
        return False
    
    # Validate core if present
    if 'core_radius_px' in localization_data:
        core_radius = localization_data['core_radius_px']
        if core_radius >= radius:
            logging.warning(f"Core radius {core_radius} >= cladding radius {radius}")
            return False
    
    return True

if __name__ == "__main__":
    """Test the fiber detection functions"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Create a test fiber image
    test_image = np.zeros((200, 200), dtype=np.uint8)
    
    # Create cladding (outer circle)
    cv2.circle(test_image, (100, 100), 80, 150, -1)
    
    # Create core (inner circle)
    cv2.circle(test_image, (100, 100), 6, 100, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_image.shape).astype(np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print("Testing fiber structure detection...")
    
    # Test localization
    localization = locate_fiber_structure_advanced(test_image, test_image)
    
    if localization:
        print(f"Localization successful: {localization}")
        
        # Validate detection
        is_valid = validate_fiber_detection(localization, test_image.shape)
        print(f"Validation result: {is_valid}")
        
        # Generate zone masks
        masks = generate_fiber_zone_masks(
            test_image.shape,
            localization['cladding_center_xy'],
            localization['cladding_radius_px'],
            localization.get('core_center_xy'),
            localization.get('core_radius_px')
        )
        
        print(f"Generated masks for zones: {list(masks.keys())}")
        
        # Test core detection directly
        cladding_center = localization['cladding_center_xy']
        cladding_radius = localization['cladding_radius_px']
        
        core_center, core_diameter = detect_core_improved(
            test_image, cladding_center, cladding_radius
        )
        
        print(f"Direct core detection: Center={core_center}, Diameter={core_diameter:.2f}")
        
    else:
        print("Localization failed")
    
    print("Fiber detection tests completed!")
