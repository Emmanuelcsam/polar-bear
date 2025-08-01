#!/usr/bin/env python3
"""
Advanced Morphological Operations Module
=======================================
Sophisticated morphological image processing operations including
safe thinning, skeletonization, and advanced filtering techniques.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List
from scipy import ndimage

def safe_thinning(binary_image: np.ndarray, method: str = "zhang_suen") -> np.ndarray:
    """
    Safe thinning implementation with multiple fallback methods.
    
    Args:
        binary_image: Input binary image
        method: Thinning method ("zhang_suen", "morphological", "opencv")
        
    Returns:
        Thinned binary image
    """
    if method == "opencv":
        try:
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
                return cv2.ximgproc.thinning(binary_image)
            else:
                logging.debug("OpenCV ximgproc not available, using fallback")
                return morphological_skeleton(binary_image)
        except AttributeError:
            logging.debug("OpenCV contrib not available, using fallback skeleton")
            return morphological_skeleton(binary_image)
    
    elif method == "zhang_suen":
        return zhang_suen_thinning(binary_image)
    
    elif method == "morphological":
        return morphological_skeleton(binary_image)
    
    else:
        logging.warning(f"Unknown thinning method '{method}', using morphological")
        return morphological_skeleton(binary_image)

def morphological_skeleton(binary_image: np.ndarray) -> np.ndarray:
    """
    Morphological skeleton using iterative erosion and opening.
    
    Args:
        binary_image: Input binary image
        
    Returns:
        Skeleton of the binary image
    """
    skeleton = np.zeros_like(binary_image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Work on a copy
    binary_copy = binary_image.copy()
    
    iteration = 0
    max_iterations = 100  # Prevent infinite loops
    
    while True:
        eroded = cv2.erode(binary_copy, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(binary_copy, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_copy = eroded.copy()
        
        iteration += 1
        if cv2.countNonZero(binary_copy) == 0 or iteration >= max_iterations:
            break
    
    return skeleton

def zhang_suen_thinning(binary_image: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning algorithm implementation.
    
    Args:
        binary_image: Input binary image (0 and 255)
        
    Returns:
        Thinned binary image
    """
    # Convert to 0 and 1
    img = (binary_image > 0).astype(np.uint8)
    h, w = img.shape
    
    # Create padded image
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = img
    
    iteration = 0
    max_iterations = 100
    
    while iteration < max_iterations:
        changed = False
        
        # Sub-iteration 1
        to_delete = []
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                if padded[i, j] == 1:
                    # Get 8-connected neighbors
                    p = [padded[i-1, j], padded[i-1, j+1], padded[i, j+1], 
                         padded[i+1, j+1], padded[i+1, j], padded[i+1, j-1],
                         padded[i, j-1], padded[i-1, j-1]]
                    
                    # Conditions for Zhang-Suen
                    if (zhang_suen_conditions(p, 1)):
                        to_delete.append((i, j))
                        changed = True
        
        # Delete marked pixels
        for i, j in to_delete:
            padded[i, j] = 0
        
        # Sub-iteration 2
        to_delete = []
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                if padded[i, j] == 1:
                    # Get 8-connected neighbors
                    p = [padded[i-1, j], padded[i-1, j+1], padded[i, j+1], 
                         padded[i+1, j+1], padded[i+1, j], padded[i+1, j-1],
                         padded[i, j-1], padded[i-1, j-1]]
                    
                    # Conditions for Zhang-Suen
                    if (zhang_suen_conditions(p, 2)):
                        to_delete.append((i, j))
                        changed = True
        
        # Delete marked pixels
        for i, j in to_delete:
            padded[i, j] = 0
        
        if not changed:
            break
        
        iteration += 1
    
    # Extract result
    result = padded[1:-1, 1:-1] * 255
    return result.astype(np.uint8)

def zhang_suen_conditions(neighbors: List[int], sub_iter: int) -> bool:
    """
    Check Zhang-Suen thinning conditions.
    
    Args:
        neighbors: List of 8-connected neighbors
        sub_iter: Sub-iteration number (1 or 2)
        
    Returns:
        True if pixel should be deleted
    """
    # Number of non-zero neighbors
    n = sum(neighbors)
    
    # Number of 0-1 transitions
    transitions = 0
    for i in range(8):
        if neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1:
            transitions += 1
    
    # Check basic conditions
    if not (2 <= n <= 6):
        return False
    if transitions != 1:
        return False
    
    # Check sub-iteration specific conditions
    if sub_iter == 1:
        # P2 * P4 * P6 = 0
        if neighbors[0] * neighbors[2] * neighbors[4] != 0:
            return False
        # P4 * P6 * P8 = 0
        if neighbors[2] * neighbors[4] * neighbors[6] != 0:
            return False
    else:  # sub_iter == 2
        # P2 * P4 * P8 = 0
        if neighbors[0] * neighbors[2] * neighbors[6] != 0:
            return False
        # P2 * P6 * P8 = 0
        if neighbors[0] * neighbors[4] * neighbors[6] != 0:
            return False
    
    return True

def advanced_opening(image: np.ndarray, 
                    kernel_size: Tuple[int, int] = (5, 5),
                    kernel_shape: str = "ellipse",
                    iterations: int = 1) -> np.ndarray:
    """
    Advanced morphological opening with various kernel shapes.
    
    Args:
        image: Input binary image
        kernel_size: Size of the structuring element
        kernel_shape: Shape of kernel ("ellipse", "rect", "cross")
        iterations: Number of iterations
        
    Returns:
        Opened image
    """
    if kernel_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    elif kernel_shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

def advanced_closing(image: np.ndarray, 
                    kernel_size: Tuple[int, int] = (5, 5),
                    kernel_shape: str = "ellipse",
                    iterations: int = 1) -> np.ndarray:
    """
    Advanced morphological closing with various kernel shapes.
    
    Args:
        image: Input binary image
        kernel_size: Size of the structuring element
        kernel_shape: Shape of kernel ("ellipse", "rect", "cross")
        iterations: Number of iterations
        
    Returns:
        Closed image
    """
    if kernel_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    elif kernel_shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

def top_hat_transform(image: np.ndarray, 
                     kernel_size: Tuple[int, int] = (15, 15),
                     transform_type: str = "white") -> np.ndarray:
    """
    Top-hat transformation for enhancing bright or dark features.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the structuring element
        transform_type: Type of transform ("white", "black")
        
    Returns:
        Transform result
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    if transform_type == "white":
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    elif transform_type == "black":
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    else:
        logging.warning(f"Unknown transform type '{transform_type}', using white")
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def morphological_gradient(image: np.ndarray, 
                          kernel_size: Tuple[int, int] = (3, 3),
                          gradient_type: str = "standard") -> np.ndarray:
    """
    Morphological gradient for edge detection.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the structuring element
        gradient_type: Type of gradient ("standard", "external", "internal")
        
    Returns:
        Gradient image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    if gradient_type == "standard":
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    elif gradient_type == "external":
        dilated = cv2.dilate(image, kernel)
        return cv2.subtract(dilated, image)
    elif gradient_type == "internal":
        eroded = cv2.erode(image, kernel)
        return cv2.subtract(image, eroded)
    else:
        logging.warning(f"Unknown gradient type '{gradient_type}', using standard")
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

def hit_or_miss_transform(image: np.ndarray, 
                         kernel_fg: np.ndarray,
                         kernel_bg: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Hit-or-miss transformation for pattern matching.
    
    Args:
        image: Input binary image
        kernel_fg: Foreground structuring element
        kernel_bg: Background structuring element (optional)
        
    Returns:
        Hit-or-miss result
    """
    if kernel_bg is None:
        # Create complement kernel
        kernel_bg = 1 - kernel_fg
    
    # Erosion with foreground kernel
    eroded_fg = cv2.erode(image, kernel_fg)
    
    # Erosion of complement with background kernel
    complement = cv2.bitwise_not(image)
    eroded_bg = cv2.erode(complement, kernel_bg)
    
    # Intersection
    result = cv2.bitwise_and(eroded_fg, eroded_bg)
    
    return result

def distance_transform_watershed(binary_image: np.ndarray, 
                                min_distance: int = 10) -> Tuple[np.ndarray, int]:
    """
    Watershed segmentation using distance transform.
    
    Args:
        binary_image: Input binary image
        min_distance: Minimum distance between peaks
        
    Returns:
        Tuple of (labeled_image, num_labels)
    """
    # Distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    
    # Find local maxima
    local_maxima = ndimage.maximum_filter(dist_transform, size=min_distance) == dist_transform
    local_maxima = local_maxima & (dist_transform > 0)
    
    # Label maxima as markers
    markers, num_markers = ndimage.label(local_maxima)
    
    # Watershed
    labels = cv2.watershed(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), markers)
    
    return labels, num_markers

def morphological_reconstruction(marker: np.ndarray, 
                               mask: np.ndarray,
                               method: str = "dilation") -> np.ndarray:
    """
    Morphological reconstruction by dilation or erosion.
    
    Args:
        marker: Marker image
        mask: Mask image
        method: Reconstruction method ("dilation", "erosion")
        
    Returns:
        Reconstructed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    if method == "dilation":
        reconstructed = marker.copy()
        while True:
            previous = reconstructed.copy()
            reconstructed = cv2.dilate(reconstructed, kernel)
            reconstructed = cv2.bitwise_and(reconstructed, mask)
            
            if np.array_equal(reconstructed, previous):
                break
    
    elif method == "erosion":
        reconstructed = marker.copy()
        while True:
            previous = reconstructed.copy()
            reconstructed = cv2.erode(reconstructed, kernel)
            reconstructed = cv2.bitwise_or(reconstructed, mask)
            
            if np.array_equal(reconstructed, previous):
                break
    
    else:
        logging.warning(f"Unknown method '{method}', using dilation")
        return morphological_reconstruction(marker, mask, "dilation")
    
    return reconstructed

def remove_small_objects(binary_image: np.ndarray, 
                        min_size: int = 50,
                        connectivity: int = 8) -> np.ndarray:
    """
    Remove small connected components from binary image.
    
    Args:
        binary_image: Input binary image
        min_size: Minimum size of objects to keep
        connectivity: Connectivity for component analysis (4 or 8)
        
    Returns:
        Filtered binary image
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=connectivity
    )
    
    # Create output image
    result = np.zeros_like(binary_image)
    
    # Keep components that meet size criteria
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            result[labels == i] = 255
    
    return result

def fill_holes(binary_image: np.ndarray, 
               max_hole_size: Optional[int] = None) -> np.ndarray:
    """
    Fill holes in binary objects.
    
    Args:
        binary_image: Input binary image
        max_hole_size: Maximum size of holes to fill (None for all holes)
        
    Returns:
        Image with filled holes
    """
    # Create a mask from the image border
    h, w = binary_image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Flood fill from border
    filled = binary_image.copy()
    cv2.floodFill(filled, mask, (0, 0), 255)
    
    # Invert the flood filled image
    filled_inv = cv2.bitwise_not(filled)
    
    # Combine with original
    result = cv2.bitwise_or(binary_image, filled_inv)
    
    # If max_hole_size is specified, only fill small holes
    if max_hole_size is not None:
        # Find holes
        holes = cv2.bitwise_and(filled_inv, cv2.bitwise_not(binary_image))
        
        # Remove large holes
        filtered_holes = remove_small_objects(holes, max_hole_size + 1, 8)
        large_holes = cv2.bitwise_and(holes, cv2.bitwise_not(filtered_holes))
        
        # Subtract large holes from result
        result = cv2.bitwise_and(result, cv2.bitwise_not(large_holes))
    
    return result

if __name__ == "__main__":
    """Test the morphological operations"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Create test binary image
    test_image = np.zeros((100, 100), dtype=np.uint8)
    
    # Add some shapes
    cv2.rectangle(test_image, (20, 20), (40, 60), 255, -1)
    cv2.circle(test_image, (70, 30), 15, (255,), -1)
    cv2.rectangle(test_image, (10, 70), (50, 90), 255, 2)  # Hollow rectangle
    
    # Add noise
    noise_points = np.random.randint(0, 100, (20, 2))
    for point in noise_points:
        test_image[point[1], point[0]] = 255
    
    print("Testing morphological operations...")
    
    # Test thinning
    print("Testing thinning algorithms...")
    skeleton1 = safe_thinning(test_image, "morphological")
    skeleton2 = safe_thinning(test_image, "zhang_suen")
    print(f"Morphological skeleton: {np.sum(skeleton1 > 0)} pixels")
    print(f"Zhang-Suen skeleton: {np.sum(skeleton2 > 0)} pixels")
    
    # Test opening and closing
    print("Testing opening and closing...")
    opened = advanced_opening(test_image, (5, 5), "ellipse")
    closed = advanced_closing(test_image, (5, 5), "ellipse")
    print(f"Opened image: {np.sum(opened > 0)} pixels")
    print(f"Closed image: {np.sum(closed > 0)} pixels")
    
    # Test top-hat transform
    print("Testing top-hat transform...")
    test_gray = cv2.GaussianBlur(test_image, (5, 5), 0)
    tophat = top_hat_transform(test_gray, (15, 15), "white")
    print(f"Top-hat result: {np.sum(tophat > 0)} pixels")
    
    # Test morphological gradient
    print("Testing morphological gradient...")
    gradient = morphological_gradient(test_gray, (3, 3), "standard")
    print(f"Gradient result: {np.sum(gradient > 0)} pixels")
    
    # Test small object removal
    print("Testing small object removal...")
    filtered = remove_small_objects(test_image, min_size=10)
    print(f"Filtered image: {np.sum(filtered > 0)} pixels")
    
    # Test hole filling
    print("Testing hole filling...")
    filled = fill_holes(test_image)
    print(f"Filled image: {np.sum(filled > 0)} pixels")
    
    # Test distance transform watershed
    print("Testing distance transform watershed...")
    labels, num_labels = distance_transform_watershed(test_image, min_distance=10)
    print(f"Watershed found {num_labels} regions")
    
    print("All morphological operations tests completed!")
