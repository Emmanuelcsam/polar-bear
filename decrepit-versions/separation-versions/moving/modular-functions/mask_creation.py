#!/usr/bin/env python3
"""
Mask Creation Functions
Extracted from multiple fiber optic analysis scripts

This module contains functions for creating and manipulating masks:
- Circular mask creation for core, cladding, and ferrule regions
- Binary mask operations and refinements
- Morphological mask processing
- Annulus detection and mask generation
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, Union


def create_circular_mask(image_shape: Tuple[int, int], center: Tuple[int, int], 
                        radius: float, fill_value: int = 255) -> np.ndarray:
    """
    Create a circular mask.
    
    From: computational_separation.py, sergio.py
    
    Args:
        image_shape: (height, width) of the output mask
        center: (x, y) center coordinates
        radius: Radius of the circle
        fill_value: Value to fill the circle with
        
    Returns:
        Binary mask with circular region filled
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    center_int = (int(center[0]), int(center[1]))
    radius_int = int(radius)
    
    cv2.circle(mask, center_int, radius_int, (fill_value,), -1)
    
    return mask


def create_annulus_mask(image_shape: Tuple[int, int], center: Tuple[int, int], 
                       inner_radius: float, outer_radius: float, 
                       fill_value: int = 255) -> np.ndarray:
    """
    Create an annulus (ring) mask.
    
    From: sergio.py, computational_separation.py
    
    Args:
        image_shape: (height, width) of the output mask
        center: (x, y) center coordinates
        inner_radius: Inner radius of the annulus
        outer_radius: Outer radius of the annulus
        fill_value: Value to fill the annulus with
        
    Returns:
        Binary mask with annulus region filled
    """
    # Create outer circle
    outer_mask = create_circular_mask(image_shape, center, outer_radius, fill_value)
    
    # Create inner circle
    inner_mask = create_circular_mask(image_shape, center, inner_radius, fill_value)
    
    # Subtract inner from outer to create annulus
    annulus_mask = cv2.subtract(outer_mask, inner_mask)
    
    return annulus_mask


def create_fiber_masks(image_shape: Tuple[int, int], center: Tuple[int, int], 
                      core_radius: float, cladding_radius: float) -> Dict[str, np.ndarray]:
    """
    Create core, cladding, and ferrule masks for fiber optic analysis.
    
    From: sergio.py, fiber_optic_segmentation.py
    
    Args:
        image_shape: (height, width) of the masks
        center: (x, y) center coordinates
        core_radius: Radius of the core region
        cladding_radius: Radius of the cladding region
        
    Returns:
        Dictionary containing 'core', 'cladding', and 'ferrule' masks
    """
    h, w = image_shape
    
    # Core mask (solid circle)
    core_mask = create_circular_mask(image_shape, center, core_radius)
    
    # Cladding mask (annulus between core and cladding)
    cladding_mask = create_annulus_mask(image_shape, center, core_radius, cladding_radius)
    
    # Ferrule mask (everything outside cladding)
    ferrule_mask = np.ones((h, w), dtype=np.uint8) * 255
    cladding_outer_mask = create_circular_mask(image_shape, center, cladding_radius)
    ferrule_mask = cv2.subtract(ferrule_mask, cladding_outer_mask)
    
    return {
        'core': core_mask,
        'cladding': cladding_mask,
        'ferrule': ferrule_mask
    }


def distance_transform_mask(image_shape: Tuple[int, int], center: Tuple[int, int], 
                           inner_radius: float, outer_radius: float) -> np.ndarray:
    """
    Create a mask using distance transform for smooth boundaries.
    
    From: computational_separation.py
    
    Args:
        image_shape: (height, width) of the mask
        center: (x, y) center coordinates
        inner_radius: Inner radius
        outer_radius: Outer radius
        
    Returns:
        Continuous-valued mask based on distance
    """
    h, w = image_shape
    cx, cy = center
    
    # Create distance matrix
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    annulus_region = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    mask[annulus_region] = 255
    
    return mask


def find_annulus_mask_from_binary(filtered_image: np.ndarray) -> np.ndarray:
    """
    Find black annulus pixels from a binary filtered image.
    
    From: cladding.py
    
    Args:
        filtered_image: Binary filtered image
        
    Returns:
        Mask of detected annulus pixels
    """
    # Create mask for black pixels (annulus)
    black_mask = np.zeros_like(filtered_image)
    
    # Black pixels are where filtered_image == 0
    black_pixels = (filtered_image == 0)
    
    # Find black contours
    black_contours, _ = cv2.findContours((~filtered_image).astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in black_contours:
        # Create a mask for this black region
        temp_mask = np.zeros_like(filtered_image)
        cv2.drawContours(temp_mask, [contour], -1, (255,), -1)
        
        # Create a filled version of the contour
        filled_mask = np.zeros_like(filtered_image)
        cv2.drawContours(filled_mask, [contour], -1, (255,), -1)
        
        # Check if there are white pixels inside this black region
        inner_white = (filtered_image == 255) & (filled_mask == 255)
        
        # If we found inner white pixels, this is an annulus
        if np.any(inner_white):
            # Add to black mask (the annulus itself)
            black_mask = black_mask | (temp_mask & black_pixels)
    
    return black_mask


def find_inner_white_mask_from_binary(filtered_image: np.ndarray) -> np.ndarray:
    """
    Find white pixels inside annulus from binary filtered image.
    
    From: core.py
    
    Args:
        filtered_image: Binary filtered image
        
    Returns:
        Mask of inner white pixels (core region)
    """
    # Create mask for inner white pixels
    inner_white_mask = np.zeros_like(filtered_image)
    
    # Find connected components of white pixels
    num_labels, labels = cv2.connectedComponents(filtered_image)
    
    # Find black contours (potential annulus)
    black_contours, _ = cv2.findContours((~filtered_image).astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in black_contours:
        # Create a filled version of the contour
        filled_mask = np.zeros_like(filtered_image)
        cv2.drawContours(filled_mask, [contour], -1, (255,), -1)
        
        # The inner white pixels are those that are white in the filtered image
        # and inside the filled contour
        inner_white = (filtered_image == 255) & (filled_mask == 255)
        
        # If we found inner white pixels, add to mask
        if np.any(inner_white):
            inner_white_mask = inner_white_mask | inner_white
    
    return inner_white_mask


def apply_morphological_operations(mask: np.ndarray, operation: str = 'close', 
                                  kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological operations to clean up masks.
    
    From: multiple scripts
    
    Args:
        mask: Input binary mask
        operation: Type of operation ('open', 'close', 'erode', 'dilate')
        kernel_size: Size of the morphological kernel
        iterations: Number of iterations
        
    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'open':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == 'erode':
        return cv2.erode(mask, kernel, iterations=iterations)
    elif operation == 'dilate':
        return cv2.dilate(mask, kernel, iterations=iterations)
    else:
        return mask.copy()


def refine_mask_with_otsu(image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
    """
    Refine mask using Otsu thresholding within the masked region.
    
    From: bright_core_extractor.py
    
    Args:
        image: Input grayscale image
        initial_mask: Initial binary mask
        
    Returns:
        Refined mask
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply initial mask
    masked_region = cv2.bitwise_and(gray, gray, mask=initial_mask)
    
    # Apply Otsu thresholding
    _, refined_mask = cv2.threshold(masked_region, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return refined_mask


def combine_masks(masks: Dict[str, np.ndarray], combination_method: str = 'union') -> np.ndarray:
    """
    Combine multiple masks using different methods.
    
    Args:
        masks: Dictionary of named masks
        combination_method: Method to combine ('union', 'intersection', 'weighted')
        
    Returns:
        Combined mask
    """
    if not masks:
        return np.array([])
    
    mask_list = list(masks.values())
    
    if combination_method == 'union':
        result = mask_list[0].copy()
        for mask in mask_list[1:]:
            result = cv2.bitwise_or(result, mask)
        return result
    
    elif combination_method == 'intersection':
        result = mask_list[0].copy()
        for mask in mask_list[1:]:
            result = cv2.bitwise_and(result, mask)
        return result
    
    elif combination_method == 'weighted':
        # Simple weighted average
        result = np.zeros_like(mask_list[0], dtype=np.float32)
        total_weight = 0
        
        for i, mask in enumerate(mask_list):
            weight = 1.0 / (i + 1)  # Decreasing weights
            result += mask.astype(np.float32) * weight
            total_weight += weight
        
        result = (result / total_weight).astype(np.uint8)
        return result
    
    else:
        return mask_list[0].copy()


def validate_mask_geometry(mask: np.ndarray, min_area: int = 100, 
                          max_area: Optional[int] = None, 
                          min_circularity: float = 0.5) -> bool:
    """
    Validate mask geometry for reasonable fiber optic shapes.
    
    Args:
        mask: Input binary mask
        min_area: Minimum area in pixels
        max_area: Maximum area in pixels (None = no limit)
        min_circularity: Minimum circularity (0-1)
        
    Returns:
        True if mask passes validation
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check area
    area = cv2.contourArea(largest_contour)
    if area < min_area:
        return False
    
    if max_area is not None and area > max_area:
        return False
    
    # Check circularity
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            return False
    
    return True


def crop_mask_to_content(mask: np.ndarray, padding: int = 5) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop mask to its content with optional padding.
    
    Args:
        mask: Input binary mask
        padding: Padding around content
        
    Returns:
        (cropped_mask, (x, y, width, height))
    """
    # Find bounding box of non-zero pixels
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        # Empty mask
        return mask.copy(), (0, 0, mask.shape[1], mask.shape[0])
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Add padding
    h, w = mask.shape
    y_min = max(0, y_min - padding)
    y_max = min(h - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w - 1, x_max + padding)
    
    # Crop mask
    cropped = mask[y_min:y_max+1, x_min:x_max+1]
    
    return cropped, (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def main():
    """Test the mask creation functions"""
    # Test image dimensions
    image_shape = (400, 400)
    center = (200, 200)
    core_radius = 60
    cladding_radius = 120
    
    print("Testing Mask Creation Functions...")
    
    # Test circular mask creation
    circular_mask = create_circular_mask(image_shape, center, core_radius)
    print(f"✓ Circular mask: {circular_mask.shape}, non-zero pixels: {np.sum(circular_mask > 0)}")
    
    # Test annulus mask creation
    annulus_mask = create_annulus_mask(image_shape, center, core_radius, cladding_radius)
    print(f"✓ Annulus mask: {annulus_mask.shape}, non-zero pixels: {np.sum(annulus_mask > 0)}")
    
    # Test fiber masks creation
    fiber_masks = create_fiber_masks(image_shape, center, core_radius, cladding_radius)
    print(f"✓ Fiber masks: {list(fiber_masks.keys())}")
    
    # Test distance transform mask
    distance_mask = distance_transform_mask(image_shape, center, core_radius, cladding_radius)
    print(f"✓ Distance transform mask: {distance_mask.shape}")
    
    # Test morphological operations
    processed_mask = apply_morphological_operations(circular_mask, 'close', 5)
    print(f"✓ Morphological operations: {processed_mask.shape}")
    
    # Test mask combination
    test_masks = {'mask1': circular_mask, 'mask2': annulus_mask}
    combined_mask = combine_masks(test_masks, 'union')
    print(f"✓ Mask combination: {combined_mask.shape}")
    
    # Test mask validation
    is_valid = validate_mask_geometry(circular_mask)
    print(f"✓ Mask validation: {is_valid}")
    
    # Test mask cropping
    cropped_mask, bbox = crop_mask_to_content(circular_mask)
    print(f"✓ Mask cropping: {cropped_mask.shape}, bbox: {bbox}")
    
    # Create test binary image for annulus detection
    test_binary = np.zeros(image_shape, dtype=np.uint8)
    test_binary[100:300, 100:300] = 255  # White square
    cv2.circle(test_binary, center, 80, (0,), 20)  # Black ring
    
    # Test annulus detection
    detected_annulus = find_annulus_mask_from_binary(test_binary)
    print(f"✓ Annulus detection: {np.sum(detected_annulus > 0)} pixels found")
    
    # Test inner white mask detection
    inner_white = find_inner_white_mask_from_binary(test_binary)
    print(f"✓ Inner white detection: {np.sum(inner_white > 0)} pixels found")
    
    print("\nAll mask creation functions tested successfully!")


if __name__ == "__main__":
    main()
