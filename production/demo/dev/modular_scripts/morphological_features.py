#!/usr/bin/env python3
"""
Morphological features extraction module for shape and structure analysis.
Uses morphological operations to detect and analyze image structures.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


def extract_morphological_features(gray: np.ndarray) -> Dict[str, float]:
    """
    Extract morphological features using various operations.
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        Dictionary of morphological features
    """
    features = {}
    
    # Multi-scale morphological operations
    for size in [3, 5, 7, 11]:
        # Create circular structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        
        # White top-hat: bright features smaller than kernel
        wth = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Black top-hat: dark features smaller than kernel
        bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Store statistics
        features[f'morph_wth_{size}_mean'] = float(np.mean(wth))
        features[f'morph_wth_{size}_max'] = float(np.max(wth))
        features[f'morph_wth_{size}_sum'] = float(np.sum(wth))
        features[f'morph_bth_{size}_mean'] = float(np.mean(bth))
        features[f'morph_bth_{size}_max'] = float(np.max(bth))
        features[f'morph_bth_{size}_sum'] = float(np.sum(bth))
    
    # Binary morphology analysis
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Standard 5x5 kernel
    kernel = np.ones((5, 5), np.uint8)
    
    # Basic operations
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    
    # Compute statistics
    features['morph_binary_area_ratio'] = float(np.sum(binary) / binary.size)
    features['morph_gradient_sum'] = float(np.sum(gradient))
    features['morph_erosion_ratio'] = float(np.sum(erosion) / (np.sum(binary) + 1e-10))
    features['morph_dilation_ratio'] = float(np.sum(dilation) / (np.sum(binary) + 1e-10))
    
    return features


def detect_morphological_defects(gray: np.ndarray,
                                kernel_sizes: List[int] = [3, 5, 7]) -> Dict[str, np.ndarray]:
    """
    Detect defects using morphological operations at multiple scales.
    
    Args:
        gray: Grayscale image (uint8)
        kernel_sizes: List of kernel sizes to use
        
    Returns:
        Dictionary containing defect maps for different types
    """
    defect_maps = {}
    
    for size in kernel_sizes:
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        
        # Top-hat for bright defects
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        defect_maps[f'bright_defects_{size}'] = tophat
        
        # Black-hat for dark defects  
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        defect_maps[f'dark_defects_{size}'] = blackhat
        
        # Combined defects
        combined = cv2.add(tophat, blackhat)
        defect_maps[f'combined_defects_{size}'] = combined
    
    return defect_maps


def extract_shape_complexity(gray: np.ndarray) -> Dict[str, float]:
    """
    Extract shape complexity features using morphological operations.
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        Dictionary of shape complexity features
    """
    features = {}
    
    # Binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Series of erosions to measure shape complexity
    erosion_levels = []
    current = binary.copy()
    
    for i in range(5):
        kernel = np.ones((3, 3), np.uint8)
        current = cv2.erode(current, kernel, iterations=1)
        erosion_levels.append(np.sum(current))
    
    # Shape persistence (how much survives erosion)
    if np.sum(binary) > 0:
        features['shape_persistence'] = float(erosion_levels[-1] / np.sum(binary))
    else:
        features['shape_persistence'] = 0.0
    
    # Erosion rate (how quickly shape erodes)
    erosion_diffs = np.diff([np.sum(binary)] + erosion_levels)
    features['shape_erosion_rate'] = float(np.mean(np.abs(erosion_diffs)))
    
    # Opening and closing to measure roughness
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    features['shape_roughness'] = float(np.sum(np.abs(binary - opening)) / (np.sum(binary) + 1e-10))
    features['shape_holes'] = float(np.sum(np.abs(closing - binary)) / (np.sum(binary) + 1e-10))
    
    return features


def extract_skeleton_features(gray: np.ndarray) -> Dict[str, float]:
    """
    Extract skeleton-based features from binary image.
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        Dictionary of skeleton features
    """
    features = {}
    
    # Convert to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological skeleton
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Iterative thinning
    temp = binary.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        subset = eroded - opening
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp = eroded.copy()
        
        if cv2.countNonZero(temp) == 0:
            break
    
    # Skeleton features
    features['skeleton_pixels'] = float(np.sum(skeleton > 0))
    features['skeleton_ratio'] = float(np.sum(skeleton > 0) / (np.sum(binary > 0) + 1e-10))
    
    # Find branch points (pixels with >2 neighbors)
    kernel = np.ones((3, 3), np.uint8)
    skeleton_dilated = cv2.dilate(skeleton, kernel, iterations=1)
    branches = cv2.bitwise_and(skeleton, skeleton_dilated)
    features['skeleton_branches'] = float(np.sum(branches > 0))
    
    return features


def apply_morphological_filter(gray: np.ndarray,
                             operation: str = 'opening',
                             kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological filter to image.
    
    Args:
        gray: Grayscale image (uint8)
        operation: Type of operation ('opening', 'closing', 'gradient', 'tophat', 'blackhat')
        kernel_size: Size of structuring element
        
    Returns:
        Filtered image
    """
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply operation
    operations = {
        'opening': cv2.MORPH_OPEN,
        'closing': cv2.MORPH_CLOSE,
        'gradient': cv2.MORPH_GRADIENT,
        'tophat': cv2.MORPH_TOPHAT,
        'blackhat': cv2.MORPH_BLACKHAT
    }
    
    if operation in operations:
        result = cv2.morphologyEx(gray, operations[operation], kernel)
    else:
        result = gray.copy()
    
    return result


def detect_connected_components(binary: np.ndarray,
                              min_area: int = 50) -> List[Dict]:
    """
    Detect and analyze connected components in binary image.
    
    Args:
        binary: Binary image (uint8)
        min_area: Minimum area for valid component
        
    Returns:
        List of component dictionaries with properties
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    
    components = []
    
    # Process each component (skip background at index 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if area >= min_area:
            # Extract component mask
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Calculate properties
            perimeter = cv2.arcLength(cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
            
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
            aspect_ratio = w / (h + 1e-10)
            extent = area / (w * h + 1e-10)
            
            components.append({
                'id': i,
                'bbox': (x, y, w, h),
                'area': int(area),
                'centroid': (float(centroids[i][0]), float(centroids[i][1])),
                'perimeter': float(perimeter),
                'circularity': float(circularity),
                'aspect_ratio': float(aspect_ratio),
                'extent': float(extent)
            })
    
    return components


def main():
    """Standalone test function."""
    print("Morphological Features Module - Standalone Test")
    print("-" * 40)
    
    # Create test image with various shapes
    test_image = np.zeros((300, 300), dtype=np.uint8)
    
    # Add circle
    cv2.circle(test_image, (75, 75), 30, 255, -1)
    
    # Add rectangle
    cv2.rectangle(test_image, (150, 50), (250, 100), 255, -1)
    
    # Add ellipse
    cv2.ellipse(test_image, (150, 200), (40, 20), 45, 0, 360, 255, -1)
    
    # Add noise
    noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
    test_image = cv2.add(test_image, noise)
    
    print("Extracting morphological features...")
    features = extract_morphological_features(test_image)
    
    print("\nMorphological Features (sample):")
    for key in sorted(features.keys())[:10]:
        print(f"  {key}: {features[key]:.3f}")
    
    print("\nExtracting shape complexity...")
    complexity = extract_shape_complexity(test_image)
    for key, value in complexity.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nExtracting skeleton features...")
    skeleton = extract_skeleton_features(test_image)
    for key, value in skeleton.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nDetecting morphological defects...")
    defect_maps = detect_morphological_defects(test_image)
    for name, defect_map in defect_maps.items():
        defect_count = np.sum(defect_map > 30)  # Threshold
        print(f"  {name}: {defect_count} pixels")
    
    print("\nDetecting connected components...")
    _, binary = cv2.threshold(test_image, 127, 255, cv2.THRESH_BINARY)
    components = detect_connected_components(binary)
    print(f"Found {len(components)} components:")
    for comp in components[:3]:
        print(f"  Component {comp['id']}: area={comp['area']}, "
              f"circularity={comp['circularity']:.2f}")
    
    # Save some results
    cv2.imwrite("morph_original_test.png", test_image)
    cv2.imwrite("morph_gradient_test.png", 
                apply_morphological_filter(test_image, 'gradient'))
    cv2.imwrite("morph_tophat_test.png",
                apply_morphological_filter(test_image, 'tophat'))
    print("\nTest images saved.")


if __name__ == "__main__":
    main()
