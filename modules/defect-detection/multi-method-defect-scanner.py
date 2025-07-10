#!/usr/bin/env python3
"""
Multi-Algorithm Defect Detection Module
======================================
Advanced defect detection using multiple algorithms with intelligent fusion.
Includes DO2MR, LEI, and matrix variance detection methods.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from scipy import ndimage

# Try to import C++ accelerator
try:
    import accelerator
    CPP_ACCELERATOR_AVAILABLE = True
    logging.info("C++ accelerator available for DO2MR detection")
except ImportError:
    CPP_ACCELERATOR_AVAILABLE = False
    logging.debug("C++ accelerator not available, using Python implementation")

def do2mr_detection(image: np.ndarray, 
                   zone_mask: np.ndarray, 
                   zone_name: str,
                   kernel_size: int = 5,
                   gamma: float = 1.5,
                   multi_scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced DO2MR (Difference of Morphological Opening and Closing) detection
    with adaptive parameters and multi-scale detection.
    
    Args:
        image: Input grayscale image
        zone_mask: Binary mask for the zone
        zone_name: Name of the zone ("Core", "Cladding", etc.)
        kernel_size: Base kernel size for morphological operations
        gamma: Sensitivity parameter
        multi_scale: Whether to use multi-scale detection
        
    Returns:
        Tuple of (defect_mask, confidence_map)
    """
    # Try C++ accelerator first if available
    if CPP_ACCELERATOR_AVAILABLE and not multi_scale:
        try:
            # Apply adaptive sensitivity for core zone
            if zone_name == "Core":
                gamma = gamma * 0.8
            
            result = accelerator.do2mr_detection(image, kernel_size, gamma)
            confidence = result.astype(np.float32) / 255.0
            return result, confidence
        except Exception as e:
            logging.debug(f"C++ accelerator failed, falling back to Python: {e}")
    
    # Enhanced Python implementation with multi-scale
    if multi_scale:
        kernel_sizes = [3, 5, 7, 9] if zone_name == "Core" else [5, 7, 11]
    else:
        kernel_sizes = [kernel_size]
    
    combined_result = np.zeros_like(image, dtype=np.float32)
    
    # Apply zone mask once
    masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
    
    # Pre-filter to reduce noise
    denoised = cv2.bilateralFilter(masked_image, 5, 50, 50)
    
    for k_size in kernel_sizes:
        # Apply adaptive sensitivity for core zone
        current_gamma = gamma * 0.8 if zone_name == "Core" else gamma
        
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        # Max and min filtering (opening and closing)
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # Calculate residual
        residual = cv2.subtract(closed, opened)
        
        # Apply guided filter for edge-preserving smoothing if available
        try:
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
                residual_filtered = cv2.ximgproc.guidedFilter(
                    guide=denoised, src=residual, radius=3, eps=10
                )
            else:
                residual_filtered = cv2.medianBlur(residual, 3)
        except:
            residual_filtered = cv2.medianBlur(residual, 3)
        
        # Get pixels within zone for statistics
        zone_pixels = residual_filtered[zone_mask > 0]
        if len(zone_pixels) < 100:
            continue
        
        # Use robust statistics (median and MAD)
        median_val = np.median(zone_pixels)
        mad = np.median(np.abs(zone_pixels - median_val))
        std_robust = 1.4826 * mad  # Conversion factor for normal distribution
        
        # Calculate adaptive threshold
        threshold = float(median_val + current_gamma * std_robust)
        
        # Apply threshold
        _, result_for_scale = cv2.threshold(residual_filtered, threshold, 255, cv2.THRESH_BINARY)
        result_for_scale = cv2.bitwise_and(result_for_scale, zone_mask)
        
        # Weight by kernel size (smaller kernels get higher weight for small defects)
        weight = 1.0 / (1 + 0.5 * np.log(k_size))
        combined_result += result_for_scale.astype(np.float32) * weight
    
    # Normalize combined result
    if len(kernel_sizes) > 0:
        total_weight = sum(1.0 / (1 + 0.5 * np.log(k)) for k in kernel_sizes)
        combined_result = combined_result / total_weight
    
    # Final thresholding
    _, defect_mask = cv2.threshold(combined_result, 127, 255, cv2.THRESH_BINARY)
    defect_mask = defect_mask.astype(np.uint8)
    
    # Apply zone mask to final result
    defect_mask = cv2.bitwise_and(defect_mask, zone_mask)
    
    # Morphological cleanup
    min_defect_size = 3 if zone_name == "Core" else 5
    
    # Opening to remove noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # Remove very small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_defect_size:
            defect_mask[labels == i] = 0
    
    # Create confidence map normalized to [0, 1]
    confidence_map = combined_result / 255.0
    confidence_map = np.clip(confidence_map, 0, 1)
    
    # Mask confidence map to zone
    zone_mask_float = zone_mask.astype(np.float32) / 255.0
    confidence_map = confidence_map * zone_mask_float
    
    return defect_mask, confidence_map

def lei_scratch_detection(image: np.ndarray, 
                         zone_mask: np.ndarray,
                         kernel_lengths: List[int] = [7, 11, 15, 21, 31],
                         angle_step: int = 10,
                         multi_scale: bool = True) -> np.ndarray:
    """
    Enhanced LEI (Line Enhancement and Identification) with multi-scale 
    and directional filtering for scratch detection.
    
    Args:
        image: Input grayscale image
        zone_mask: Binary mask for the zone
        kernel_lengths: List of kernel lengths for different scales
        angle_step: Angular step in degrees for directional filtering
        multi_scale: Whether to use multi-scale detection
        
    Returns:
        Binary mask of detected scratches
    """
    # Apply zone mask
    masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
    
    # Multi-scale preprocessing
    if multi_scale:
        scales = [1.0, 0.75, 1.25]
    else:
        scales = [1.0]
    
    all_scratch_maps = []
    
    for scale in scales:
        # Resize image
        if scale != 1.0:
            scaled_h = int(image.shape[0] * scale)
            scaled_w = int(image.shape[1] * scale)
            scaled_image = cv2.resize(masked_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            scaled_mask = cv2.resize(zone_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        else:
            scaled_image = masked_image
            scaled_mask = zone_mask
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled_image)
        
        # Apply top-hat transform to enhance linear structures
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_tophat)
        
        # Initialize scratch map for this scale
        scratch_map = np.zeros_like(enhanced, dtype=np.float32)
        
        # Enhanced directional filtering
        for angle in range(0, 180, angle_step):
            angle_rad = np.deg2rad(angle)
            
            for kernel_length in kernel_lengths:
                # Create enhanced linear kernel with Gaussian profile
                kernel_size = kernel_length + 4  # Padding for rotation
                kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
                center = kernel_size // 2
                
                # Create Gaussian-weighted line
                sigma = kernel_length / 6.0
                for i in range(kernel_length):
                    pos = i - kernel_length // 2
                    weight = np.exp(-pos**2 / (2 * sigma**2))
                    x = center
                    y = center + pos
                    if 0 <= y < kernel_size:
                        kernel[y, x] = weight
                
                # Rotate kernel
                M = cv2.getRotationMatrix2D((center, center), angle, 1)
                rotated_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
                
                # Normalize kernel
                kernel_sum = np.sum(rotated_kernel)
                if kernel_sum > 0:
                    rotated_kernel = rotated_kernel / kernel_sum
                
                # Apply directional filter
                response = cv2.filter2D(tophat, cv2.CV_32F, rotated_kernel)
                
                # Update scratch map with maximum response
                scratch_map = np.maximum(scratch_map, response)
        
        # Resize back to original size if needed
        if scale != 1.0:
            scratch_map = cv2.resize(scratch_map, (image.shape[1], image.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        
        all_scratch_maps.append(scratch_map)
    
    # Combine multi-scale results
    combined_scratch_map = np.mean(all_scratch_maps, axis=0)
    
    # Normalize
    combined_scratch_map_norm = np.zeros_like(combined_scratch_map, dtype=np.uint8)
    cv2.normalize(combined_scratch_map, combined_scratch_map_norm, 0, 255, 
                  cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    combined_scratch_map = combined_scratch_map_norm
    
    # Advanced thresholding with Otsu's method
    _, otsu_thresh = cv2.threshold(combined_scratch_map, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive threshold for local variations
    adaptive_thresh = cv2.adaptiveThreshold(combined_scratch_map, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 15, -2)
    
    # Combine both thresholding methods
    combined_binary = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
    
    # Apply zone mask
    result = cv2.bitwise_and(combined_binary, zone_mask)
    
    # Morphological operations to connect scratch fragments
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_connect)
    
    # Remove small non-linear components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        
        # Keep only linear structures
        if area < 10 or aspect_ratio < 2.5:
            result[labels == i] = 0
    
    return result

def matrix_variance_detection(image: np.ndarray, 
                            zone_mask: np.ndarray,
                            variance_threshold: float = 15.0,
                            local_window_size: int = 3) -> np.ndarray:
    """
    Divides image into 9 segments and detects anomalies based on local pixel variance.
    
    Args:
        image: Input grayscale image
        zone_mask: Binary mask for the zone
        variance_threshold: Threshold for variance-based detection
        local_window_size: Size of local analysis window
        
    Returns:
        Binary mask of detected anomalies
    """
    h, w = image.shape[:2]
    result_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define 9 segments (3x3 grid)
    segment_h = h // 3
    segment_w = w // 3
    
    for row in range(3):
        for col in range(3):
            # Calculate segment boundaries
            y_start = row * segment_h
            y_end = (row + 1) * segment_h if row < 2 else h
            x_start = col * segment_w
            x_end = (col + 1) * segment_w if col < 2 else w
            
            # Extract segment
            segment = image[y_start:y_end, x_start:x_end]
            segment_zone_mask = zone_mask[y_start:y_end, x_start:x_end]
            
            # Only process pixels within the zone
            if np.sum(segment_zone_mask) == 0:
                continue
            
            # Analyze each pixel in the segment
            seg_h, seg_w = segment.shape
            half_window = local_window_size // 2
            
            for y in range(half_window, seg_h - half_window):
                for x in range(half_window, seg_w - half_window):
                    if segment_zone_mask[y, x] == 0:
                        continue
                    
                    # Get local neighborhood
                    local_region = segment[y-half_window:y+half_window+1,
                                         x-half_window:x+half_window+1]
                    
                    # Calculate local statistics
                    center_value = float(segment[y, x])
                    local_mean = np.mean(local_region)
                    local_std = np.std(local_region)
                    
                    # Check for significant variance
                    if local_std > 0:
                        # Calculate how many standard deviations the center pixel is from local mean
                        z_score = abs(center_value - local_mean) / local_std
                        
                        # Also check absolute difference
                        abs_diff = abs(center_value - local_mean)
                        
                        # Mark as anomaly if high variance
                        if z_score > 2.0 or abs_diff > variance_threshold:
                            # Convert back to full image coordinates
                            full_y = y_start + y
                            full_x = x_start + x
                            result_mask[full_y, full_x] = 255
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)
    
    return result_mask

def validate_defects(defect_mask: np.ndarray, 
                    original_image: np.ndarray, 
                    zone_mask: np.ndarray,
                    zone_name: str = "",
                    min_contrast: float = 10) -> np.ndarray:
    """
    Enhanced defect validation with zone-specific rules and texture analysis.
    
    Args:
        defect_mask: Binary mask of detected defects
        original_image: Original grayscale image
        zone_mask: Binary mask for the zone
        zone_name: Name of the zone for zone-specific validation
        min_contrast: Minimum contrast threshold
        
    Returns:
        Validated defect mask
    """
    validated_mask = np.zeros_like(defect_mask)
    
    # Zone-specific validation parameters
    zone_params = {
        "Core": {"min_contrast": 15, "min_area": 3, "texture_threshold": 0.8},
        "Cladding": {"min_contrast": 10, "min_area": 5, "texture_threshold": 0.6},
        "default": {"min_contrast": 10, "min_area": 5, "texture_threshold": 0.5}
    }
    
    params = zone_params.get(zone_name, zone_params["default"])
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    
    for i in range(1, num_labels):
        # Get component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Skip very small components
        if area < params["min_area"]:
            continue
            
        # Calculate local contrast
        defect_pixels = original_image[component_mask > 0]
        if len(defect_pixels) == 0:
            continue
            
        # Enhanced surrounding region analysis
        kernel_size = max(5, int(np.sqrt(area) / 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(component_mask, kernel)
        surrounding_mask = cv2.bitwise_and(dilated - component_mask, zone_mask)
        surrounding_pixels = original_image[surrounding_mask > 0]
        
        if len(surrounding_pixels) < 10:
            continue
            
        # Calculate validation metrics
        defect_mean = np.mean(defect_pixels)
        surrounding_mean = np.mean(surrounding_pixels)
        contrast = abs(defect_mean - surrounding_mean)
        
        # Statistical significance test
        if len(defect_pixels) > 5 and len(surrounding_pixels) > 5:
            defect_std = np.std(defect_pixels)
            surrounding_std = np.std(surrounding_pixels)
            pooled_std = np.sqrt((defect_std**2 + surrounding_std**2) / 2)
            t_statistic = abs(defect_mean - surrounding_mean) / (pooled_std + 1e-6)
            is_significant = t_statistic > 2.0  # Approximately 95% confidence
        else:
            is_significant = contrast > params["min_contrast"]
        
        # Validate based on zone-specific criteria
        if zone_name == "Core":
            # Stricter validation for core zone
            is_valid = contrast >= params["min_contrast"] * 1.5 and is_significant
        else:
            # Regular validation for other zones
            is_valid = contrast >= params["min_contrast"] and is_significant
        
        if is_valid:
            validated_mask = cv2.bitwise_or(validated_mask, component_mask)
    
    return validated_mask

def detect_defects_multi_algorithm(image: np.ndarray,
                                  zone_mask: np.ndarray,
                                  zone_name: str,
                                  algorithms: List[str] = ["do2mr", "lei"],
                                  algorithm_weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced defect detection using multiple algorithms with intelligent fusion.
    
    Args:
        image: Input grayscale image
        zone_mask: Binary mask for the zone
        zone_name: Name of the zone
        algorithms: List of algorithms to use
        algorithm_weights: Weights for combining algorithm results
        
    Returns:
        Tuple of (combined_defect_mask, confidence_map)
    """
    if image is None or zone_mask is None:
        return np.zeros_like(image), np.zeros_like(image, dtype=np.float32)
    
    if np.sum(zone_mask) == 0:
        return np.zeros_like(image), np.zeros_like(image, dtype=np.float32)
    
    # Default weights
    if algorithm_weights is None:
        algorithm_weights = {
            "do2mr": 1.0,
            "lei": 1.0,
            "matrix_variance": 0.8
        }
    
    # Store individual algorithm results
    algorithm_results = {}
    algorithm_confidences = {}
    
    # Apply pre-filtering to reduce noise
    preprocessed = cv2.bilateralFilter(image, 5, 50, 50)
    
    # Run algorithms
    if "do2mr" in algorithms:
        mask, conf = do2mr_detection(preprocessed, zone_mask, zone_name)
        algorithm_results["do2mr"] = mask
        algorithm_confidences["do2mr"] = conf
    
    if "lei" in algorithms:
        scratch_mask = lei_scratch_detection(preprocessed, zone_mask)
        algorithm_results["lei"] = scratch_mask
        algorithm_confidences["lei"] = scratch_mask.astype(np.float32) / 255.0
    
    if "matrix_variance" in algorithms:
        matrix_mask = matrix_variance_detection(preprocessed, zone_mask)
        algorithm_results["matrix_variance"] = matrix_mask
        algorithm_confidences["matrix_variance"] = matrix_mask.astype(np.float32) / 255.0
    
    # Intelligent fusion based on consensus
    num_algorithms = len(algorithm_results)
    if num_algorithms == 0:
        return np.zeros_like(image), np.zeros_like(image, dtype=np.float32)
    
    # Create consensus map
    consensus_map = np.zeros_like(image, dtype=np.float32)
    total_weight = 0
    
    for algo_name, result in algorithm_results.items():
        weight = algorithm_weights.get(algo_name, 1.0)
        consensus_map += (result > 0).astype(np.float32) * weight
        total_weight += weight
    
    # Normalize consensus
    if total_weight > 0:
        consensus_map = consensus_map / total_weight
    
    # Dynamic thresholding based on zone
    consensus_threshold = 0.6 if zone_name == "Core" else 0.4
    
    # Apply consensus threshold
    consensus_mask = (consensus_map >= consensus_threshold).astype(np.uint8) * 255
    
    # Post-processing
    kernel_median = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    consensus_mask = cv2.medianBlur(consensus_mask, 3)
    
    # Fill small holes in defects
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    consensus_mask = cv2.morphologyEx(consensus_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Enhanced validation
    validated_mask = validate_defects(consensus_mask, image, zone_mask, zone_name)
    
    # Create final confidence map
    final_confidence = np.zeros_like(image, dtype=np.float32)
    for algo_name, conf in algorithm_confidences.items():
        weight = algorithm_weights.get(algo_name, 1.0)
        final_confidence += conf * weight
    
    if total_weight > 0:
        final_confidence = final_confidence / total_weight
    
    # Apply validation mask to confidence
    final_confidence = final_confidence * (validated_mask > 0).astype(np.float32)
    
    return validated_mask, final_confidence

if __name__ == "__main__":
    """Test the defect detection functions"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Create a test image with defects
    test_image = np.random.randint(100, 150, (200, 200), dtype=np.uint8)
    
    # Add some defects
    cv2.rectangle(test_image, (50, 50), (55, 60), 80, -1)  # Small pit
    cv2.rectangle(test_image, (100, 30), (140, 33), 70, -1)  # Scratch
    cv2.circle(test_image, (150, 150), 3, (60,), -1)  # Small defect
    
    # Create zone mask
    zone_mask = np.zeros_like(test_image)
    cv2.circle(zone_mask, (100, 100), 80, (255,), -1)
    
    print("Testing defect detection algorithms...")
    
    # Test DO2MR
    print("Testing DO2MR detection...")
    do2mr_mask, do2mr_conf = do2mr_detection(test_image, zone_mask, "Core")
    print(f"DO2MR detected {np.sum(do2mr_mask > 0)} defect pixels")
    
    # Test LEI
    print("Testing LEI scratch detection...")
    lei_mask = lei_scratch_detection(test_image, zone_mask)
    print(f"LEI detected {np.sum(lei_mask > 0)} scratch pixels")
    
    # Test matrix variance
    print("Testing matrix variance detection...")
    matrix_mask = matrix_variance_detection(test_image, zone_mask)
    print(f"Matrix variance detected {np.sum(matrix_mask > 0)} anomaly pixels")
    
    # Test multi-algorithm detection
    print("Testing multi-algorithm detection...")
    combined_mask, combined_conf = detect_defects_multi_algorithm(
        test_image, zone_mask, "Core", ["do2mr", "lei", "matrix_variance"]
    )
    print(f"Combined detection found {np.sum(combined_mask > 0)} defect pixels")
    
    # Test validation
    print("Testing defect validation...")
    validated = validate_defects(combined_mask, test_image, zone_mask, "Core")
    print(f"After validation: {np.sum(validated > 0)} valid defect pixels")
    
    print("All defect detection tests completed!")
