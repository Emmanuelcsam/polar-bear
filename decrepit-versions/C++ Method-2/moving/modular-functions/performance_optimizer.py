#!/usr/bin/env python3
"""
Modular Performance Optimization Functions
==========================================
Standalone performance optimization functions for image processing
and computational efficiency in fiber inspection applications.
"""

import cv2
import numpy as np
import time
from functools import wraps
from typing import Tuple, Optional, Dict, Any, Callable, List
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def performance_timer(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

def memory_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage during function execution.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with memory monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before starting
        gc.collect()
        result = func(*args, **kwargs)
        # Force garbage collection after completion
        gc.collect()
        return result
    return wrapper

def resize_for_processing(
    image: np.ndarray, 
    max_dimension: int = 1024,
    maintain_aspect_ratio: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Resize image for optimal processing while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_dimension: Maximum allowed dimension
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_dimension:
        return image, 1.0
    
    if maintain_aspect_ratio:
        scale_factor = max_dimension / max(height, width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    else:
        scale_factor = max_dimension / max(height, width)
        new_width = new_height = max_dimension
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale_factor

def scale_coordinates(
    coordinates: np.ndarray, 
    scale_factor: float, 
    inverse: bool = False
) -> np.ndarray:
    """
    Scale coordinates based on image resize factor.
    
    Args:
        coordinates: Array of coordinates to scale
        scale_factor: Scale factor from resize operation
        inverse: If True, scale from small to large (inverse scaling)
        
    Returns:
        Scaled coordinates
    """
    if inverse:
        scale_factor = 1.0 / scale_factor
    
    return coordinates * scale_factor

def optimize_kernel_size(image_size: Tuple[int, int], base_kernel_size: int = 5) -> int:
    """
    Optimize kernel size based on image dimensions.
    
    Args:
        image_size: (height, width) of the image
        base_kernel_size: Base kernel size to scale from
        
    Returns:
        Optimized kernel size (always odd)
    """
    height, width = image_size
    image_diagonal = np.sqrt(height**2 + width**2)
    
    # Scale kernel size based on image diagonal
    scale_factor = image_diagonal / 1000.0  # Baseline for 1000px diagonal
    optimized_size = int(base_kernel_size * scale_factor)
    
    # Ensure it's odd and at least 3
    if optimized_size % 2 == 0:
        optimized_size += 1
    
    return max(3, optimized_size)

def create_processing_roi(
    image_shape: Tuple[int, int], 
    center: Tuple[int, int], 
    radius: int,
    padding: float = 0.2
) -> Tuple[slice, slice]:
    """
    Create a region of interest (ROI) for focused processing.
    
    Args:
        image_shape: (height, width) of the image
        center: (x, y) center coordinates
        radius: Radius of the main feature
        padding: Additional padding factor
        
    Returns:
        Tuple of (row_slice, col_slice) for ROI
    """
    height, width = image_shape
    center_x, center_y = center
    
    # Calculate ROI bounds with padding
    roi_radius = int(radius * (1 + padding))
    
    y_min = max(0, center_y - roi_radius)
    y_max = min(height, center_y + roi_radius)
    x_min = max(0, center_x - roi_radius)
    x_max = min(width, center_x + roi_radius)
    
    return slice(y_min, y_max), slice(x_min, x_max)

def adaptive_threshold_parameters(
    image: np.ndarray,
    base_block_size: int = 11,
    base_c: float = 2.0
) -> Tuple[int, float]:
    """
    Adaptively determine threshold parameters based on image characteristics.
    
    Args:
        image: Input grayscale image
        base_block_size: Base block size for adaptive threshold
        base_c: Base C parameter for adaptive threshold
        
    Returns:
        Tuple of (optimized_block_size, optimized_c)
    """
    # Analyze image statistics
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    # Adjust block size based on image size
    height, width = image.shape
    image_size_factor = np.sqrt(height * width) / 500.0  # Baseline for 500x500 image
    
    optimized_block_size = int(base_block_size * image_size_factor)
    if optimized_block_size % 2 == 0:
        optimized_block_size += 1
    optimized_block_size = max(3, optimized_block_size)
    
    # Adjust C based on image contrast
    contrast_factor = std_intensity / 128.0  # Normalize to typical range
    optimized_c = base_c * (1 + contrast_factor)
    
    return optimized_block_size, optimized_c

def multi_scale_processing(
    image: np.ndarray,
    processing_func: Callable,
    scales: List[float] = [0.5, 1.0, 2.0],
    combine_method: str = 'max'
) -> np.ndarray:
    """
    Apply processing function at multiple scales and combine results.
    
    Args:
        image: Input image
        processing_func: Function to apply at each scale
        scales: List of scale factors
        combine_method: Method to combine results ('max', 'mean', 'weighted')
        
    Returns:
        Combined result image
    """
    original_shape = image.shape[:2]
    results = []
    
    for scale in scales:
        if scale != 1.0:
            # Resize image
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            scaled_image = image
        
        # Apply processing function
        processed = processing_func(scaled_image)
        
        # Resize result back to original size
        if scale != 1.0:
            processed = cv2.resize(processed, (original_shape[1], original_shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
        
        results.append(processed)
    
    # Combine results
    if combine_method == 'max':
        combined = np.maximum.reduce(results)
    elif combine_method == 'mean':
        combined = np.mean(results, axis=0).astype(results[0].dtype)
    elif combine_method == 'weighted':
        # Weight by scale (larger scales get more weight)
        weights = np.array(scales) / np.sum(scales)
        combined = np.zeros_like(results[0], dtype=np.float32)
        for result, weight in zip(results, weights):
            combined += result.astype(np.float32) * weight
        combined = combined.astype(results[0].dtype)
    else:
        combined = results[0]  # Default to first result
    
    return combined

def batch_process_images(
    images: List[np.ndarray],
    processing_func: Callable,
    batch_size: int = 4,
    show_progress: bool = True
) -> List[Any]:
    """
    Process a batch of images efficiently.
    
    Args:
        images: List of input images
        processing_func: Function to apply to each image
        batch_size: Number of images to process in each batch
        show_progress: Whether to show progress information
        
    Returns:
        List of processing results
    """
    results = []
    total_images = len(images)
    
    for i in range(0, total_images, batch_size):
        batch_end = min(i + batch_size, total_images)
        batch = images[i:batch_end]
        
        if show_progress:
            logger.info(f"Processing batch {i//batch_size + 1} "
                       f"({i+1}-{batch_end} of {total_images})")
        
        # Process batch
        batch_results = []
        for image in batch:
            result = processing_func(image)
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Force garbage collection between batches
        gc.collect()
    
    return results

def check_gpu_availability() -> bool:
    """
    Check if GPU acceleration is available for OpenCV.
    
    Returns:
        True if GPU acceleration is available, False otherwise
    """
    try:
        # Check if OpenCV was built with CUDA support
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("GPU acceleration available")
            return True
        else:
            logger.info("No CUDA-enabled devices found")
            return False
    except AttributeError:
        logger.info("OpenCV not built with CUDA support")
        return False

def optimize_image_dtype(image: np.ndarray, target_dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Optimize image data type for processing efficiency.
    
    Args:
        image: Input image
        target_dtype: Target data type
        
    Returns:
        Image with optimized data type
    """
    if image.dtype == target_dtype:
        return image
    
    # Handle different conversions
    if target_dtype == np.uint8:
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Assume float images are normalized to [0, 1]
            return (np.clip(image, 0, 1) * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            return (image >> 8).astype(np.uint8)  # Simple bit shift
    elif target_dtype == np.float32:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
    
    return image.astype(target_dtype)

def memory_efficient_morphology(
    image: np.ndarray,
    kernel: np.ndarray,
    operation: str = 'open',
    iterations: int = 1
) -> np.ndarray:
    """
    Perform morphological operations with memory optimization.
    
    Args:
        image: Input binary image
        kernel: Morphological kernel
        operation: Type of operation ('open', 'close', 'erode', 'dilate')
        iterations: Number of iterations
        
    Returns:
        Processed image
    """
    # Use in-place operations when possible
    if operation == 'erode':
        result = cv2.erode(image, kernel, iterations=iterations)
    elif operation == 'dilate':
        result = cv2.dilate(image, kernel, iterations=iterations)
    elif operation == 'open':
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        result = image
    
    return result

def parallel_region_processing(
    image: np.ndarray,
    processing_func: Callable,
    num_regions: int = 4,
    overlap: int = 50
) -> np.ndarray:
    """
    Process image in parallel regions (simulated parallel processing).
    
    Args:
        image: Input image
        processing_func: Function to apply to each region
        num_regions: Number of regions to divide the image into
        overlap: Overlap between regions in pixels
        
    Returns:
        Reconstructed processed image
    """
    height, width = image.shape[:2]
    
    # Calculate region dimensions
    region_height = height // num_regions + overlap
    
    results = []
    for i in range(num_regions):
        # Calculate region bounds
        start_y = max(0, i * (height // num_regions) - overlap // 2)
        end_y = min(height, start_y + region_height)
        
        # Extract region
        region = image[start_y:end_y, :]
        
        # Process region
        processed_region = processing_func(region)
        
        # Store with position info
        results.append((start_y, end_y, processed_region))
    
    # Reconstruct image
    output = np.zeros_like(image)
    for start_y, end_y, processed_region in results:
        # Handle overlap by blending
        if start_y > 0 and overlap > 0:
            # Blend overlapping area
            blend_start = overlap // 2
            for y in range(blend_start):
                alpha = y / blend_start
                blend_y = start_y + y
                if blend_y < height:
                    output[blend_y] = (alpha * processed_region[y] + 
                                     (1 - alpha) * output[blend_y])
            # Copy non-overlapping area
            copy_start = start_y + blend_start
            copy_end = min(end_y, height)
            region_start = blend_start
            region_end = region_start + (copy_end - copy_start)
            output[copy_start:copy_end] = processed_region[region_start:region_end]
        else:
            # No overlap, direct copy
            output[start_y:end_y] = processed_region
    
    return output

# Test function
def test_performance_functions():
    """Test the performance optimization functions."""
    logger.info("Testing performance optimization functions...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    
    # Test resize for processing
    resized, scale = resize_for_processing(test_image, max_dimension=400)
    logger.info(f"Resized from {test_image.shape} to {resized.shape}, scale: {scale:.3f}")
    
    # Test coordinate scaling
    coords = np.array([[100, 200], [300, 400]])
    scaled_coords = scale_coordinates(coords, scale)
    logger.info(f"Coordinates scaled: {coords} -> {scaled_coords}")
    
    # Test kernel size optimization
    kernel_size = optimize_kernel_size(test_gray.shape)
    logger.info(f"Optimized kernel size for {test_gray.shape}: {kernel_size}")
    
    # Test adaptive threshold parameters
    block_size, c = adaptive_threshold_parameters(test_gray)
    logger.info(f"Adaptive threshold params: block_size={block_size}, c={c:.2f}")
    
    # Test ROI creation
    roi_slices = create_processing_roi(test_gray.shape, (300, 400), 100)
    logger.info(f"ROI slices: {roi_slices}")
    
    # Test performance timer decorator
    @performance_timer
    def dummy_processing(img):
        return cv2.GaussianBlur(img, (15, 15), 0)
    
    blurred = dummy_processing(test_gray)
    logger.info(f"Performance timer test completed")
    
    # Test multi-scale processing
    def simple_edge_detection(img):
        return cv2.Canny(img, 50, 150)
    
    multi_scale_result = multi_scale_processing(test_gray, simple_edge_detection)
    logger.info(f"Multi-scale processing result shape: {multi_scale_result.shape}")
    
    # Test GPU availability check
    gpu_available = check_gpu_availability()
    logger.info(f"GPU available: {gpu_available}")
    
    # Test dtype optimization
    float_image = test_gray.astype(np.float32) / 255.0
    optimized = optimize_image_dtype(float_image, np.uint8)
    logger.info(f"Dtype optimization: {float_image.dtype} -> {optimized.dtype}")
    
    logger.info("Performance optimization function tests completed")

if __name__ == "__main__":
    test_performance_functions()
