#!/usr/bin/env python3
"""
Defect Detection Algorithms Module
=================================
Standalone module containing various defect detection algorithms
for fiber optic cable inspection.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Any
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def do2mr_detection(image: np.ndarray, kernel_size: int = 5, 
                   gamma: float = 1.5, min_area: int = 5) -> np.ndarray:
    """
    DO2MR (Difference of 2nd order Moment Response) algorithm for defect detection.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the morphological kernel
        gamma: Enhancement parameter
        min_area: Minimum defect area in pixels
        
    Returns:
        Binary mask of detected defects
    """
    logger.info("Running DO2MR defect detection...")
    
    # Ensure input is float
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply morphological operations
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Calculate DO2MR response
    do2mr = (image - opened) - (closed - image)
    
    # Apply gamma correction for enhancement
    do2mr_enhanced = np.power(np.abs(do2mr), gamma) * np.sign(do2mr)
    
    # Threshold using Otsu's method
    do2mr_uint8 = cv2.normalize(do2mr_enhanced, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, binary = cv2.threshold(do2mr_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small components
    binary = remove_small_components(binary, min_area)
    
    logger.info(f"DO2MR detected {cv2.countNonZero(binary)} defect pixels")
    return binary

def lei_scratch_detection(image: np.ndarray, kernel_lengths: List[int] = [11, 17, 23],
                         angle_step: int = 15, min_length: int = 10) -> np.ndarray:
    """
    LEI (Linear Edge Intensity) algorithm for scratch detection.
    
    Args:
        image: Input grayscale image
        kernel_lengths: Lengths of linear kernels to use
        angle_step: Angular step in degrees
        min_length: Minimum scratch length
        
    Returns:
        Binary mask of detected scratches
    """
    logger.info("Running LEI scratch detection...")
    
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    angles = np.arange(0, 180, angle_step)
    responses = []
    
    # Apply linear filters at different angles and scales
    for length in kernel_lengths:
        for angle in angles:
            # Create linear kernel
            kernel = create_linear_kernel(length, angle)
            
            # Convolve with image
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(np.abs(response))
    
    # Combine responses (maximum response across all filters)
    combined_response = np.maximum.reduce(responses)
    
    # Threshold and post-process
    threshold = np.percentile(combined_response, 95)
    binary = (combined_response > threshold).astype(np.uint8) * 255
    
    # Morphological operations to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Filter by line length
    binary = filter_by_line_length(binary, min_length)
    
    logger.info(f"LEI detected {cv2.countNonZero(binary)} scratch pixels")
    return binary

def hessian_based_detection(image: np.ndarray, scales: List[float] = [1.0, 2.0, 3.0]) -> np.ndarray:
    """
    Hessian-based ridge detection for linear defects.
    
    Args:
        image: Input grayscale image
        scales: List of scales for multi-scale detection
        
    Returns:
        Binary mask of detected ridges/scratches
    """
    logger.info("Running Hessian-based ridge detection...")
    
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    responses = []
    
    for scale in scales:
        # Calculate Hessian matrix components
        sigma = scale
        
        # Second derivatives
        Ixx = gaussian_filter(image, sigma, order=[0, 2])
        Ixy = gaussian_filter(image, sigma, order=[1, 1])
        Iyy = gaussian_filter(image, sigma, order=[2, 0])
        
        # Eigenvalues of Hessian matrix
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        
        discriminant = np.sqrt(np.maximum(trace**2 - 4*det, 0))
        lambda1 = 0.5 * (trace + discriminant)
        lambda2 = 0.5 * (trace - discriminant)
        
        # Ridge strength (absolute value of smaller eigenvalue)
        ridge_strength = np.abs(lambda2)
        
        # Scale normalization
        ridge_strength *= scale**2
        
        responses.append(ridge_strength)
    
    # Maximum response across scales
    max_response = np.maximum.reduce(responses)
    
    # Threshold
    threshold = np.percentile(max_response, 98)
    binary = (max_response > threshold).astype(np.uint8) * 255
    
    logger.info(f"Hessian detected {cv2.countNonZero(binary)} ridge pixels")
    return binary

def scale_normalized_log(image: np.ndarray, scales: List[float] = [1.0, 2.0, 3.0, 4.0]) -> np.ndarray:
    """
    Scale-normalized Laplacian of Gaussian for blob detection.
    
    Args:
        image: Input grayscale image
        scales: List of scales
        
    Returns:
        Binary mask of detected blobs
    """
    logger.info("Running Scale-normalized LoG detection...")
    
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    responses = []
    
    for scale in scales:
        # Apply Gaussian filter
        smoothed = gaussian_filter(image, scale)
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(smoothed, cv2.CV_32F)
        
        # Scale normalization
        normalized = laplacian * (scale**2)
        
        responses.append(np.abs(normalized))
    
    # Maximum response across scales
    max_response = np.maximum.reduce(responses)
    
    # Threshold
    threshold = np.percentile(max_response, 95)
    binary = (max_response > threshold).astype(np.uint8) * 255
    
    logger.info(f"LoG detected {cv2.countNonZero(binary)} blob pixels")
    return binary

def gabor_filter_bank(image: np.ndarray, orientations: int = 8, 
                     frequencies: List[float] = [0.1, 0.2, 0.3]) -> np.ndarray:
    """
    Gabor filter bank for texture-based defect detection.
    
    Args:
        image: Input grayscale image
        orientations: Number of orientations
        frequencies: List of frequencies
        
    Returns:
        Binary mask of detected texture anomalies
    """
    logger.info("Running Gabor filter bank detection...")
    
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    responses = []
    
    for freq in frequencies:
        for i in range(orientations):
            theta = i * np.pi / orientations
            
            # Create Gabor kernel
            kernel = cv2.getGaborKernel((21, 21), 3, theta, 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
            
            # Apply filter
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(np.abs(response))
    
    # Combine responses
    combined = np.mean(responses, axis=0)
    
    # Threshold
    threshold = np.percentile(combined, 95)
    binary = (combined > threshold).astype(np.uint8) * 255
    
    logger.info(f"Gabor filters detected {cv2.countNonZero(binary)} texture anomaly pixels")
    return binary

def morphological_blob_detection(image: np.ndarray) -> np.ndarray:
    """
    Morphological blob detection using top-hat transform.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Binary mask of detected blobs
    """
    logger.info("Running morphological blob detection...")
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Multiple kernel sizes for multi-scale detection
    kernel_sizes = [3, 5, 7, 9]
    responses = []
    
    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        
        # Black-hat transform for dark blobs
        black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # White-hat transform for bright blobs
        white_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Combine
        combined = cv2.add(black_hat, white_hat)
        responses.append(combined)
    
    # Maximum response across scales
    max_response = np.maximum.reduce(responses)
    
    # Threshold
    _, binary = cv2.threshold(max_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    logger.info(f"Morphological detection found {cv2.countNonZero(binary)} blob pixels")
    return binary

def combined_defect_detection(image: np.ndarray, methods: List[str] = None, 
                            weights: Dict[str, float] = None) -> np.ndarray:
    """
    Combined defect detection using multiple algorithms.
    
    Args:
        image: Input grayscale image
        methods: List of methods to use
        weights: Weights for each method
        
    Returns:
        Combined binary mask
    """
    if methods is None:
        methods = ['do2mr', 'lei', 'hessian', 'log', 'gabor']
    
    if weights is None:
        weights = {method: 1.0 for method in methods}
    
    logger.info(f"Running combined detection with methods: {methods}")
    
    # Detection methods registry
    method_functions = {
        'do2mr': do2mr_detection,
        'lei': lei_scratch_detection,
        'hessian': hessian_based_detection,
        'log': scale_normalized_log,
        'gabor': gabor_filter_bank,
        'morphological': morphological_blob_detection
    }
    
    # Run all methods
    results = {}
    for method in methods:
        if method in method_functions:
            try:
                result = method_functions[method](image)
                results[method] = result.astype(np.float32) / 255.0
                logger.info(f"{method} completed successfully")
            except Exception as e:
                logger.error(f"Error in {method}: {e}")
                results[method] = np.zeros_like(image, dtype=np.float32)
    
    # Weighted combination
    combined = np.zeros_like(image, dtype=np.float32)
    total_weight = 0
    
    for method, result in results.items():
        weight = weights.get(method, 1.0)
        combined += weight * result
        total_weight += weight
    
    if total_weight > 0:
        combined /= total_weight
    
    # Convert to binary
    binary = (combined > 0.5).astype(np.uint8) * 255
    
    logger.info(f"Combined detection found {cv2.countNonZero(binary)} defect pixels")
    return binary

# Helper functions
def create_linear_kernel(length: int, angle: float) -> np.ndarray:
    """Create a linear kernel for line detection."""
    angle_rad = np.radians(angle)
    
    # Create kernel
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    
    # Draw line in kernel
    for i in range(length):
        x = int(center + (i - center) * np.cos(angle_rad))
        y = int(center + (i - center) * np.sin(angle_rad))
        
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    
    # Normalize
    kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
    
    return kernel

def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create output image
    result = np.zeros_like(binary)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    
    return result

def filter_by_line_length(binary: np.ndarray, min_length: int) -> np.ndarray:
    """Filter connected components by minimum line length."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = np.zeros_like(binary)
    
    for contour in contours:
        # Fit line to contour
        if len(contour) >= 5:
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Calculate approximate line length
            rect = cv2.boundingRect(contour)
            length = max(rect[2], rect[3])
            
            if length >= min_length:
                cv2.drawContours(result, [contour], -1, 255, -1)
    
    return result

def test_defect_detection():
    """Test all defect detection functions."""
    logger.info("Testing defect detection algorithms...")
    
    # Create synthetic test image with defects
    test_image = np.ones((256, 256), dtype=np.float32) * 0.5
    
    # Add synthetic defects
    # Scratch (line)
    test_image[100:105, 50:200] = 0.2
    
    # Blob defect
    cv2.circle(test_image, (180, 180), 8, 0.8, -1)
    
    # Small defects
    test_image[50:53, 150:153] = 0.1
    
    # Add noise
    noise = np.random.normal(0, 0.05, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 1)
    
    logger.info(f"Test image shape: {test_image.shape}")
    
    # Test individual methods
    methods = ['do2mr', 'lei', 'hessian', 'log', 'gabor', 'morphological']
    results = {}
    
    for method in methods:
        try:
            if method == 'do2mr':
                result = do2mr_detection(test_image)
            elif method == 'lei':
                result = lei_scratch_detection(test_image)
            elif method == 'hessian':
                result = hessian_based_detection(test_image)
            elif method == 'log':
                result = scale_normalized_log(test_image)
            elif method == 'gabor':
                result = gabor_filter_bank(test_image)
            elif method == 'morphological':
                result = morphological_blob_detection(test_image)
            
            results[method] = result
            defect_count = cv2.countNonZero(result)
            logger.info(f"{method}: {defect_count} defect pixels detected")
            
        except Exception as e:
            logger.error(f"Error testing {method}: {e}")
    
    # Test combined detection
    combined = combined_defect_detection(test_image, methods)
    combined_count = cv2.countNonZero(combined)
    logger.info(f"Combined detection: {combined_count} defect pixels")
    
    logger.info("All defect detection tests completed!")
    
    return {
        'test_image': test_image,
        'individual_results': results,
        'combined_result': combined
    }

if __name__ == "__main__":
    # Run tests
    test_results = test_defect_detection()
    logger.info("Defect detection module is ready for use!")
