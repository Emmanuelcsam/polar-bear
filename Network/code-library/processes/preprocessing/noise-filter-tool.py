#!/usr/bin/env python3
"""
Image Filtering and Preprocessing Functions
Extracted from multiple fiber optic analysis scripts

This module contains essential image preprocessing functions for fiber optic analysis:
- Binary thresholding with morphological operations
- Homomorphic filtering for illumination correction
- CLAHE contrast enhancement
- Noise reduction techniques
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def apply_binary_filter(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Apply binary threshold and morphological closing to clean up image.
    
    From: cladding.py, core.py, ferral.py
    
    Args:
        image: Input image (BGR or grayscale)
        threshold: Binary threshold value (0-255)
        
    Returns:
        Filtered binary image
    """
    result = image.copy()
    
    # Convert to grayscale if needed
    if len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, result = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result


def homomorphic_filter(img: np.ndarray, gamma_low: float = 0.25, gamma_high: float = 1.5, 
                      cutoff: float = 30.0) -> np.ndarray:
    """
    Apply homomorphic filtering for illumination correction.
    
    From: separation_b.py, seperation_blinux.py
    
    Args:
        img: Input grayscale image
        gamma_low: Low frequency gain
        gamma_high: High frequency gain  
        cutoff: Cutoff frequency
        
    Returns:
        Illumination corrected image
    """
    # Convert to float and add small epsilon to avoid log(0)
    img_float = img.astype(np.float64) + 1e-6
    
    # Logarithmic transform
    img_log = np.log(img_float)
    
    # FFT
    img_fft = np.fft.fft2(img_log)
    img_fft = np.fft.fftshift(img_fft)
    
    # Create high-pass filter
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Butterworth high-pass filter
    H = (gamma_high - gamma_low) * (1 - np.exp(-d**2 / (2 * cutoff**2))) + gamma_low
    
    # Apply filter
    img_fft_filtered = img_fft * H
    
    # Inverse FFT
    img_fft_filtered = np.fft.ifftshift(img_fft_filtered)
    img_filtered = np.real(np.fft.ifft2(img_fft_filtered))
    
    # Exponential transform
    img_exp = np.exp(img_filtered) - 1.0
    
    # Normalize to 8-bit
    img_normalized = np.zeros_like(img_exp)
    cv2.normalize(img_exp, img_normalized, 0, 255, cv2.NORM_MINMAX)
    
    return img_normalized.astype(np.uint8)


def apply_clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0, 
                           tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    From: separation_b.py, multiple other scripts
    
    Args:
        image: Input grayscale image
        clip_limit: Clipping limit for contrast enhancement
        tile_grid_size: Size of neighborhood for histogram equalization
        
    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def denoise_bilateral(image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                     sigma_space: float = 75) -> np.ndarray:
    """
    Edge-preserving bilateral filtering for noise reduction.
    
    From: separation_b.py, segmentation.py
    
    Args:
        image: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Denoised image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def gaussian_blur_adaptive(image: np.ndarray, kernel_size: Optional[int] = None, 
                          sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur with adaptive kernel size based on image dimensions.
    
    From: multiple scripts
    
    Args:
        image: Input image
        kernel_size: Blur kernel size (auto-calculated if None)
        sigma: Gaussian sigma (auto-calculated if 0)
        
    Returns:
        Blurred image
    """
    if kernel_size is None:
        # Adaptive kernel size based on image dimensions
        min_dim = min(image.shape[:2])
        kernel_size = max(5, min_dim // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def compute_local_variance(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Compute local variance for texture analysis.
    
    From: pixel_separation.py
    
    Args:
        img: Input grayscale image
        window_size: Size of local window
        
    Returns:
        Local variance map
    """
    # Local mean
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    local_mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
    
    # Local variance
    local_sq_mean = cv2.filter2D(img.astype(np.float32)**2, -1, kernel)
    variance = local_sq_mean - local_mean**2
    variance[variance < 0] = 0
    
    return variance


def create_sharpening_kernel() -> np.ndarray:
    """
    Create sharpening kernel for edge enhancement.
    
    From: pixel_separation_2.py
    
    Returns:
        Sharpening kernel
    """
    return np.array([[-1, -1, -1],
                     [-1,  9, -1],
                     [-1, -1, -1]])


def apply_sharpening(image: np.ndarray) -> np.ndarray:
    """
    Apply sharpening filter to enhance edges.
    
    From: pixel_separation_2.py
    
    Args:
        image: Input image
        
    Returns:
        Sharpened image
    """
    kernel = create_sharpening_kernel()
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def main():
    """Test the image filtering functions"""
    # Create a test image if no input provided
    test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    print("Testing Image Filtering Functions...")
    
    # Test binary filtering
    binary_result = apply_binary_filter(test_image)
    print(f"✓ Binary filter: {binary_result.shape}")
    
    # Test CLAHE
    gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    clahe_result = apply_clahe_enhancement(gray_test)
    print(f"✓ CLAHE enhancement: {clahe_result.shape}")
    
    # Test homomorphic filtering
    homo_result = homomorphic_filter(gray_test)
    print(f"✓ Homomorphic filter: {homo_result.shape}")
    
    # Test bilateral denoising
    bilateral_result = denoise_bilateral(gray_test)
    print(f"✓ Bilateral denoising: {bilateral_result.shape}")
    
    # Test Gaussian blur
    blur_result = gaussian_blur_adaptive(gray_test)
    print(f"✓ Adaptive Gaussian blur: {blur_result.shape}")
    
    # Test local variance
    variance_result = compute_local_variance(gray_test)
    print(f"✓ Local variance: {variance_result.shape}")
    
    # Test sharpening
    sharp_result = apply_sharpening(gray_test)
    print(f"✓ Sharpening: {sharp_result.shape}")
    
    print("\nAll image filtering functions tested successfully!")


if __name__ == "__main__":
    main()
