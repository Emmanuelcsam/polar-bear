#!/usr/bin/env python3
"""
Image Preprocessing Module
=========================
Standalone module for advanced image preprocessing techniques
including denoising, enhancement, and normalization.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def advanced_preprocessing(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Advanced preprocessing with anisotropic diffusion and enhancement.
    
    Args:
        image: Input grayscale image
        **kwargs: Optional parameters for preprocessing
        
    Returns:
        Enhanced image
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to float
    image_float = image.astype(np.float32) / 255.0
    
    # 1. Bilateral filtering for noise reduction while preserving edges
    bilateral = cv2.bilateralFilter(
        (image_float * 255).astype(np.uint8), 
        kwargs.get('bilateral_d', 9),
        kwargs.get('bilateral_sigma_color', 75),
        kwargs.get('bilateral_sigma_space', 75)
    ).astype(np.float32) / 255.0
    
    # 2. Anisotropic diffusion
    diffused = anisotropic_diffusion(
        bilateral,
        iterations=kwargs.get('diffusion_iterations', 10),
        kappa=kwargs.get('diffusion_kappa', 30.0),
        gamma=kwargs.get('diffusion_gamma', 0.15)
    )
    
    # 3. Coherence enhancement
    coherence_enhanced = coherence_enhancement(diffused)
    
    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    enhanced_uint8 = (coherence_enhanced * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=kwargs.get('clahe_clip_limit', 2.0),
        tileGridSize=kwargs.get('clahe_tile_size', (8, 8))
    )
    enhanced = clahe.apply(enhanced_uint8).astype(np.float32) / 255.0
    
    logger.info("Advanced preprocessing completed")
    return enhanced

def anisotropic_diffusion(image: np.ndarray, iterations: int = 10, 
                         kappa: float = 30.0, gamma: float = 0.15) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion for edge-preserving smoothing.
    
    Args:
        image: Input image
        iterations: Number of diffusion iterations
        kappa: Diffusion threshold
        gamma: Diffusion rate
        
    Returns:
        Diffused image
    """
    result = image.copy()
    
    for _ in range(iterations):
        # Calculate gradients
        grad_x = np.gradient(result, axis=1)
        grad_y = np.gradient(result, axis=0)
        
        # Calculate diffusion coefficients
        grad_mag_sq = grad_x**2 + grad_y**2
        diffusion_coeff = np.exp(-(grad_mag_sq / (kappa**2)))
        
        # Apply diffusion
        diff_x = np.gradient(diffusion_coeff * grad_x, axis=1)
        diff_y = np.gradient(diffusion_coeff * grad_y, axis=0)
        
        result = result + gamma * (diff_x + diff_y)
    
    return np.clip(result, 0, 1)

def coherence_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhance linear structures using coherence-based filtering.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image with improved linear structure visibility
    """
    # Calculate structure tensor
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    # Gaussian smoothing for structure tensor components
    sigma = 1.0
    J11 = cv2.GaussianBlur(grad_x * grad_x, (0, 0), sigma)
    J12 = cv2.GaussianBlur(grad_x * grad_y, (0, 0), sigma)
    J22 = cv2.GaussianBlur(grad_y * grad_y, (0, 0), sigma)
    
    # Calculate eigenvalues
    trace = J11 + J22
    det = J11 * J22 - J12 * J12
    discriminant = np.sqrt(np.maximum(trace**2 - 4*det, 0))
    
    lambda1 = 0.5 * (trace + discriminant)
    lambda2 = 0.5 * (trace - discriminant)
    
    # Coherence measure
    coherence = np.divide(
        (lambda1 - lambda2)**2,
        (lambda1 + lambda2)**2 + 1e-10,
        out=np.zeros_like(lambda1),
        where=((lambda1 + lambda2)**2 + 1e-10) != 0
    )
    
    # Enhancement based on coherence
    enhanced = image + 0.5 * coherence * (lambda1 - lambda2)
    
    return np.clip(enhanced, 0, 1)

def multiscale_enhancement(image: np.ndarray, scales: list = [0.5, 1.0, 2.0]) -> np.ndarray:
    """
    Multi-scale enhancement for detecting defects at different scales.
    
    Args:
        image: Input image
        scales: List of scale factors
        
    Returns:
        Multi-scale enhanced image
    """
    h, w = image.shape
    enhanced_pyramid = []
    
    for scale in scales:
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (new_w, new_h))
        else:
            scaled = image.copy()
        
        # Apply enhancement at this scale
        enhanced_scale = advanced_preprocessing(scaled)
        
        # Resize back to original size
        if scale != 1.0:
            enhanced_scale = cv2.resize(enhanced_scale, (w, h))
            
        enhanced_pyramid.append(enhanced_scale)
    
    # Combine scales
    result = np.mean(enhanced_pyramid, axis=0)
    return np.clip(result, 0, 1)

def adaptive_threshold_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding-based enhancement for defect visibility.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to uint8 for adaptive threshold
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Multiple adaptive thresholds
    thresh1 = cv2.adaptiveThreshold(
        img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    thresh2 = cv2.adaptiveThreshold(
        img_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Combine thresholds
    combined = cv2.bitwise_and(thresh1, thresh2)
    
    # Use as enhancement mask
    mask = combined.astype(np.float32) / 255.0
    enhanced = image * (1 + 0.3 * mask)
    
    return np.clip(enhanced, 0, 1)

def test_preprocessing():
    """Test the preprocessing functions with a synthetic image."""
    logger.info("Testing image preprocessing functions...")
    
    # Create synthetic test image
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    # Add some structure
    test_image[100:120, :] = 0.8  # Horizontal line
    test_image[:, 128:135] = 0.9  # Vertical line
    
    # Add noise
    noise = np.random.normal(0, 0.1, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 1)
    
    logger.info(f"Input image shape: {test_image.shape}")
    logger.info(f"Input image range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    
    # Test advanced preprocessing
    enhanced = advanced_preprocessing(test_image)
    logger.info(f"Enhanced image range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    
    # Test multiscale enhancement
    multiscale = multiscale_enhancement(test_image)
    logger.info(f"Multiscale enhanced range: [{multiscale.min():.3f}, {multiscale.max():.3f}]")
    
    # Test adaptive enhancement
    adaptive = adaptive_threshold_enhancement(test_image)
    logger.info(f"Adaptive enhanced range: [{adaptive.min():.3f}, {adaptive.max():.3f}]")
    
    logger.info("All preprocessing tests completed successfully!")
    
    return {
        'original': test_image,
        'enhanced': enhanced,
        'multiscale': multiscale,
        'adaptive': adaptive
    }

if __name__ == "__main__":
    # Run tests
    results = test_preprocessing()
    logger.info("Image preprocessing module is ready for use!")
