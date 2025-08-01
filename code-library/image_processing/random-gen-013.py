#!/usr/bin/env python3
"""
Enhanced Image Preprocessing Module
==================================
Advanced image preprocessing functions for fiber optic and general image analysis.
Includes adaptive resizing, illumination correction, and multi-scale filtering.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, Callable

class ImageProcessor:
    """Advanced image processing with optimization and enhancement capabilities"""
    
    def __init__(self, max_processing_size: int = 1024):
        """
        Initialize the image processor.
        
        Args:
            max_processing_size: Maximum size for processing optimization
        """
        self.max_processing_size = max_processing_size
    
    def resize_for_processing(self, image: np.ndarray, max_size: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Resize image for faster processing if needed.
        
        Args:
            image: Input image
            max_size: Maximum size constraint (optional)
            
        Returns:
            Tuple of (processed_image, scale_factor)
        """
        if max_size is None:
            max_size = self.max_processing_size
            
        h, w = image.shape[:2]
        scale_factor = 1.0
        
        if max(h, w) > max_size:
            scale_factor = max_size / max(h, w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logging.debug(f"Image resized from {w}x{h} to {new_w}x{new_h} (scale: {scale_factor:.2f})")
            return resized, scale_factor
        
        return image, scale_factor

def load_and_preprocess_image(image_path: Union[str, Path], 
                            clahe_clip_limit: float = 2.0,
                            clahe_tile_size: Tuple[int, int] = (8, 8),
                            blur_kernel_size: Tuple[int, int] = (5, 5)) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load and preprocess an image with adaptive enhancement.
    
    Args:
        image_path: Path to the image file
        clahe_clip_limit: CLAHE contrast limiting parameter
        clahe_tile_size: CLAHE tile grid size
        blur_kernel_size: Gaussian blur kernel size
        
    Returns:
        Tuple of (original_bgr, original_gray, processed_image) or None if failed
    """
    image_path = Path(image_path)
    if not image_path.exists() or not image_path.is_file():
        logging.error(f"Image file not found: {image_path}")
        return None

    # Load image
    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        logging.error(f"Failed to load image: {image_path}")
        return None
    
    logging.info(f"Image '{image_path.name}' loaded successfully")

    # Convert to grayscale
    gray_image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
    enhanced_image = clahe.apply(gray_image)
    
    # Apply Gaussian blur for noise reduction
    # Ensure kernel size is odd
    blur_kernel_size_fixed = (blur_kernel_size[0] if blur_kernel_size[0] % 2 == 1 else blur_kernel_size[0] + 1,
                              blur_kernel_size[1] if blur_kernel_size[1] % 2 == 1 else blur_kernel_size[1] + 1)
    processed_image = cv2.GaussianBlur(enhanced_image, blur_kernel_size_fixed, 0)
    
    logging.debug(f"Preprocessing complete: CLAHE({clahe_clip_limit}, {clahe_tile_size}), Blur{blur_kernel_size}")
    
    return original_bgr, gray_image, processed_image

def correct_illumination_advanced(gray_image: np.ndarray, 
                                kernel_size: int = 50,
                                method: str = "rolling_ball") -> np.ndarray:
    """
    Advanced illumination correction using various methods.
    
    Args:
        gray_image: Input grayscale image
        kernel_size: Size of the correction kernel
        method: Correction method ("rolling_ball", "top_hat", "gaussian")
        
    Returns:
        Illumination-corrected image
    """
    if method == "rolling_ball":
        # Rolling ball algorithm using morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Subtract background and normalize
        corrected_int16 = cv2.subtract(gray_image.astype(np.int16), background.astype(np.int16))
        corrected_int16 = corrected_int16 + 128  # Shift to mid-gray
        corrected = np.clip(corrected_int16, 0, 255).astype(np.uint8)
        
    elif method == "top_hat":
        # Top-hat transformation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        corrected = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
        
    elif method == "gaussian":
        # Gaussian background subtraction
        background = cv2.GaussianBlur(gray_image, (kernel_size*2+1, kernel_size*2+1), kernel_size/3)
        corrected = cv2.subtract(gray_image, background)
        corrected = cv2.add(corrected, np.full_like(corrected, 128))  # Add offset
        
    else:
        logging.warning(f"Unknown method '{method}', returning original image")
        return gray_image
    
    return corrected

def enhance_contrast_adaptive(image: np.ndarray, 
                            method: str = "clahe",
                            clip_limit: float = 2.0,
                            tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Adaptive contrast enhancement with multiple methods.
    
    Args:
        image: Input grayscale image
        method: Enhancement method ("clahe", "histogram_eq", "adaptive_histogram")
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile size
        
    Returns:
        Contrast-enhanced image
    """
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(image)
        
    elif method == "histogram_eq":
        return cv2.equalizeHist(image)
        
    elif method == "adaptive_histogram":
        # Local histogram equalization
        enhanced = np.zeros_like(image)
        h, w = image.shape
        tile_h, tile_w = h // tile_size[0], w // tile_size[1]
        
        for i in range(tile_size[0]):
            for j in range(tile_size[1]):
                y1, y2 = i * tile_h, min((i + 1) * tile_h, h)
                x1, x2 = j * tile_w, min((j + 1) * tile_w, w)
                tile = image[y1:y2, x1:x2]
                enhanced[y1:y2, x1:x2] = cv2.equalizeHist(tile)
        
        return enhanced
    
    else:
        logging.warning(f"Unknown method '{method}', returning original image")
        return image

def denoise_image(image: np.ndarray, 
                 method: str = "bilateral",
                 strength: float = 5.0) -> np.ndarray:
    """
    Advanced denoising with multiple algorithms.
    
    Args:
        image: Input image
        method: Denoising method ("bilateral", "non_local_means", "gaussian", "median")
        strength: Denoising strength parameter
        
    Returns:
        Denoised image
    """
    if method == "bilateral":
        return cv2.bilateralFilter(image, int(strength*2), strength*10, strength*10)
        
    elif method == "non_local_means":
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
            
    elif method == "gaussian":
        kernel_size = int(strength * 2) * 2 + 1  # Ensure odd
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), strength)
        
    elif method == "median":
        kernel_size = int(strength * 2) * 2 + 1  # Ensure odd
        return cv2.medianBlur(image, kernel_size)
    
    else:
        logging.warning(f"Unknown denoising method '{method}', returning original image")
        return image

def multi_scale_enhancement(image: np.ndarray, 
                          scales: list = [0.5, 1.0, 1.5],
                          enhancement_func: Optional[Callable] = None) -> np.ndarray:
    """
    Apply enhancement at multiple scales and combine results.
    
    Args:
        image: Input image
        scales: List of scale factors
        enhancement_func: Function to apply at each scale
        
    Returns:
        Multi-scale enhanced image
    """
    if enhancement_func is None:
        enhancement_func = lambda x: enhance_contrast_adaptive(x, "clahe")
    
    h, w = image.shape[:2]
    enhanced_images = []
    
    for scale in scales:
        if scale == 1.0:
            enhanced = enhancement_func(image)
        else:
            # Resize
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Enhance
            enhanced_resized = enhancement_func(resized)
            
            # Resize back
            enhanced = cv2.resize(enhanced_resized, (w, h), interpolation=cv2.INTER_LINEAR)
        
        enhanced_images.append(enhanced.astype(np.float32))
    
    # Combine using weighted average
    weights = [1.0 / len(scales)] * len(scales)
    combined = np.zeros_like(enhanced_images[0])
    
    for enhanced, weight in zip(enhanced_images, weights):
        combined += enhanced * weight
    
    return np.clip(combined, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    """Test the preprocessing functions"""
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Create a test image
    test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    # Add some structure
    cv2.circle(test_image, (100, 100), 80, (180,), -1)
    cv2.circle(test_image, (100, 100), 10, (120,), -1)
    
    # Add noise
    noise = np.random.normal(0, 20, test_image.shape).astype(np.int16)
    noisy_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print("Testing image preprocessing functions...")
    
    # Test contrast enhancement
    enhanced = enhance_contrast_adaptive(noisy_image, "clahe")
    print(f"Enhanced image shape: {enhanced.shape}")
    
    # Test denoising
    denoised = denoise_image(enhanced, "bilateral", 5.0)
    print(f"Denoised image shape: {denoised.shape}")
    
    # Test illumination correction
    corrected = correct_illumination_advanced(denoised, 30, "rolling_ball")
    print(f"Illumination corrected image shape: {corrected.shape}")
    
    # Test multi-scale enhancement
    multi_enhanced = multi_scale_enhancement(test_image, [0.5, 1.0, 1.5])
    print(f"Multi-scale enhanced image shape: {multi_enhanced.shape}")
    
    # Test image processor
    processor = ImageProcessor(512)
    resized, scale = processor.resize_for_processing(test_image, 100)
    print(f"Resized image: {resized.shape}, scale factor: {scale:.2f}")
    
    print("All preprocessing tests completed successfully!")
