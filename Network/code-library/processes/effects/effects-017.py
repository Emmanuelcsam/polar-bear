#!/usr/bin/env python3
"""
Vignette Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Vignette Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Add vignette effect
    rows, cols = result.shape[:2]
    # Create vignette mask
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols/2)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/2)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    # Apply to each channel
    if len(result.shape) == 3:
        for i in range(3):
            result[:,:,i] = result[:,:,i] * mask
    else:
        result = (result * mask).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"vignette_effect_output.png", result)
            print(f"Saved to vignette_effect_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python vignette_effect.py <image_path>")
