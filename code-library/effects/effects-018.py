#!/usr/bin/env python3
"""
Vintage Photo Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Vintage Photo Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Create vintage effect
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    # Add yellow tint
    result[:,:,0] = np.clip(result[:,:,0] * 0.7, 0, 255)  # Reduce blue
    # Add noise
    noise = np.random.normal(0, 10, result.shape)
    result = np.clip(result + noise, 0, 255).astype(np.uint8)
    # Add vignette
    rows, cols = result.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2.5)
    kernel_y = cv2.getGaussianKernel(rows, rows/2.5)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    for i in range(3):
        result[:,:,i] = (result[:,:,i] * mask).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"vintage_effect_output.png", result)
            print(f"Saved to vintage_effect_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python vintage_effect.py <image_path>")
