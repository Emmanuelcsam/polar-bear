#!/usr/bin/env python3
"""
Gaussian Blur - Smoothing with Gaussian kernel
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Gaussian Blur - Smoothing with Gaussian kernel
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    result = cv2.GaussianBlur(result, (5, 5), 1.0)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"gaussian_blur_output.png", result)
            print(f"Saved to gaussian_blur_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python gaussian_blur.py <image_path>")
