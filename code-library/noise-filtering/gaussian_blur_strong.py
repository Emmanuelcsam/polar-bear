#!/usr/bin/env python3
"""
Strong Gaussian Blur
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Strong Gaussian Blur
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    result = cv2.GaussianBlur(result, (15, 15), 5.0)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"gaussian_blur_strong_output.png", result)
            print(f"Saved to gaussian_blur_strong_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python gaussian_blur_strong.py <image_path>")
