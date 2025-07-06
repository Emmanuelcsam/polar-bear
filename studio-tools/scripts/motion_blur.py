#!/usr/bin/env python3
"""
Motion Blur - Simulate motion effect
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Motion Blur - Simulate motion effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = np.zeros((15, 15))
    np.fill_diagonal(kernel, 1)
    kernel = kernel / 15
    result = cv2.filter2D(result, -1, kernel)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"motion_blur_output.png", result)
            print(f"Saved to motion_blur_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python motion_blur.py <image_path>")
