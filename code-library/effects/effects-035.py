#!/usr/bin/env python3
"""
Super Resolution Upscaling
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Super Resolution Upscaling
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Simple super resolution using cubic interpolation
    result = cv2.resize(result, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Apply sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    result = cv2.filter2D(result, -1, kernel)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"super_resolution_output.png", result)
            print(f"Saved to super_resolution_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python super_resolution.py <image_path>")
