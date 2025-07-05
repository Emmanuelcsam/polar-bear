#!/usr/bin/env python3
"""
Flip Image Vertically
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Flip Image Vertically
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    result = cv2.flip(result, 0)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"flip_vertical_output.png", result)
            print(f"Saved to flip_vertical_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python flip_vertical.py <image_path>")
