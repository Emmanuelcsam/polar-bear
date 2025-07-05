#!/usr/bin/env python3
"""
Resize Image to Half Size
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Resize Image to Half Size
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    height, width = result.shape[:2]
    new_height, new_width = height // 2, width // 2
    result = cv2.resize(result, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"resize_half_output.png", result)
            print(f"Saved to resize_half_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python resize_half.py <image_path>")
