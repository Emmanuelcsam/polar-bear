#!/usr/bin/env python3
"""
Pad Image with Border
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Pad Image with Border
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"pad_image_output.png", result)
            print(f"Saved to pad_image_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python pad_image.py <image_path>")
