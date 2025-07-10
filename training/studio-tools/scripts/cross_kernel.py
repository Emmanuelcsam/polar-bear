#!/usr/bin/env python3
"""
Morphology with Cross Kernel
Category: morphology
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Morphology with Cross Kernel
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"cross_kernel_output.png", result)
            print(f"Saved to cross_kernel_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python cross_kernel.py <image_path>")
