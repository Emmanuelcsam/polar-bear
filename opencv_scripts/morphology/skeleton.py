#!/usr/bin/env python3
"""
Morphological Skeleton - Thin objects to lines
Category: morphology
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Morphological Skeleton - Thin objects to lines
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    size = np.size(gray)
    skel = np.zeros(gray.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(gray, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(gray, temp)
        skel = cv2.bitwise_or(skel, temp)
        gray = eroded.copy()
        zeros = size - cv2.countNonZero(gray)
        if zeros == size:
            done = True
    result = skel
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"skeleton_output.png", result)
            print(f"Saved to skeleton_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python skeleton.py <image_path>")
