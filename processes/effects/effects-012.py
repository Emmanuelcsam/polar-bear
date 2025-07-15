#!/usr/bin/env python3
"""
Pencil Sketch Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Pencil Sketch Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Create pencil sketch effect
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result
    # Invert
    inv = 255 - gray
    # Blur
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    # Blend
    result = cv2.divide(gray, 255 - blur, scale=256)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"pencil_sketch_output.png", result)
            print(f"Saved to pencil_sketch_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python pencil_sketch.py <image_path>")
