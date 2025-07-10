#!/usr/bin/env python3
"""
Kaleidoscope Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Kaleidoscope Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Create kaleidoscope effect
    h, w = result.shape[:2]
    center = (w // 2, h // 2)
    # Get quadrant
    quadrant = result[:h//2, :w//2]
    # Create mirrored sections
    result[:h//2, :w//2] = quadrant
    result[:h//2, w//2:] = cv2.flip(quadrant, 1)
    result[h//2:, :w//2] = cv2.flip(quadrant, 0)
    result[h//2:, w//2:] = cv2.flip(quadrant, -1)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"kaleidoscope_output.png", result)
            print(f"Saved to kaleidoscope_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python kaleidoscope.py <image_path>")
