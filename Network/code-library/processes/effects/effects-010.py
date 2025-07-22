#!/usr/bin/env python3
"""
Negative (Invert) Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Negative (Invert) Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Create negative effect
    result = 255 - result
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"negative_effect_output.png", result)
            print(f"Saved to negative_effect_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python negative_effect.py <image_path>")
