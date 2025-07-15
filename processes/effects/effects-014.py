#!/usr/bin/env python3
"""
Posterize Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Posterize Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Posterize effect
    n_bits = 4
    mask = 256 - (1 << n_bits)
    result = cv2.bitwise_and(result, mask)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"posterize_effect_output.png", result)
            print(f"Saved to posterize_effect_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python posterize_effect.py <image_path>")
