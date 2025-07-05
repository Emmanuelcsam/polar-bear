#!/usr/bin/env python3
"""
Pixelate Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Pixelate Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Pixelate effect
    pixel_size = 10
    h, w = result.shape[:2]
    # Resize down
    temp = cv2.resize(result, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    # Resize up
    result = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"pixelate_effect_output.png", result)
            print(f"Saved to pixelate_effect_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python pixelate_effect.py <image_path>")
