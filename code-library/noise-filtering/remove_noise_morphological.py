#!/usr/bin/env python3
"""
Remove Noise with Morphological Operations
Category: noise
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Remove Noise with Morphological Operations
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Remove noise using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    # Remove salt noise
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    # Remove pepper noise
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"remove_noise_morphological_output.png", result)
            print(f"Saved to remove_noise_morphological_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python remove_noise_morphological.py <image_path>")
