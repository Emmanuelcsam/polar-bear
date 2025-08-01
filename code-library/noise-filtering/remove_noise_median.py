#!/usr/bin/env python3
"""
Remove Noise with Median Filter
Category: noise
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Remove Noise with Median Filter
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Remove noise using median filter
    result = cv2.medianBlur(result, 5)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"remove_noise_median_output.png", result)
            print(f"Saved to remove_noise_median_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python remove_noise_median.py <image_path>")
