#!/usr/bin/env python3
"""
Cross Processing Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Cross Processing Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Simulate cross processing
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    # Adjust individual channels
    result[:,:,0] = np.clip(result[:,:,0] * 0.7, 0, 255)  # Blue
    result[:,:,1] = np.clip(result[:,:,1] * 1.2, 0, 255)  # Green
    result[:,:,2] = np.clip(result[:,:,2] * 1.5, 0, 255)  # Red
    result = result.astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"cross_process_output.png", result)
            print(f"Saved to cross_process_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python cross_process.py <image_path>")
