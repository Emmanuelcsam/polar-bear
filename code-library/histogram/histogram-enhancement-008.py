#!/usr/bin/env python3
"""
Histogram Stretching
Category: histogram
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Histogram Stretching
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Stretch histogram to full range
    if len(result.shape) == 3:
        for i in range(3):
            channel = result[:,:,i]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                result[:,:,i] = ((channel - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    else:
        min_val = result.min()
        max_val = result.max()
        if max_val > min_val:
            result = ((result - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"histogram_stretching_output.png", result)
            print(f"Saved to histogram_stretching_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python histogram_stretching.py <image_path>")
