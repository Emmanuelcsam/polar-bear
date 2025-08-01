#!/usr/bin/env python3
"""
Histogram Equalization
Category: histogram
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Histogram Equalization
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    if len(result.shape) == 3:
        # Convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = cv2.equalizeHist(result)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"histogram_equalization_output.png", result)
            print(f"Saved to histogram_equalization_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python histogram_equalization.py <image_path>")
