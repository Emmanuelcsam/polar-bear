#!/usr/bin/env python3
"""
Histogram Matching
Category: histogram
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Histogram Matching
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Match histogram to a Gaussian distribution
    if len(result.shape) == 3:
        # Convert to grayscale for simplicity
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result
    # Create target histogram (Gaussian)
    hist_target = np.exp(-0.5 * ((np.arange(256) - 128) / 50) ** 2)
    hist_target = (hist_target / hist_target.sum() * gray.size).astype(int)
    # Calculate CDF
    hist_source, _ = np.histogram(gray.flatten(), 256, [0, 256])
    cdf_source = hist_source.cumsum()
    cdf_target = hist_target.cumsum()
    # Create lookup table
    lookup = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and cdf_source[i] > cdf_target[j]:
            j += 1
        lookup[i] = j
    # Apply lookup table
    result = lookup[gray]
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"histogram_matching_output.png", result)
            print(f"Saved to histogram_matching_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python histogram_matching.py <image_path>")
