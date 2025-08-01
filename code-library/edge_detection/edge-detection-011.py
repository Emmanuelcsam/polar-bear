#!/usr/bin/env python3
"""
Difference of Gaussians Edge Detection
Category: edge_detection
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Difference of Gaussians Edge Detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    # Apply two Gaussian blurs with different sigmas
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
    # Compute difference
    dog = g1.astype(np.float32) - g2.astype(np.float32)
    # Normalize and threshold
    result = np.uint8(np.clip(np.abs(dog) * 10, 0, 255))
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"dog_edges_output.png", result)
            print(f"Saved to dog_edges_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python dog_edges.py <image_path>")
