#!/usr/bin/env python3
"""
Cartoon Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Cartoon Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Apply cartoon effect
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    # Apply bilateral filter
    smooth = cv2.bilateralFilter(result, 15, 80, 80)
    # Get edges
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
    # Convert edges to color
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Combine
    result = cv2.bitwise_and(smooth, edges)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"cartoon_effect_output.png", result)
            print(f"Saved to cartoon_effect_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python cartoon_effect.py <image_path>")
