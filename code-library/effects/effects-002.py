#!/usr/bin/env python3
"""
Color Quantization
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Color Quantization
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Reduce number of colors
    n_colors = 8
    if len(result.shape) == 3:
        data = result.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        result = quantized.reshape(result.shape)
    else:
        # For grayscale, use simple quantization
        levels = np.linspace(0, 255, n_colors)
        result = np.digitize(result, levels) * (255 // (n_colors - 1))
        result = result.astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"color_quantization_output.png", result)
            print(f"Saved to color_quantization_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python color_quantization.py <image_path>")
