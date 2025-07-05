#!/usr/bin/env python3
"""
Add Gaussian Noise
Category: noise
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Add Gaussian Noise
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Add Gaussian noise
    mean = 0
    sigma = 25
    noise = np.random.normal(mean, sigma, result.shape).astype(np.float32)
    result = cv2.add(result.astype(np.float32), noise)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"add_gaussian_noise_output.png", result)
            print(f"Saved to add_gaussian_noise_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python add_gaussian_noise.py <image_path>")
