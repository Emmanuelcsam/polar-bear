#!/usr/bin/env python3
"""
Add Poisson Noise
Category: noise
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Add Poisson Noise
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Add Poisson noise
    noise = np.random.poisson(result).astype(np.float32)
    result = np.clip(noise, 0, 255).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"add_poisson_noise_output.png", result)
            print(f"Saved to add_poisson_noise_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python add_poisson_noise.py <image_path>")
