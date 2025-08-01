#!/usr/bin/env python3
"""
Add Uniform Noise
Category: noise
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Add Uniform Noise
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Add uniform noise
    noise_level = 50
    noise = np.random.uniform(-noise_level, noise_level, result.shape)
    result = result.astype(np.float32) + noise
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"add_uniform_noise_output.png", result)
            print(f"Saved to add_uniform_noise_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python add_uniform_noise.py <image_path>")
