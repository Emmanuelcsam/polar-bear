#!/usr/bin/env python3
"""
Add Speckle Noise
Category: noise
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Add Speckle Noise
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Add multiplicative speckle noise
    noise = np.random.randn(*result.shape) * 0.1
    result = result + result * noise
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"add_speckle_noise_output.png", result)
            print(f"Saved to add_speckle_noise_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python add_speckle_noise.py <image_path>")
