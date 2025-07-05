#!/usr/bin/env python3
"""
Add Salt and Pepper Noise
Category: noise
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Add Salt and Pepper Noise
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Add salt and pepper noise
    noise_ratio = 0.05
    total_pixels = result.size
    num_salt = int(total_pixels * noise_ratio / 2)
    num_pepper = int(total_pixels * noise_ratio / 2)
    # Add salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in result.shape]
    result[coords[0], coords[1]] = 255
    # Add pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in result.shape]
    result[coords[0], coords[1]] = 0
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"add_salt_pepper_output.png", result)
            print(f"Saved to add_salt_pepper_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python add_salt_pepper.py <image_path>")
