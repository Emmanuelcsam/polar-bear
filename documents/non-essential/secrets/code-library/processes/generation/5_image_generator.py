# 5_image_generator.py
import numpy as np
import cv2
try:
    import torch
except ImportError:
    print("Warning: PyTorch not installed. Some functionality may be limited.")
    torch = None
# Import configuration from 0_config.py
from importlib import import_module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
config = import_module('0_config')

def generate_image(width=256, height=256):
    """Generates a new image based on the learned pixel distribution."""
    print("--- Image Generator Started ---")

    if torch is None:
        print("PyTorch not available. Cannot generate image.")
        return None

    try:
        # Load the learned model
        pixel_distribution = torch.load(config.MODEL_PATH)
        print(f"Loaded learned model from {config.MODEL_PATH}")
    except FileNotFoundError:
        print(f"Model not found at {config.MODEL_PATH}. Run learner first.")
        return None

    # Generate new pixel values by sampling from the learned distribution
    # The 'num_samples' is the total number of pixels in the new image
    num_pixels = width * height
    new_pixels = torch.multinomial(pixel_distribution, num_samples=num_pixels, replacement=True)

    # Reshape the 1D array of pixels into a 2D image
    img_array = new_pixels.numpy().reshape((height, width)).astype(np.uint8)

    output_path = os.path.join(config.OUTPUT_DIR, "generated_image.png")
    cv2.imwrite(output_path, img_array)
    print(f"--- New image generated and saved to {output_path} ---")
    return img_array

if __name__ == "__main__":
    generate_image()
