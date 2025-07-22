# 7_geometry_recognizer.py
import os
import cv2
import numpy as np
# Import configuration from 0_config.py
from importlib import import_module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
config = import_module('0_config')

def find_geometry():
    """Applies Canny edge detection to find geometric outlines in an image."""
    print("--- Geometry Recognizer Started ---")

    # Ensure input directory exists
    if not os.path.exists(config.INPUT_DIR):
        os.makedirs(config.INPUT_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(config.INPUT_DIR) if f.endswith(('.png', '.jpg'))]
    if not image_files:
        print(f"No image found in {config.INPUT_DIR}. Exiting.")
        return None

    # Use the first image for demonstration
    target_image_path = os.path.join(config.INPUT_DIR, image_files[0])
    print(f"Processing {target_image_path} for geometric patterns.")

    img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {target_image_path}")
        return None

    # Blur the image slightly to reduce noise before edge detection
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny edge detection finds sharp changes in intensity (lines, shapes)
    # The two thresholds determine how sensitive the detection is.
    edges = cv2.Canny(image=img_blurred, threshold1=50, threshold2=150)

    output_path = os.path.join(config.OUTPUT_DIR, "geometric_patterns.png")
    cv2.imwrite(output_path, edges)
    print(f"--- Edge map saved to {output_path} ---")
    return edges

if __name__ == "__main__":
    find_geometry()
