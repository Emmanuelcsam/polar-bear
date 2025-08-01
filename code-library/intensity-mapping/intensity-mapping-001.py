# 2_intensity_reader.py
import os
import cv2
import numpy as np
# Import configuration from 0_config.py
from importlib import import_module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
config = import_module('0_config')

def read_intensities():
    """Reads all images, converts to grayscale, and saves all pixel values."""
    print("--- Intensity Reader Started ---")

    # Ensure input directory exists
    if not os.path.exists(config.INPUT_DIR):
        os.makedirs(config.INPUT_DIR, exist_ok=True)
        print(f"Created input directory: {config.INPUT_DIR}")

    image_paths = [os.path.join(config.INPUT_DIR, f) for f in os.listdir(config.INPUT_DIR) if f.endswith(('.png', '.jpg'))]

    if not image_paths:
        print("No images to process.")
        return

    all_intensities = np.array([], dtype=np.uint8)

    for path in image_paths:
        # Read image in grayscale to get a single intensity value per pixel
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"Reading intensities from: {os.path.basename(path)}")
            # Flatten the 2D image array into a 1D array and append
            all_intensities = np.concatenate((all_intensities, img.flatten()))

    # Save the collected data for other scripts to use
    np.save(config.INTENSITY_DATA_PATH, all_intensities)
    print(f"--- Successfully saved {len(all_intensities)} pixel values to {config.INTENSITY_DATA_PATH} ---")
    return all_intensities

if __name__ == "__main__":
    read_intensities()
