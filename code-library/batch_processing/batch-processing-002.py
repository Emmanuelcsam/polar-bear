# 1_batch_processor.py
import os
import cv2
import numpy as np
import time
# Import configuration from 0_config.py
from importlib import import_module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
config = import_module('0_config')

def process_images():
    """Main function to find images and delegate tasks."""
    print(f"--- Batch Processor Started in '{config.LEARNING_MODE}' mode ---")
    image_files = [f for f in os.listdir(config.INPUT_DIR) if f.endswith(('.png', '.jpg'))]

    if not image_files:
        print(f"No images found in '{config.INPUT_DIR}'. Exiting.")
        return

    # In manual mode, only use the first image.
    if config.LEARNING_MODE == 'manual':
        image_files = image_files[:1]
        print(f"Manual mode: processing {image_files[0]}")

    # This is a conceptual link. You would run these scripts sequentially.
    # For this example, we'll just announce what would be done.
    print("\nNext steps would be to run:")
    print("2_intensity_reader.py -> To extract pixel data.")
    print("3_pattern_recognizer.py -> To analyze statistics.")
    print("And so on...")
    print(f"\nProcessing {len(image_files)} image(s): {', '.join(image_files)}")
    print("--- Batch Processor Finished ---")

if __name__ == '__main__':
    if config.LEARNING_MODE == 'auto':
        # Loop until the fixed duration is met
        while time.time() < config.END_TIME:
            process_images()
            print(f"\nLooping... Time left: {config.END_TIME - time.time():.0f}s")
            time.sleep(10) # Wait 10 seconds before next cycle
    else:
        process_images()
