# 10_hpc_parallel_cpu.py
import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import time
# Import configuration from 0_config.py
from importlib import import_module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
config = import_module('0_config')

def get_image_mean(image_path):
    """A simple function that reads an image and calculates its mean intensity."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (os.path.basename(image_path), None)
        mean_val = np.mean(img)
        print(f"Processed {os.path.basename(image_path)}, Mean: {mean_val:.2f}")
        return (os.path.basename(image_path), mean_val)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return (os.path.basename(image_path), None)

def run_parallel_processing():
    """Uses a multiprocessing Pool to process all images in parallel."""
    print("--- HPC CPU Parallel Processing Started ---")

    # Ensure input directory exists
    if not os.path.exists(config.INPUT_DIR):
        os.makedirs(config.INPUT_DIR, exist_ok=True)

    image_paths = [os.path.join(config.INPUT_DIR, f) for f in os.listdir(config.INPUT_DIR) if f.endswith(('.png', '.jpg'))]
    if not image_paths:
        print("No images to process.")
        return []

    # Use all available CPU cores
    num_cores = cpu_count()
    print(f"Starting parallel processing with {num_cores} CPU cores.")

    start_time = time.time()

    # A Pool distributes tasks to worker processes automatically.
    # The map function applies 'get_image_mean' to every item in 'image_paths'.
    try:
        with Pool(processes=num_cores) as pool:
            results = pool.map(get_image_mean, image_paths)
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        return []

    duration = time.time() - start_time
    print(f"\n--- Finished in {duration:.4f} seconds ---")
    print("Results:")
    for name, mean in results:
        if mean is not None:
            print(f"  - {name}: {mean:.2f}")
        else:
            print(f"  - {name}: Error processing")

    return results

if __name__ == "__main__":
    run_parallel_processing()
