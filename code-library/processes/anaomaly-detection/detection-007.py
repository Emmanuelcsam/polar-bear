# 2_analyze_images.py
# This module reads all images from a folder, analyzes their pixel intensities,
# and saves the findings to a shared data file.
import cv2
import numpy as np
import os
import json
from shared_config import IMAGE_INPUT_DIR, ANALYSIS_RESULTS_PATH, DATA_DIR

print("--- Module: Starting Image Analysis (Batch Processing) ---")
os.makedirs(DATA_DIR, exist_ok=True) # Ensure data directory exists

analysis_data = {}
# Loop through all files in the input directory
image_files = [f for f in os.listdir(IMAGE_INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("No images found in 'input_images' directory. Nothing to analyze.")
    exit()

for filename in image_files:
    try:
        path = os.path.join(IMAGE_INPUT_DIR, filename)
        # Read image in grayscale to focus on intensity
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read '{filename}', skipping.")
            continue

        # Calculate basic statistics (the "intensity-reader" function)
        mean_intensity = np.mean(image)
        std_dev_intensity = np.std(image)

        # Store the results for this image
        analysis_data[filename] = {
            'mean_intensity': mean_intensity,
            'std_dev_intensity': std_dev_intensity
        }
        print(f"Analyzed '{filename}': Mean={mean_intensity:.2f}, StdDev={std_dev_intensity:.2f}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save all analysis results to a single JSON file for other modules to use
with open(ANALYSIS_RESULTS_PATH, 'w') as f:
    json.dump(analysis_data, f, indent=4)

print(f"\nAnalysis complete. Results saved to '{ANALYSIS_RESULTS_PATH}'")