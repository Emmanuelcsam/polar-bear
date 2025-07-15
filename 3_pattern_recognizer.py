# 3_pattern_recognizer.py
import numpy as np
# Import configuration from 0_config.py
from importlib import import_module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
config = import_module('0_config')

def recognize_patterns():
    """Loads intensity data and computes statistical patterns."""
    print("--- Pattern Recognizer Started ---")
    try:
        # Load the data created by the intensity reader
        intensities = np.load(config.INTENSITY_DATA_PATH)
        print(f"Loaded {len(intensities)} data points.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {config.INTENSITY_DATA_PATH}")
        print("Please run '2_intensity_reader.py' first.")
        return None

    if intensities.size == 0:
        print("Data file is empty. No patterns to recognize.")
        return None

    # Calculate basic statistics to identify trends
    mean_val = np.mean(intensities)
    median_val = np.median(intensities)
    std_dev = np.std(intensities)
    # Find the most common pixel value (the mode)
    mode_val = np.bincount(intensities).argmax()

    results = {
        'mean': mean_val,
        'median': median_val,
        'std_dev': std_dev,
        'mode': mode_val
    }

    print("\n--- Intensity Patterns Recognized ---")
    print(f"Average Pixel Intensity (Mean):   {mean_val:.2f}")
    print(f"Median Pixel Intensity:           {median_val:.2f}")
    print(f"Standard Deviation:               {std_dev:.2f}")
    print(f"Most Frequent Intensity (Mode):   {mode_val}")
    print("--- Pattern Recognizer Finished ---")

    return results

if __name__ == "__main__":
    recognize_patterns()
