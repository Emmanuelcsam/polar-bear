# 4_generative_learner.py
import numpy as np
import torch
# Import configuration from 0_config.py
from importlib import import_module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
config = import_module('0_config')

def learn_pixel_distribution():
    """'Learns' the probability distribution of pixel intensities from data."""
    print("--- Generative Learner Started ---")
    try:
        # Load the raw pixel data from the intensity reader
        intensities = np.load(config.INTENSITY_DATA_PATH)
    except FileNotFoundError:
        print(f"Data file not found at {config.INTENSITY_DATA_PATH}. Run reader first.")
        return None

    if intensities.size == 0:
        print("Data file is empty. No distribution to learn.")
        return None

    # Calculate the frequency of each pixel value (0-255)
    # This frequency distribution is our 'learned model'
    counts = torch.bincount(torch.from_numpy(intensities), minlength=256)

    # Normalize counts to get probabilities (sum to 1)
    distribution = counts.float() / counts.sum()

    # Save the learned distribution for the generator script
    torch.save(distribution, config.MODEL_PATH)
    print(f"--- Learned pixel distribution and saved model to {config.MODEL_PATH} ---")
    return distribution

if __name__ == "__main__":
    learn_pixel_distribution()
