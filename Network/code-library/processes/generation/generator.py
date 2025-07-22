import numpy as np
import os

# --- Configuration ---
IMG_SIZE = 128 # Must match Script 1
MODEL_FILENAME = 'generator_model.npy'

# --- Main Execution ---
def initialize_generator():
    """
    Initializes a uniform probability distribution for the image
    generator and saves it to a file.
    """
    # 1. Check if a model already exists to avoid overwriting
    if os.path.exists(MODEL_FILENAME):
        print(f"Model '{MODEL_FILENAME}' already exists. Skipping initialization.")
        return

    print("Initializing a new generator model...")

    # 2. Define the shape of our model.
    # We need a probability for each of the 256 color values, for each
    # of the 3 color channels (BGR), for each pixel (128x128).
    model_shape = (IMG_SIZE, IMG_SIZE, 3, 256)

    # 3. Create a uniform probability distribution.
    # np.ones creates an array of 1s, which we then normalize so each
    # pixel's color probabilities sum to 1.
    # This means initially, any color is equally likely.
    generator_model = np.ones(model_shape, dtype=np.float32)
    generator_model /= 256.0

    # 4. Save the initial model to a file
    np.save(MODEL_FILENAME, generator_model)
    print(f"Generator model initialized and saved to '{MODEL_FILENAME}'")

if __name__ == '__main__':
    initialize_generator()

