# 1_setup_directories.py
# This script prepares the necessary directory structure for the program.
import os
from shared_config import DATA_DIR, IMAGE_INPUT_DIR

print("--- Module: Setting up project directories ---")

# Create the main data directory for storing outputs
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created directory: {DATA_DIR}")

# Create the directory for user-provided input images
if not os.path.exists(IMAGE_INPUT_DIR):
    os.makedirs(IMAGE_INPUT_DIR)
    print(f"Created directory: {IMAGE_INPUT_DIR}")
    # Create a placeholder file to guide the user
    with open(os.path.join(IMAGE_INPUT_DIR, "place_your_images_here.txt"), "w") as f:
        f.write("Place your .jpg or .png images in this folder for processing.")

print("\nSetup complete. You can now place images in the 'input_images' folder.")