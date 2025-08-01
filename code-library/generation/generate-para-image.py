# 4_generate_new_image.py
# This module generates a new image based on the learned parameters.
# It uses PyTorch for efficient tensor operations on CPU or GPU.
import torch
import json
import os
import cv2
import numpy as np
from shared_config import (
    GENERATOR_PARAMS_PATH,
    GENERATED_IMAGE_SIZE,
    DEVICE,
    DATA_DIR
)

print(f"--- Module: Generating New Image on device: {DEVICE} ---")
os.makedirs(DATA_DIR, exist_ok=True)

# Load the learned parameters; use defaults if not found
if os.path.exists(GENERATOR_PARAMS_PATH):
    with open(GENERATOR_PARAMS_PATH, 'r') as f:
        params = json.load(f)
    print(f"Loaded learned parameters: {params}")
else:
    params = {'target_mean': 128.0, 'target_std': 50.0}
    print("No learned parameters file found, using default values.")

# Generate random pixel data with a normal distribution (from torch)
noise_tensor = torch.randn(GENERATED_IMAGE_SIZE, device=DEVICE)

# Use the learned parameters to scale the random data
image_tensor = noise_tensor * params['target_std'] + params['target_mean']

# Clamp values to the valid 0-255 image range
image_tensor = torch.clamp(image_tensor, 0, 255)

# Convert the PyTorch tensor to a NumPy array for saving with OpenCV
# If the tensor is on the GPU, it must be moved to the CPU first.
image_np = image_tensor.cpu().numpy().astype(np.uint8)

# Save the generated image
output_path = os.path.join(DATA_DIR, "generated_image.png")
cv2.imwrite(output_path, image_np)

print(f"\nSuccessfully generated and saved new image to '{output_path}'")