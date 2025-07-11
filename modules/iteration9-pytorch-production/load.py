import os
import numpy as np
import cv2
import torch

# --- Configuration ---
IMG_SIZE = 128
OUTPUT_FILE = 'target_data.pt'

# --- Main Execution ---
def load_and_prepare_data():
    """
    Prompts for a directory, loads images, calculates the average
    image, and saves it as a PyTorch tensor.
    """
    path = input("Enter the path to the directory with your images: ")
    if not os.path.isdir(path):
        print("Error: Invalid directory path.")
        return

    images = []
    for filename in os.listdir(path):
        try:
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
        except Exception as e:
            print(f"Could not read {filename}: {e}")

    if not images:
        print("No images found or loaded. Exiting.")
        return

    # Calculate the average image with NumPy first
    avg_image_np = np.mean(images, axis=0).astype(np.uint8)

    # Convert the final target image to a PyTorch tensor
    # We use .long() because these are class indices (0-255) for the loss function
    target_tensor = torch.from_numpy(avg_image_np).long()

    # Save the target tensor for the training script
    torch.save(target_tensor, OUTPUT_FILE)

    # Display the target image for verification
    cv2.imshow('Target Image (Press any key to continue)', avg_image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Target tensor saved to '{OUTPUT_FILE}'")

if __name__ == '__main__':
    load_and_prepare_data()

