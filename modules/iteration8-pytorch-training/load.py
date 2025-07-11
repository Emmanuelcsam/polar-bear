import os
import numpy as np
import cv2

# --- Configuration ---
IMG_SIZE = 128

# --- Main Execution ---
def load_and_prepare_data():
    """
    Prompts user for a directory, loads images, calculates the
    average image, and saves it to a file.
    """
    # 1. Get image directory from user input
    path = input("Enter the path to the directory with your images: ")
    if not os.path.isdir(path):
        print("Error: Invalid directory path.")
        return

    images = []
    # 2. Load and resize all images in the directory
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

    # 3. Calculate the average image from the loaded batch
    # This average image will be our target for the generator
    avg_image = np.mean(images, axis=0).astype(np.uint8)

    # 4. Save the target image for the next script
    np.savez_compressed('target_data.npz', target=avg_image)

    # Display the target image for verification
    cv2.imshow('Target Image (Press any key to continue)', avg_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Target image has been calculated and saved to 'target_data.npz'")

if __name__ == '__main__':
    load_and_prepare_data()

