import numpy as np
import cv2

# --- Configuration ---
IMG_SIZE = 128
MODEL_FILENAME = 'generator_model.npy'
OUTPUT_FILENAME = 'final_generated_image.png'

# --- Helper Function ---
def generate_from_model(model):
    """Generates an image by sampling from the model's probabilities."""
    print("Generating final image from trained model...")
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # For each pixel and channel, choose a color value based on its probability
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            for c in range(3):
                # We use the learned probabilities to make a weighted random choice
                probs = model[y, x, c]
                color_val = np.random.choice(256, p=probs)
                img[y, x, c] = color_val
    return img

# --- Main Execution ---
def generate_final_image():
    """
    Loads the trained model and uses it to generate and save a
    final high-quality image.
    """
    # 1. Load the final trained model
    try:
        trained_model = np.load(MODEL_FILENAME)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILENAME}' not found.")
        print("Please run scripts 2 and 3 first.")
        return

    # 2. Generate one image using the model
    final_image = generate_from_model(trained_model)

    # 3. Save the final image to a file
    cv2.imwrite(OUTPUT_FILENAME, final_image)
    print(f"Successfully saved final image as '{OUTPUT_FILENAME}'")

    # 4. Display the result
    cv2.imshow('Final Generated Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    generate_final_image()

