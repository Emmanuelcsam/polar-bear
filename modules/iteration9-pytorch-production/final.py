import torch
import numpy as np
import cv2
from generative_script_pytorch_2 import PixelGenerator # Import model class

# --- Configuration ---
IMG_SIZE = 128
MODEL_FILENAME = 'generator_model.pth'
OUTPUT_FILENAME = 'final_generated_image.png'

# --- Helper Function ---
def generate_final_image(model):
    """Generates one image by sampling from the model's probabilities."""
    print("Generating final image from trained model...")
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # No need to track gradients here
        # Get the learned probability distributions
        probs = model.get_probs()
        # Create a categorical distribution to sample from
        dist = torch.distributions.Categorical(probs)
        # Sample one value for each pixel's distribution
        generated_tensor = dist.sample()
        # Convert to a NumPy array suitable for OpenCV (uint8)
        img_np = generated_tensor.byte().cpu().numpy()
    return img_np

# --- Main Execution ---
def main():
    """
    Loads the trained model and uses it to generate and save a
    final high-quality image.
    """
    # 1. Load the model structure and the trained parameters
    try:
        model = PixelGenerator(IMG_SIZE)
        model.load_state_dict(torch.load(MODEL_FILENAME))
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILENAME}' not found.")
        print("Please run scripts 2 and 3 first.")
        return

    # 2. Generate one image using the trained model
    final_image = generate_final_image(model)

    # 3. Save the final image to a file
    cv2.imwrite(OUTPUT_FILENAME, final_image)
    print(f"Successfully saved final image as '{OUTPUT_FILENAME}'")

    # 4. Display the result
    cv2.imshow('Final Generated Image (PyTorch)', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

