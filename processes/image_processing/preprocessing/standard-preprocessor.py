
import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, and applies Gaussian blur for denoising.
    This corresponds to Stage 1 of the inspection pipeline. 

    Args:
        image_path (str): The file path to the fiber optic end face image.

    Returns:
        tuple: A tuple containing the original color image and the preprocessed grayscale image.
    """
    # Load Image: Read the image file.
    original_image = cv2.imread(image_path)
    if original_image is None:
        # Create a dummy image if not found
        print(f"Image not found at path: {image_path}. Creating a dummy image.")
        sz = 600
        original_image = np.full((sz, sz, 3), (200, 200, 200), dtype=np.uint8)
        cv2.circle(original_image, (sz//2, sz//2), 200, (150, 150, 150), -1)
        cv2.imwrite(image_path, original_image)

    # Convert to Grayscale: Operations work best on single-channel images.
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Denoise: Apply a Gaussian blur to reduce noise from image acquisition. 
    # A 5x5 kernel is a common choice for moderate smoothing.
    preprocessed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    print("Stage 1: Image Pre-processing Complete.")
    return original_image, preprocessed_image

if __name__ == '__main__':
    IMAGE_PATH = 'dummy_fiber_image.png'
    
    # Run the preprocessing function
    original, preprocessed = preprocess_image(IMAGE_PATH)
    
    # Display the results
    cv2.imshow('Original Image', original)
    cv2.imshow('Preprocessed (Grayscale & Blurred)', preprocessed)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
