
import cv2
import numpy as np
from typing import Tuple

def preprocess_image_test3(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the image for defect detection (from test3.py)
    Returns: (grayscale, denoised)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Denoise using Gaussian blur
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return gray, denoised

if __name__ == '__main__':
    # Create a dummy color image
    sz = 400
    dummy_image = np.zeros((sz, sz, 3), dtype=np.uint8)
    cv2.circle(dummy_image, (sz//2, sz//2), 150, (128, 50, 200), -1)
    
    # Run the preprocessing function
    gray_result, denoised_result = preprocess_image_test3(dummy_image)
    
    # Display results
    cv2.imshow('Original Dummy Image', dummy_image)
    cv2.imshow('Grayscale Result', gray_result)
    cv2.imshow('Denoised Result', denoised_result)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
