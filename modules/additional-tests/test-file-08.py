from typing import Dict
import cv2
import numpy as np

# Assuming these are in separate files
from log_message import log_message
from inspector_config import InspectorConfig
from load_single_image import load_single_image
from pathlib import Path

def preprocess_image(image: np.ndarray, config: InspectorConfig) -> Dict[str, np.ndarray]:
    """
    Applies various preprocessing techniques to the input image.
    
    Args:
        image: The input BGR image.
        config: An InspectorConfig object with preprocessing parameters.
        
    Returns:
        A dictionary of preprocessed images (grayscale).
    """
    log_message("Starting image preprocessing...")
    if image is None:
        log_message("Input image for preprocessing is None.", level="ERROR")
        return {}

    # Convert to grayscale if color, else work on a copy.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
    
    processed_images: Dict[str, np.ndarray] = {'original_gray': gray.copy()}

    # Apply Gaussian Blur
    try:
        processed_images['gaussian_blurred'] = cv2.GaussianBlur(
            gray, config.GAUSSIAN_BLUR_KERNEL_SIZE, config.GAUSSIAN_BLUR_SIGMA
        )
    except Exception as e:
        log_message(f"Error during Gaussian Blur: {e}", level="WARNING")
        processed_images['gaussian_blurred'] = gray.copy()

    # Apply Bilateral Filter
    try:
        processed_images['bilateral_filtered'] = cv2.bilateralFilter(
            gray, config.BILATERAL_FILTER_D, config.BILATERAL_FILTER_SIGMA_COLOR, config.BILATERAL_FILTER_SIGMA_SPACE
        )
    except Exception as e:
        log_message(f"Error during Bilateral Filter: {e}", level="WARNING")
        processed_images['bilateral_filtered'] = gray.copy()

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    try:
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE)
        # Apply to a smoothed image for better results
        base_for_clahe = processed_images.get('bilateral_filtered', gray)
        processed_images['clahe_enhanced'] = clahe.apply(base_for_clahe)
    except Exception as e:
        log_message(f"Error during CLAHE: {e}", level="WARNING")
        processed_images['clahe_enhanced'] = gray.copy()

    # Apply standard Histogram Equalization
    try:
        processed_images['hist_equalized'] = cv2.equalizeHist(gray)
    except Exception as e:
        log_message(f"Error during Histogram Equalization: {e}", level="WARNING")
        processed_images['hist_equalized'] = gray.copy()
        
    log_message("Image preprocessing complete.")
    return processed_images

if __name__ == '__main__':
    # Example of how to use the preprocess_image function
    
    # 1. Setup: Load a config and an image
    conf = InspectorConfig()
    
    # Use a real image from the project's output for a realistic test
    # This path assumes you run from the 'version10' directory
    image_path = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    
    print(f"--- Loading image for preprocessing: {image_path} ---")
    bgr_image = load_single_image(image_path)
    
    if bgr_image is not None:
        # 2. Run the preprocessing function
        preprocessed_bundle = preprocess_image(bgr_image, conf)
        
        # 3. Verify the output
        print("\n--- Preprocessing Results ---")
        if preprocessed_bundle:
            print(f"Successfully generated {len(preprocessed_bundle)} processed images.")
            for name, img in preprocessed_bundle.items():
                print(f" - '{name}': shape={img.shape}, dtype={img.dtype}")
                # Optionally, save the images to disk to inspect them visually
                output_filename = f"modularized_scripts/z_test_output_{name}.png"
                cv2.imwrite(output_filename, img)
                print(f"   Saved to {output_filename}")
            print("\nCheck the 'modularized_scripts' directory for the output images.")
        else:
            print("Preprocessing failed to produce results.")
    else:
        print(f"Could not load the image at {image_path}. Cannot run preprocessing example.")
