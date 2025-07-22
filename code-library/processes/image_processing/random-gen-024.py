
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from log_message import log_message

def load_single_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Loads a single image from the given path.
    
    Args:
        image_path: The Path object pointing to the image file.
        
    Returns:
        The loaded image as a NumPy array (BGR format), or None if loading fails.
    """
    if not image_path.exists():
        log_message(f"File does not exist: {image_path}", level="ERROR")
        return None

    log_message(f"Loading image: {image_path.name}")
    try:
        image = cv2.imread(str(image_path))
        
        if image is None:
            log_message(f"Failed to load image with OpenCV: {image_path}", level="ERROR")
            return None
            
        # Handle different image formats for consistency
        if len(image.shape) == 2: # Grayscale
            log_message(f"Image '{image_path.name}' is grayscale. Converting to BGR for consistency.")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: # BGRA (with alpha)
            log_message(f"Image '{image_path.name}' has an alpha channel. Converting to BGR.")
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        log_message(f"Successfully loaded image: {image_path.name} with shape {image.shape}")
        return image
        
    except Exception as e:
        log_message(f"An unexpected error occurred while loading image {image_path}: {e}", level="ERROR")
        return None

if __name__ == '__main__':
    # Example of how to use the load_single_image function
    
    # This assumes you are running the script from the 'version10' directory
    # and the 'fiber_inspection_output' directory exists.
    base_path = Path("./fiber_inspection_output")
    
    # 1. Test with a valid image path
    valid_image_path = base_path / "ima18" / "ima18_annotated.jpg"
    print(f"--- Attempting to load a valid image: {valid_image_path} ---")
    loaded_image = load_single_image(valid_image_path)
    if loaded_image is not None:
        print(f"Success! Loaded image shape: {loaded_image.shape}, dtype: {loaded_image.dtype}")
        # You could display the image if you have a GUI environment
        # cv2.imshow("Loaded Image", loaded_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Failed to load the image.")

    # 2. Test with a non-existent image path
    invalid_image_path = base_path / "non_existent_image.jpg"
    print(f"\n--- Attempting to load a non-existent image: {invalid_image_path} ---")
    failed_image = load_single_image(invalid_image_path)
    if failed_image is None:
        print("Success! The function correctly returned None for a non-existent file.")
    else:
        print("Failure! The function should have returned None.")

    # 3. Test with a non-image file (e.g., a CSV)
    non_image_file_path = base_path / "batch_inspection_summary.csv"
    print(f"\n--- Attempting to load a non-image file: {non_image_file_path} ---")
    not_an_image = load_single_image(non_image_file_path)
    if not_an_image is None:
        print("Success! The function correctly returned None for a file that is not a valid image.")
    else:
        print("Failure! The function should have returned None.")
