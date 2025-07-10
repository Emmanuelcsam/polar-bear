
import cv2
import numpy as np
from typing import Dict, Tuple

def preprocess_image(image: np.ndarray, GAUSSIAN_BLUR_KERNEL_SIZE: Tuple[int, int] = (7, 7), GAUSSIAN_BLUR_SIGMA: int = 2, BILATERAL_FILTER_D: int = 9, BILATERAL_FILTER_SIGMA_COLOR: int = 75, BILATERAL_FILTER_SIGMA_SPACE: int = 75, CLAHE_CLIP_LIMIT: float = 2.0, CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)) -> Dict[str, np.ndarray]:
    """
    Applies various preprocessing techniques to the input image:
    Converts to grayscale, Gaussian blur, Bilateral filter, CLAHE, histogram equalization.
    """
    if image is None:
        print("Input image for preprocessing is None.")
        return {}

    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image.copy()
    else:
        print(f"Unsupported image format for preprocessing: shape {image.shape}")
        return {}

    processed_images: Dict[str, np.ndarray] = {}
    processed_images['original_gray'] = gray.copy()

    try:
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA)
        processed_images['gaussian_blurred'] = blurred
    except Exception as e:
        print(f"Error during Gaussian Blur: {e}")
        processed_images['gaussian_blurred'] = gray.copy()

    try:
        bilateral = cv2.bilateralFilter(
            gray,
            BILATERAL_FILTER_D,
            BILATERAL_FILTER_SIGMA_COLOR,
            BILATERAL_FILTER_SIGMA_SPACE
        )
        processed_images['bilateral_filtered'] = bilateral
    except Exception as e:
        print(f"Error during Bilateral Filter: {e}")
        processed_images['bilateral_filtered'] = gray.copy()

    try:
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )
        clahe_enhanced = clahe.apply(processed_images.get('bilateral_filtered', gray))
        processed_images['clahe_enhanced'] = clahe_enhanced
    except Exception as e:
        print(f"Error during CLAHE: {e}")
        processed_images['clahe_enhanced'] = gray.copy()

    try:
        hist_equalized = cv2.equalizeHist(gray)
        processed_images['hist_equalized'] = hist_equalized
    except Exception as e:
        print(f"Error during Histogram Equalization: {e}")
        processed_images['hist_equalized'] = gray.copy()

    return processed_images

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python preprocess_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    processed = preprocess_image(image)
    for key, img in processed.items():
        cv2.imwrite(f"preprocessed_{key}.jpg", img)
    print("Preprocessing complete. Images saved.")
