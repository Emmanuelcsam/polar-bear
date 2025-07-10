from typing import Dict
import numpy as np
import cv2

from utils import _log_message, _log_duration

def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
    """Applies various preprocessing techniques to the input image."""
    preprocess_start_time = self._start_timer()
    _log_message("Starting image preprocessing...")
    if image is None:
        _log_message("Input image for preprocessing is None.", level="ERROR")
        return {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 and image.shape[2] == 3 else image.copy()
    processed_images: Dict[str, np.ndarray] = {'original_gray': gray.copy()}
    try:
        processed_images['gaussian_blurred'] = cv2.GaussianBlur(gray, self.config.GAUSSIAN_BLUR_KERNEL_SIZE, self.config.GAUSSIAN_BLUR_SIGMA)
    except Exception as e:
        _log_message(f"Error during Gaussian Blur: {e}", level="WARNING")
        processed_images['gaussian_blurred'] = gray.copy()
    try:
        processed_images['bilateral_filtered'] = cv2.bilateralFilter(gray, self.config.BILATERAL_FILTER_D, self.config.BILATERAL_FILTER_SIGMA_COLOR, self.config.BILATERAL_FILTER_SIGMA_SPACE)
    except Exception as e:
        _log_message(f"Error during Bilateral Filter: {e}", level="WARNING")
        processed_images['bilateral_filtered'] = gray.copy()
    try:
        clahe = cv2.createCLAHE(clipLimit=self.config.CLAHE_CLIP_LIMIT, tileGridSize=self.config.CLAHE_TILE_GRID_SIZE)
        processed_images['clahe_enhanced'] = clahe.apply(processed_images.get('bilateral_filtered', gray))
    except Exception as e:
        _log_message(f"Error during CLAHE: {e}", level="WARNING")
        processed_images['clahe_enhanced'] = gray.copy()
    try:
        processed_images['hist_equalized'] = cv2.equalizeHist(gray)
    except Exception as e:
        _log_message(f"Error during Histogram Equalization: {e}", level="WARNING")
        processed_images['hist_equalized'] = gray.copy()
    _log_duration("Image Preprocessing", preprocess_start_time, self.current_image_result)
    return processed_images
