
from typing import Optional
import cv2
import numpy as np

from log_message import log_message
from inspector_config import InspectorConfig
from load_single_image import load_single_image
from preprocess_image import preprocess_image
from create_zone_masks import create_zone_masks
from pathlib import Path

def detect_defects_canny(
    image_gray: np.ndarray, 
    zone_mask: np.ndarray, 
    config: InspectorConfig,
    zone_name: str
) -> Optional[np.ndarray]:
    """Detects defects using Canny edge detection followed by morphological operations."""
    log_message(f"Starting Canny defect detection for zone '{zone_name}'...")
    blurred_for_canny = cv2.GaussianBlur(image_gray, (3, 3), 0)
    edges = cv2.Canny(blurred_for_canny, config.CANNY_LOW_THRESHOLD, config.CANNY_HIGH_THRESHOLD)
    edges_masked = cv2.bitwise_and(edges, edges, mask=zone_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.morphologyEx(edges_masked, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    log_message(f"Canny detection for '{zone_name}' complete.")
    return closed_edges

def detect_defects_adaptive_thresh(
    image_gray: np.ndarray, 
    zone_mask: np.ndarray, 
    config: InspectorConfig,
    zone_name: str
) -> Optional[np.ndarray]:
    """Detects defects using adaptive thresholding."""
    log_message(f"Starting Adaptive Threshold defect detection for zone '{zone_name}'...")
    
    block_size = config.ADAPTIVE_THRESH_BLOCK_SIZE
    if block_size <= 1 or block_size % 2 == 0:
        block_size = 11 # Fallback to a valid default
        log_message(f"Config ADAPTIVE_THRESH_BLOCK_SIZE invalid, using {block_size}", level="WARNING")

    adaptive_thresh_mask = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, block_size, config.ADAPTIVE_THRESH_C
    )
    defects_masked = cv2.bitwise_and(adaptive_thresh_mask, adaptive_thresh_mask, mask=zone_mask)
    
    log_message(f"Adaptive Threshold detection for '{zone_name}' complete.")
    return defects_masked

if __name__ == '__main__':
    # Example of how to use the additional detector functions
    
    # 1. Setup
    conf = InspectorConfig()
    image_path = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    bgr_image = load_single_image(image_path)
    
    if bgr_image is not None:
        preprocessed = preprocess_image(bgr_image, conf)
        # Use a smoothed image for these detectors
        detection_image = preprocessed.get('bilateral_filtered', preprocessed['original_gray'])
        
        h, w = bgr_image.shape[:2]
        center, radius = (w//2, h//2), 150.0
        zones = create_zone_masks(bgr_image.shape[:2], center, radius, conf.DEFAULT_ZONES)
        
        target_zone_name = "cladding"
        cladding_mask = zones.get(target_zone_name)
        
        if cladding_mask and cladding_mask.mask is not None:
            # 2. Run Canny detector
            print(f"\n--- Running Canny Detector on the '{target_zone_name}' zone ---")
            canny_map = detect_defects_canny(detection_image, cladding_mask.mask, conf, target_zone_name)
            if canny_map is not None:
                output_filename_canny = f"modularized_scripts/z_test_output_canny_{target_zone_name}.png"
                cv2.imwrite(output_filename_canny, canny_map)
                print(f"Saved Canny map to '{output_filename_canny}'")

            # 3. Run Adaptive Threshold detector
            print(f"\n--- Running Adaptive Threshold on the '{target_zone_name}' zone ---")
            adaptive_map = detect_defects_adaptive_thresh(detection_image, cladding_mask.mask, conf, target_zone_name)
            if adaptive_map is not None:
                output_filename_adaptive = f"modularized_scripts/z_test_output_adaptive_{target_zone_name}.png"
                cv2.imwrite(output_filename_adaptive, adaptive_map)
                print(f"Saved Adaptive Threshold map to '{output_filename_adaptive}'")
        else:
            print(f"Could not find the '{target_zone_name}' zone mask to run the example.")
    else:
        print(f"Could not load image at {image_path} to run the example.")
