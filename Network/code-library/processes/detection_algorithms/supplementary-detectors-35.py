
from typing import Optional
import cv2
import numpy as np

from neural_framework.core.logger import log
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
    log.info(f"Starting Canny defect detection for zone '{zone_name}'...")
    # Defaulting to a common Canny threshold range
    canny = cv2.Canny(image, 50, 150)
    canny = cv2.bitwise_and(canny, canny, mask=zone_mask)
    log.info(f"Canny detection for '{zone_name}' complete.")
    return canny

def adaptive_threshold_detection(image: np.ndarray, zone_mask: np.ndarray, config: object) -> np.ndarray:
    """Adaptive threshold detection for defects"""
    log.info(f"Starting Adaptive Threshold defect detection for zone '{zone_name}'...")
    
    block_size = getattr(config, 'ADAPTIVE_THRESH_BLOCK_SIZE', 21)
    c_value = getattr(config, 'ADAPTIVE_THRESH_C_VALUE', 5)

    if not isinstance(block_size, int) or block_size <= 1 or block_size % 2 == 0:
        log.warning(f"Config ADAPTIVE_THRESH_BLOCK_SIZE invalid, using {block_size}")
        block_size = 21

    adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, c_value)
    adaptive = cv2.bitwise_and(adaptive, adaptive, mask=zone_mask)
    log.info(f"Adaptive Threshold detection for '{zone_name}' complete.")
    return adaptive

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
