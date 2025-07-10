
from typing import Optional, Dict
import cv2
import numpy as np

from log_message import log_message
from inspector_config import InspectorConfig
from load_single_image import load_single_image
from preprocess_image import preprocess_image
from create_zone_masks import create_zone_masks
from pathlib import Path

def detect_region_defects_do2mr(
    image_gray: np.ndarray, 
    zone_mask: np.ndarray, 
    config: InspectorConfig,
    zone_name: str
) -> Optional[np.ndarray]:
    """
    Detects region-based defects using a DO2MR-inspired method.
    
    Args:
        image_gray: Grayscale image to inspect.
        zone_mask: Binary mask for the current zone.
        config: An InspectorConfig object with DO2MR parameters.
        zone_name: Name of the zone for logging.
        
    Returns:
        A binary mask of detected region defects.
    """
    log_message(f"Starting DO2MR region defect detection for zone '{zone_name}'...")
    if image_gray is None or zone_mask is None or np.sum(zone_mask) == 0:
        log_message("Input image or mask is None or empty for DO2MR.", level="WARNING")
        return np.zeros_like(image_gray, dtype=np.uint8)

    masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)
    vote_map = np.zeros_like(image_gray, dtype=np.float32)

    for kernel_size in config.DO2MR_KERNEL_SIZES:
        struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        min_filtered = cv2.erode(masked_image, struct_element)
        max_filtered = cv2.dilate(masked_image, struct_element)
        residual = cv2.subtract(max_filtered, min_filtered)
        
        blur_ksize = config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE
        res_blurred = cv2.medianBlur(residual, blur_ksize) if blur_ksize > 0 else residual

        for gamma in config.DO2MR_GAMMA_VALUES:
            masked_res_vals = res_blurred[zone_mask > 0]
            if masked_res_vals.size == 0: continue
            
            mean_val, std_val = np.mean(masked_res_vals), np.std(masked_res_vals)
            thresh_val = np.clip(mean_val + gamma * std_val, 0, 255)
            
            _, defect_mask_pass = cv2.threshold(res_blurred, thresh_val, 255, cv2.THRESH_BINARY)
            defect_mask_pass = cv2.bitwise_and(defect_mask_pass, defect_mask_pass, mask=zone_mask)
            
            open_k = config.DO2MR_MORPH_OPEN_KERNEL_SIZE
            if open_k[0] > 0 and open_k[1] > 0:
                open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_k)
                defect_mask_pass = cv2.morphologyEx(defect_mask_pass, cv2.MORPH_OPEN, open_kernel)
                
            vote_map += (defect_mask_pass / 255.0)

    num_param_sets = len(config.DO2MR_KERNEL_SIZES) * len(config.DO2MR_GAMMA_VALUES)
    min_votes = max(1, int(num_param_sets * 0.3)) # Require 30% of votes
    combined_map = np.where(vote_map >= min_votes, 255, 0).astype(np.uint8)
    
    log_message(f"DO2MR detection for '{zone_name}' complete. Found {np.count_nonzero(combined_map)} defect pixels.")
    return combined_map

if __name__ == '__main__':
    # Example of how to use the detect_region_defects_do2mr function
    
    # 1. Setup: Load config, image, preprocess, and create zone masks
    conf = InspectorConfig()
    image_path = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    bgr_image = load_single_image(image_path)
    
    if bgr_image is not None:
        preprocessed = preprocess_image(bgr_image, conf)
        
        # Use a representative image for detection
        detection_image = preprocessed.get('clahe_enhanced', preprocessed['original_gray'])
        
        # Create mock zone masks as in the previous example
        h, w = bgr_image.shape[:2]
        center, radius = (w//2, h//2), 150.0
        zones = create_zone_masks(bgr_image.shape[:2], center, radius, conf.DEFAULT_ZONES)
        
        # 2. Run DO2MR on a specific zone, e.g., the "cladding"
        target_zone_name = "cladding"
        cladding_mask = zones.get(target_zone_name)
        
        print(f"\n--- Running DO2MR on the '{target_zone_name}' zone ---")
        if cladding_mask and cladding_mask.mask is not None:
            do2mr_defect_map = detect_region_defects_do2mr(
                image_gray=detection_image,
                zone_mask=cladding_mask.mask,
                config=conf,
                zone_name=target_zone_name
            )
            
            if do2mr_defect_map is not None:
                # 3. Visualize the results
                # Create an image showing the original zone and the detected defects
                zone_viz = cv2.bitwise_and(bgr_image, bgr_image, mask=cladding_mask.mask)
                # Color defects in red
                defects_colored = cv2.merge([
                    np.zeros_like(do2mr_defect_map), 
                    np.zeros_like(do2mr_defect_map), 
                    do2mr_defect_map
                ])
                
                # Add defects to the visualization
                output_image = cv2.add(zone_viz, defects_colored)
                
                output_filename = f"modularized_scripts/z_test_output_do2mr_{target_zone_name}.png"
                cv2.imwrite(output_filename, output_image)
                print(f"Success! Saved DO2MR defect map visualization to '{output_filename}'")
            else:
                print("DO2MR detection failed to return a map.")
        else:
            print(f"Could not find the '{target_zone_name}' zone mask to run the example.")
    else:
        print(f"Could not load image at {image_path} to run the example.")
