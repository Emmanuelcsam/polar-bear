
from typing import Optional
import cv2
import numpy as np

from log_message import log_message
from inspector_config import InspectorConfig
from load_single_image import load_single_image
from preprocess_image import preprocess_image
from create_zone_masks import create_zone_masks
from pathlib import Path

def detect_scratches_lei(
    image_gray: np.ndarray, 
    zone_mask: np.ndarray, 
    config: InspectorConfig,
    zone_name: str
) -> Optional[np.ndarray]:
    """
    Detects linear scratches using an LEI-inspired method.
    
    Args:
        image_gray: Grayscale image to inspect.
        zone_mask: Binary mask for the current zone.
        config: An InspectorConfig object with LEI parameters.
        zone_name: Name of the zone for logging.
        
    Returns:
        A binary mask of detected scratches.
    """
    log_message(f"Starting LEI scratch detection for zone '{zone_name}'...")
    if image_gray is None or zone_mask is None or np.sum(zone_mask) == 0:
        log_message("Input image or mask is None or empty for LEI.", level="WARNING")
        return np.zeros_like(image_gray, dtype=np.uint8)

    masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)
    # Enhance contrast specifically for scratch detection
    enhanced_image = cv2.equalizeHist(masked_image)
    enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=zone_mask)
    
    max_response_map = np.zeros_like(enhanced_image, dtype=np.float32)

    for kernel_length in config.LEI_KERNEL_LENGTHS:
        for angle_deg in range(0, 180, config.LEI_ANGLE_STEP):
            # Create a rotated linear kernel
            line_kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
            rot_matrix = cv2.getRotationMatrix2D((kernel_length // 2, 0), angle_deg, 1.0)
            bbox_size = kernel_length 
            if bbox_size <= 0: continue
            
            rotated_kernel = cv2.warpAffine(line_kernel_base, rot_matrix, (bbox_size, bbox_size))
            
            kernel_sum = np.sum(rotated_kernel)
            if kernel_sum < 1e-6: continue
            
            normalized_kernel = rotated_kernel.astype(np.float32) / kernel_sum
            response = cv2.filter2D(enhanced_image.astype(np.float32), -1, normalized_kernel)
            max_response_map = np.maximum(max_response_map, response)

    if np.max(max_response_map) > 0:
        cv2.normalize(max_response_map, max_response_map, 0, 255, cv2.NORM_MINMAX)
    
    response_8u = max_response_map.astype(np.uint8)
    
    _, scratch_mask = cv2.threshold(response_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use a general closing kernel as scratch orientations vary
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, close_kernel)
    scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=zone_mask)
    
    log_message(f"LEI detection for '{zone_name}' complete. Found {np.count_nonzero(scratch_mask)} defect pixels.")
    return scratch_mask

if __name__ == '__main__':
    # Example of how to use the detect_scratches_lei function
    
    # 1. Setup
    conf = InspectorConfig()
    image_path = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    bgr_image = load_single_image(image_path)
    
    if bgr_image is not None:
        preprocessed = preprocess_image(bgr_image, conf)
        detection_image = preprocessed.get('clahe_enhanced', preprocessed['original_gray'])
        
        h, w = bgr_image.shape[:2]
        center, radius = (w//2, h//2), 150.0
        zones = create_zone_masks(bgr_image.shape[:2], center, radius, conf.DEFAULT_ZONES)
        
        # 2. Run LEI on the "cladding" zone
        target_zone_name = "cladding"
        cladding_mask = zones.get(target_zone_name)
        
        print(f"\n--- Running LEI on the '{target_zone_name}' zone ---")
        if cladding_mask and cladding_mask.mask is not None:
            lei_defect_map = detect_scratches_lei(
                image_gray=detection_image,
                zone_mask=cladding_mask.mask,
                config=conf,
                zone_name=target_zone_name
            )
            
            if lei_defect_map is not None:
                # 3. Visualize the results
                zone_viz = cv2.bitwise_and(bgr_image, bgr_image, mask=cladding_mask.mask)
                # Color defects in magenta
                defects_colored = cv2.merge([lei_defect_map, np.zeros_like(lei_defect_map), lei_defect_map])
                
                output_image = cv2.add(zone_viz, defects_colored)
                
                output_filename = f"modularized_scripts/z_test_output_lei_{target_zone_name}.png"
                cv2.imwrite(output_filename, output_image)
                print(f"Success! Saved LEI defect map visualization to '{output_filename}'")
            else:
                print("LEI detection failed to return a map.")
        else:
            print(f"Could not find the '{target_zone_name}' zone mask to run the example.")
    else:
        print(f"Could not load image at {image_path} to run the example.")
