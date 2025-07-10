
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from log_message import log_message
from inspector_config import InspectorConfig
from load_single_image import load_single_image
from preprocess_image import preprocess_image
from create_zone_masks import create_zone_masks
from detect_region_defects_do2mr import detect_region_defects_do2mr
from detect_scratches_lei import detect_scratches_lei
from additional_detectors import detect_defects_canny, detect_defects_adaptive_thresh
from pathlib import Path

def combine_defect_masks(
    defect_maps: Dict[str, Optional[np.ndarray]], 
    image_shape: Tuple[int, int],
    config: InspectorConfig
) -> np.ndarray:
    """
    Combines defect masks from multiple methods using a weighted voting scheme.
    
    Args:
        defect_maps: Dictionary of method names to binary defect masks.
        image_shape: Tuple (height, width) for the output map.
        config: An InspectorConfig object with confidence weights.
        
    Returns:
        A single binary mask representing confirmed defects.
    """
    log_message("Combining defect masks from multiple methods...")
    h, w = image_shape
    vote_map = np.zeros((h, w), dtype=np.float32)

    for method_name, mask in defect_maps.items():
        if mask is not None:
            base_method_key = method_name.split('_')[0]
            weight = config.CONFIDENCE_WEIGHTS.get(base_method_key, 0.5)
            vote_map[mask == 255] += weight
            
    # A defect is confirmed if its weighted vote meets the threshold
    confirmation_threshold = float(config.MIN_METHODS_FOR_CONFIRMED_DEFECT)
    combined_mask = np.where(vote_map >= confirmation_threshold, 255, 0).astype(np.uint8)
    
    log_message(f"Mask combination complete. Final mask has {np.count_nonzero(combined_mask)} defect pixels.")
    return combined_mask

if __name__ == '__main__':
    # Example of how to use the combine_defect_masks function
    
    # 1. Setup: Run all detectors to get their individual maps
    conf = InspectorConfig()
    image_path = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    bgr_image = load_single_image(image_path)
    
    if bgr_image is not None:
        preprocessed = preprocess_image(bgr_image, conf)
        detection_image = preprocessed.get('clahe_enhanced', preprocessed['original_gray'])
        
        h, w = bgr_image.shape[:2]
        center, radius = (w//2, h//2), 150.0
        zones = create_zone_masks(bgr_image.shape[:2], center, radius, conf.DEFAULT_ZONES)
        
        target_zone_name = "cladding"
        cladding_mask = zones.get(target_zone_name)
        
        if cladding_mask and cladding_mask.mask is not None:
            print("\n--- Running all detectors to generate input maps ---")
            
            all_maps: Dict[str, Optional[np.ndarray]] = {}
            all_maps['do2mr'] = detect_region_defects_do2mr(detection_image, cladding_mask.mask, conf, target_zone_name)
            all_maps['lei'] = detect_scratches_lei(detection_image, cladding_mask.mask, conf, target_zone_name)
            all_maps['canny'] = detect_defects_canny(detection_image, cladding_mask.mask, conf, target_zone_name)
            all_maps['adaptive_thresh'] = detect_defects_adaptive_thresh(detection_image, cladding_mask.mask, conf, target_zone_name)
            
            # 2. Run the combination function
            print("\n--- Combining defect masks ---")
            final_mask = combine_defect_masks(all_maps, (h, w), conf)
            
            # 3. Visualize the result
            if final_mask is not None:
                # Color the final defects in bright green for visibility
                final_defects_colored = cv2.merge([
                    np.zeros_like(final_mask), 
                    final_mask, 
                    np.zeros_like(final_mask)
                ])
                
                output_image = cv2.addWeighted(bgr_image, 0.7, final_defects_colored, 0.3, 0)
                
                output_filename = "modularized_scripts/z_test_output_combined_defects.png"
                cv2.imwrite(output_filename, output_image)
                print(f"Success! Saved combined defect map visualization to '{output_filename}'")
            else:
                print("Mask combination failed.")
        else:
                print(f"Could not find the '{target_zone_name}' zone mask.")
    else:
        print(f"Could not load image at {image_path}.")
