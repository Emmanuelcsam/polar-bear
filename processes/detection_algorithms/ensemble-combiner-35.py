import numpy as np
import cv2
from typing import Dict, Tuple

from .defect_detection_config import DefectDetectionConfig

def ensemble_combination(all_detections: Dict[str, Dict[str, np.ndarray]], 
                         image_shape: Tuple[int, int],
                         config: DefectDetectionConfig) -> Dict[str, np.ndarray]:
    """Combine all detection results using weighted voting"""
    h, w = image_shape
    combined_masks = {}
    
    # Separate masks for scratches and regions
    scratch_methods = ['lei', 'hessian', 'frangi', 'radon', 'phase_congruency']
    region_methods = ['do2mr', 'log', 'doh', 'mser', 'lbp', 'otsu']
    general_methods = ['gradient', 'watershed', 'canny', 'adaptive', 'morphological']
    
    for zone_name, zone_detections in all_detections.items():
        # Initialize vote maps
        scratch_votes = np.zeros((h, w), dtype=np.float32)
        region_votes = np.zeros((h, w), dtype=np.float32)
        
        # Accumulate weighted votes
        for method_name, mask in zone_detections.items():
            if mask is None:
                continue
                
            weight = config.confidence_weights.get(method_name, 0.5)
            
            if method_name in scratch_methods:
                scratch_votes += (mask > 0).astype(np.float32) * weight
            elif method_name in region_methods:
                region_votes += (mask > 0).astype(np.float32) * weight
            else:  # general methods contribute to both
                scratch_votes += (mask > 0).astype(np.float32) * weight * 0.5
                region_votes += (mask > 0).astype(np.float32) * weight * 0.5
        
        # Normalize vote maps
        max_scratch_vote = sum(config.confidence_weights.get(m, 0.5) for m in scratch_methods) + \
                           sum(config.confidence_weights.get(m, 0.5) * 0.5 for m in general_methods)
        max_region_vote = sum(config.confidence_weights.get(m, 0.5) for m in region_methods) + \
                          sum(config.confidence_weights.get(m, 0.5) * 0.5 for m in general_methods)
        
        if max_scratch_vote > 0:
            scratch_votes /= max_scratch_vote
        if max_region_vote > 0:
            region_votes /= max_region_vote
        
        # Apply threshold
        scratch_mask = (scratch_votes >= config.ensemble_vote_threshold).astype(np.uint8) * 255
        region_mask = (region_votes >= config.ensemble_vote_threshold).astype(np.uint8) * 255
        
        # Morphological refinement
        scratch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, scratch_kernel)
        
        region_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, region_kernel)
        
        combined_masks[f'{zone_name}_scratches'] = scratch_mask
        combined_masks[f'{zone_name}_regions'] = region_mask
        combined_masks[f'{zone_name}_all'] = cv2.bitwise_or(scratch_mask, region_mask)
    
    return combined_masks

if __name__ == '__main__':
    config = DefectDetectionConfig()
    shape = (200, 200)

    # Create dummy detection masks
    detections = {
        'core': {
            'lei': np.zeros(shape, dtype=np.uint8),
            'do2mr': np.zeros(shape, dtype=np.uint8)
        }
    }
    cv2.line(detections['core']['lei'], (20, 20), (180, 20), 255, 2) # A scratch
    cv2.circle(detections['core']['do2mr'], (100, 100), 15, 255, -1) # A region

    print("Running ensemble combination on dummy detection masks...")
    combined = ensemble_combination(detections, shape, config)

    for name, mask in combined.items():
        path = f"ensemble_{name}.png"
        cv2.imwrite(path, mask)
        print(f"Saved '{path}' for visual inspection.")
