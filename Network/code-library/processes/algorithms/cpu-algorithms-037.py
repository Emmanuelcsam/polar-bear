import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .defect_detection_config import DefectDetectionConfig
from .advanced_preprocessing import advanced_preprocessing
from .find_fiber_center_enhanced import find_fiber_center_enhanced
from .classical_defect_detector import create_zone_masks
from .all_methods_detector import apply_all_detection_methods
from .ensemble_combiner import ensemble_combination
from .false_positive_reducer import reduce_false_positives
from .analyze_defects import analyze_defects
from .apply_pass_fail_criteria import apply_pass_fail_criteria


def detect_defects(image_path: Union[str, Path], config:
    DefectDetectionConfig, cladding_diameter_um: Optional[float]=None,
    core_diameter_um: Optional[float]=None) ->Dict[str, Any]:
    """Main method to detect all defects in an image"""
    print(f'Starting defect detection for: {image_path}')
    start_time = time.time()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Could not load image: {image_path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape
        ) == 3 else image.copy()
    print('Step 1: Advanced preprocessing...')
    preprocessed_images = advanced_preprocessing(gray, config)
    print('Step 2: Finding fiber center and creating zones...')
    fiber_info = find_fiber_center_enhanced(preprocessed_images, config)
    if not fiber_info:
        h, w = gray.shape
        fiber_info = {'center': (w // 2, h // 2), 'radius': min(h, w) // 4,
            'confidence': 0.0}
        print(
            'Warning: Could not detect fiber center accurately. Using image center as fallback.'
            )
    pixels_per_micron = 2 * fiber_info['radius'
        ] / cladding_diameter_um if cladding_diameter_um and fiber_info[
        'radius'] > 0 else None
    if pixels_per_micron:
        print(f'Calculated scale: {pixels_per_micron:.4f} pixels/micron')
    zone_masks = create_zone_masks(gray.shape, fiber_info['center'],
        fiber_info['radius'], core_diameter_um, cladding_diameter_um)
    print('Step 3: Applying all detection methods...')
    all_detections = apply_all_detection_methods(preprocessed_images,
        zone_masks, config)
    print('Step 4: Ensemble combination of results...')
    combined_masks = ensemble_combination(all_detections, gray.shape, config)
    print('Step 5: Reducing false positives...')
    refined_masks = reduce_false_positives(combined_masks,
        preprocessed_images, config)
    print('Step 6: Analyzing and classifying defects...')
    defects = analyze_defects(refined_masks, pixels_per_micron)
    print('Step 7: Applying pass/fail criteria...')
    pass_fail_result = apply_pass_fail_criteria(defects, pixels_per_micron)
    duration = time.time() - start_time
    print(f'Detection completed in {duration:.2f} seconds')
    return {'defects': defects, 'pass_fail': pass_fail_result, 'fiber_info':
        fiber_info, 'zone_masks': zone_masks, 'detection_masks':
        refined_masks, 'processing_time': duration, 'image_path': str(
        image_path), 'pixels_per_micron': pixels_per_micron}


if __name__ == '__main__':
    print(
        "This script contains the main 'detect_defects' orchestrator function."
        )
    print(
        'It is intended to be used as part of the unified defect detection system.'
        )
    print("To run a full detection, use 'jill_main.py'.")
