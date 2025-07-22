from typing import Dict
import numpy as np
from .defect_detection_config import DefectDetectionConfig
from .enhanced_do2mr_detector import detect_do2mr_enhanced
from .enhanced_lei_detector import detect_lei_enhanced
from .scratch_detector import hessian_ridge_detection
from .frangi_linear_detector import frangi_vesselness
from .phase_congruency_detector import phase_congruency_detection
from .scratch_detector import radon_line_detection
from .gradient_defect_detector import gradient_based_detection
from .main_defect_detector import scale_normalized_log
from .hessian_determinant_detector import determinant_of_hessian
from .mser_region_detector import mser_detection
from .watershed_defect_segmenter import watershed_detection
from .canny_edge_detector import canny_detection
from .adaptive_threshold_detector import adaptive_threshold_detection
from .texture_pattern_detector import lbp_detection
from .otsu_threshold_detector import otsu_based_detection
from .morphological_defect_finder import morphological_detection


def apply_all_detection_methods(preprocessed_images: Dict[str, np.ndarray],
    zone_masks: Dict[str, np.ndarray], config: DefectDetectionConfig) ->Dict[
    str, Dict[str, np.ndarray]]:
    """Apply all detection methods to all zones"""
    all_detections = {}
    img_for_region = preprocessed_images.get('clahe_0', preprocessed_images
        ['original'])
    img_for_scratch = preprocessed_images.get('coherence',
        preprocessed_images['original'])
    img_for_general = preprocessed_images.get('bilateral_0',
        preprocessed_images['original'])
    for zone_name, zone_mask in zone_masks.items():
        print(f'  Processing zone: {zone_name}')
        zone_detections = {'do2mr': detect_do2mr_enhanced(img_for_region,
            zone_mask, config), 'lei': detect_lei_enhanced(img_for_scratch,
            zone_mask, config), 'hessian': hessian_ridge_detection(
            img_for_scratch, zone_mask, config.hessian_scales), 'frangi':
            frangi_vesselness(img_for_scratch, zone_mask, config.
            frangi_scales), 'phase_congruency': phase_congruency_detection(
            img_for_general, zone_mask), 'radon': radon_line_detection(
            img_for_scratch, zone_mask), 'gradient':
            gradient_based_detection(img_for_general, zone_mask), 'log':
            scale_normalized_log(img_for_region, zone_mask, config.
            log_scales), 'doh': determinant_of_hessian(img_for_region,
            zone_mask, config.log_scales[::2]), 'mser': mser_detection(
            img_for_region, zone_mask), 'watershed': watershed_detection(
            img_for_general, zone_mask), 'canny': canny_detection(
            img_for_general, zone_mask), 'adaptive':
            adaptive_threshold_detection(img_for_general, zone_mask), 'lbp':
            lbp_detection(img_for_general, zone_mask), 'otsu':
            otsu_based_detection(img_for_general, zone_mask),
            'morphological': morphological_detection(img_for_general,
            zone_mask)}
        all_detections[zone_name] = zone_detections
    return all_detections


if __name__ == '__main__':
    print("This script contains the 'apply_all_detection_methods' function.")
    print(
        'It is intended to be used as part of the unified defect detection system.'
        )
