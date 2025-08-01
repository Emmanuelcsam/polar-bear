from typing import Dict
import numpy as np
from .defect_detection_config import DefectDetectionConfig
from .scratch_false_positive_reducer import reduce_false_positives_scratches
from .region_false_positive_reducer import reduce_false_positives_regions


def reduce_false_positives(combined_masks: Dict[str, np.ndarray],
    preprocessed_images: Dict[str, np.ndarray], config: DefectDetectionConfig
    ) ->Dict[str, np.ndarray]:
    """Apply advanced false positive reduction"""
    refined_masks = {}
    validation_image = preprocessed_images.get('bilateral_0',
        preprocessed_images.get('original'))
    for mask_name, mask in combined_masks.items():
        if 'scratches' in mask_name:
            refined = reduce_false_positives_scratches(mask,
                validation_image, config)
        elif 'regions' in mask_name:
            refined = reduce_false_positives_regions(mask, validation_image,
                config)
        else:
            scratch_mask = combined_masks.get(mask_name.replace('_all',
                '_scratches'), np.zeros_like(mask))
            region_mask = combined_masks.get(mask_name.replace('_all',
                '_regions'), np.zeros_like(mask))
            refined_scratches = reduce_false_positives_scratches(scratch_mask,
                validation_image, config)
            refined_regions = reduce_false_positives_regions(region_mask,
                validation_image, config)
            refined = np.bitwise_or(refined_scratches, refined_regions)
        refined_masks[mask_name] = refined
    return refined_masks


if __name__ == '__main__':
    print("This script contains the 'reduce_false_positives' wrapper function."
        )
    print(
        'It is intended to be used as part of the unified defect detection system.'
        )
