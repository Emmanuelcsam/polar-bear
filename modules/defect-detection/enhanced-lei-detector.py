import cv2
import numpy as np
from .defect_detection_config import DefectDetectionConfig
from .linear_defect_detector import apply_linear_detector


def detect_lei_enhanced(image: np.ndarray, zone_mask: np.ndarray, config:
    DefectDetectionConfig) ->np.ndarray:
    """Enhanced LEI scratch detection"""
    scratch_strength = np.zeros_like(image, dtype=np.float32)
    masked_img = cv2.bitwise_and(image, image, mask=zone_mask)
    if np.any(masked_img):
        enhanced = cv2.equalizeHist(masked_img)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=zone_mask)
    else:
        enhanced = masked_img
    for kernel_length in config.lei_kernel_lengths:
        for angle_step in config.lei_angle_steps:
            for angle in range(0, 180, angle_step):
                angle_rad = np.deg2rad(angle)
                kernel_points = []
                for i in range(-kernel_length // 2, kernel_length // 2 + 1):
                    if i == 0:
                        continue
                    x = int(round(i * np.cos(angle_rad)))
                    y = int(round(i * np.sin(angle_rad)))
                    kernel_points.append((x, y))
                if kernel_points:
                    response = apply_linear_detector(enhanced, kernel_points)
                    scratch_strength = np.maximum(scratch_strength, response)
    if scratch_strength.max() > 0:
        cv2.normalize(scratch_strength, scratch_strength, 0, 255, cv2.
            NORM_MINMAX)
        scratch_strength_uint8 = scratch_strength.astype(np.uint8)
        scratch_mask = np.zeros_like(scratch_strength_uint8)
        masked_pixels = scratch_strength_uint8[zone_mask > 0]
        if masked_pixels.size > 0:
            mean_val = np.mean(masked_pixels)
            std_val = np.std(masked_pixels)
            for factor in config.lei_threshold_factors:
                threshold = mean_val + factor * std_val
                _, mask = cv2.threshold(scratch_strength_uint8, threshold, 
                    255, cv2.THRESH_BINARY)
                scratch_mask = cv2.bitwise_or(scratch_mask, mask)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE,
            kernel_close)
        scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=
            zone_mask)
        return scratch_mask
    return np.zeros_like(image, dtype=np.uint8)


if __name__ == '__main__':
    config = DefectDetectionConfig()
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.line(sample_image, (30, 150), (170, 50), 80, 2)
    zone_mask = np.zeros_like(sample_image)
    cv2.circle(zone_mask, (100, 100), 90, 255, -1)
    print('Running enhanced LEI scratch detection...')
    lei_mask = detect_lei_enhanced(sample_image, zone_mask, config)
    cv2.imwrite('lei_input.png', sample_image)
    cv2.imwrite('lei_detection_mask.png', lei_mask)
    print(
        "Saved 'lei_input.png' and 'lei_detection_mask.png' for visual inspection."
        )
