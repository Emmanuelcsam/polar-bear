
import cv2
import numpy as np
from typing import Tuple, List, Optional

def detect_scratches_lei(image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str, LEI_KERNEL_LENGTHS: List[int] = [11, 17, 23], LEI_ANGLE_STEP: int = 15, LEI_THRESHOLD_FACTOR: float = 2.0, LEI_MORPH_CLOSE_KERNEL_SIZE: Tuple[int, int] = (5, 1), LEI_MIN_SCRATCH_AREA_PX: int = 15) -> Optional[np.ndarray]:
    """
    Detects linear scratches using an LEI-inspired method.
    Returns a binary mask of detected scratches, or None on error.
    """
    if image_gray is None or zone_mask is None:
        print("Input image or mask is None for LEI.")
        return None

    masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)
    enhanced_image = cv2.equalizeHist(masked_image)
    enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=zone_mask)
    max_response_map = np.zeros_like(enhanced_image, dtype=np.float32)

    for kernel_length in LEI_KERNEL_LENGTHS:
        for angle_deg in range(0, 180, LEI_ANGLE_STEP):
            line_kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
            rot_matrix = cv2.getRotationMatrix2D((kernel_length // 2, 0), angle_deg, 1.0)
            bbox_size = int(np.ceil(kernel_length * 1.5))
            rotated_kernel = cv2.warpAffine(line_kernel_base, rot_matrix, (bbox_size, bbox_size))

            if np.sum(rotated_kernel) > 0:
                rotated_kernel = rotated_kernel.astype(np.float32) / np.sum(rotated_kernel)
            else:
                continue

            response = cv2.filter2D(enhanced_image.astype(np.float32), -1, rotated_kernel)
            max_response_map = np.maximum(max_response_map, response)

    if np.max(max_response_map) > 0:
        cv2.normalize(max_response_map, max_response_map, 0, 255, cv2.NORM_MINMAX)
    response_8u = max_response_map.astype(np.uint8)

    _, scratch_mask = cv2.threshold(response_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, close_kernel)

    scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=zone_mask)
    return scratch_mask

if __name__ == '__main__':
    import sys
    from preprocess_image import preprocess_image
    from find_fiber_center import find_fiber_center_and_radius

    if len(sys.argv) != 2:
        print("Usage: python detect_scratches.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    processed_images = preprocess_image(image)
    center_radius = find_fiber_center_and_radius(processed_images)

    if center_radius:
        center, radius = center_radius
        # Create a circular mask for the entire fiber for demonstration
        h, w = image.shape[:2]
        zone_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(zone_mask, center, int(radius), 255, -1)

        scratches = detect_scratches_lei(processed_images['clahe_enhanced'], zone_mask, 'cladding')
        if scratches is not None:
            cv2.imwrite("lei_scratches_detected.jpg", scratches)
            print("Saved detected scratches to lei_scratches_detected.jpg")
