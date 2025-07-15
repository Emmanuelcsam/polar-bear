
import cv2
import numpy as np
from typing import Tuple, List, Optional

def detect_region_defects_do2mr(image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str, DO2MR_KERNEL_SIZES: List[Tuple[int, int]] = [(5, 5), (9, 9), (13, 13)], DO2MR_GAMMA_VALUES: List[float] = [2.0, 2.5, 3.0], DO2MR_MEDIAN_BLUR_KERNEL_SIZE: int = 5, DO2MR_MORPH_OPEN_KERNEL_SIZE: Tuple[int, int] = (3, 3)) -> Optional[np.ndarray]:
    """
    Detects region-based defects using a DO2MR-inspired method.
    Returns a binary mask of detected region defects, or None if an error occurs.
    """
    if image_gray is None or zone_mask is None:
        print("Input image or mask is None for DO2MR.")
        return None

    masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)
    vote_map = np.zeros_like(image_gray, dtype=np.float32)

    for kernel_size in DO2MR_KERNEL_SIZES:
        struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        min_filtered = cv2.erode(masked_image, struct_element)
        max_filtered = cv2.dilate(masked_image, struct_element)
        residual = cv2.subtract(max_filtered, min_filtered)

        if DO2MR_MEDIAN_BLUR_KERNEL_SIZE > 0:
            res_blurred = cv2.medianBlur(residual, DO2MR_MEDIAN_BLUR_KERNEL_SIZE)
        else:
            res_blurred = residual

        for gamma in DO2MR_GAMMA_VALUES:
            masked_res_vals = res_blurred[zone_mask > 0]
            if masked_res_vals.size == 0:
                print(
                    f"Zone mask for '{zone_name}' is empty or residual is all zero. "
                    f"Skipping DO2MR for gamma={gamma}, kernel={kernel_size}."
                )
                continue

            mean_val = np.mean(masked_res_vals)
            std_val = np.std(masked_res_vals)
            thresh_val = np.clip(mean_val + gamma * std_val, 0, 255)

            _, defect_mask_pass = cv2.threshold(res_blurred, thresh_val, 255, cv2.THRESH_BINARY)
            defect_mask_pass = cv2.bitwise_and(defect_mask_pass, defect_mask_pass, mask=zone_mask)

            open_k = DO2MR_MORPH_OPEN_KERNEL_SIZE
            if open_k[0] > 0 and open_k[1] > 0:
                open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_k)
                defect_mask_pass = cv2.morphologyEx(defect_mask_pass, cv2.MORPH_OPEN, open_kernel)

            vote_map += (defect_mask_pass / 255.0)

    num_param_sets = len(DO2MR_KERNEL_SIZES) * len(DO2MR_GAMMA_VALUES)
    min_votes = max(1, int(num_param_sets * 0.3))
    combined_map = np.where(vote_map >= min_votes, 255, 0).astype(np.uint8)

    return combined_map

if __name__ == '__main__':
    import sys
    from preprocess_image import preprocess_image
    from find_fiber_center import find_fiber_center_and_radius

    if len(sys.argv) != 2:
        print("Usage: python detect_region_defects.py <image_path>")
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

        defects = detect_region_defects_do2mr(processed_images['clahe_enhanced'], zone_mask, 'cladding')
        if defects is not None:
            cv2.imwrite("do2mr_defects_detected.jpg", defects)
            print("Saved detected defects to do2mr_defects_detected.jpg")

