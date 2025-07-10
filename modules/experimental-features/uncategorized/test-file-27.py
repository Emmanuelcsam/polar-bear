
import cv2
import numpy as np
from typing import List, Tuple, Optional

def find_fiber_center_and_radius(processed_images: dict, HOUGH_DP_VALUES: List[float] = [1.0, 1.2, 1.5], HOUGH_MIN_DIST_FACTOR: float = 0.25, HOUGH_PARAM1_VALUES: List[int] = [70, 100, 130], HOUGH_PARAM2_VALUES: List[int] = [35, 45, 55], HOUGH_MIN_RADIUS_FACTOR: float = 0.1, HOUGH_MAX_RADIUS_FACTOR: float = 0.45, CIRCLE_CONFIDENCE_THRESHOLD: float = 0.3) -> Optional[Tuple[Tuple[int, int], float]]:
    """
    Robustly finds the primary circular feature (assumed cladding) center and radius
    using HoughCircles on multiple preprocessed images with varying parameters.
    """
    all_detected_circles: List[Tuple[int, int, int, float, str]] = []
    h, w = processed_images['original_gray'].shape[:2]
    min_dist_circles = int(min(h, w) * HOUGH_MIN_DIST_FACTOR)
    min_radius_hough = int(min(h, w) * HOUGH_MIN_RADIUS_FACTOR)
    max_radius_hough = int(min(h, w) * HOUGH_MAX_RADIUS_FACTOR)

    for image_key in ['gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced']:
        img_to_process = processed_images.get(image_key)
        if img_to_process is None:
            continue

        for dp in HOUGH_DP_VALUES:
            for param1 in HOUGH_PARAM1_VALUES:
                for param2 in HOUGH_PARAM2_VALUES:
                    try:
                        circles = cv2.HoughCircles(
                            img_to_process,
                            cv2.HOUGH_GRADIENT,
                            dp=dp,
                            minDist=min_dist_circles,
                            param1=param1,
                            param2=param2,
                            minRadius=min_radius_hough,
                            maxRadius=max_radius_hough
                        )
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for i in circles[0, :]:
                                cx, cy, r = int(i[0]), int(i[1]), int(i[2])
                                dist_to_img_center = np.sqrt((cx - w // 2) ** 2 + (cy - h // 2) ** 2)
                                normalized_r = r / max_radius_hough if max_radius_hough > 0 else 0
                                confidence = (
                                    (param2 / max(HOUGH_PARAM2_VALUES)) * 0.5
                                    + (normalized_r) * 0.5
                                    - (dist_to_img_center / (min(h, w) / 2)) * 0.2
                                )
                                confidence = max(0, min(1, confidence))
                                all_detected_circles.append((cx, cy, r, confidence, image_key))
                    except Exception as e:
                        print(
                            f"Error in HoughCircles on {image_key} with params ({dp},{param1},{param2}): {e}"
                        )

    if not all_detected_circles:
        print("No circles detected by Hough Transform.")
        return None

    all_detected_circles.sort(key=lambda x: x[3], reverse=True)
    best_cx, best_cy, best_r, best_conf, source = all_detected_circles[0]
    if best_conf < CIRCLE_CONFIDENCE_THRESHOLD:
        print(
            f"Best detected circle confidence ({best_conf:.2f} from {source}) is below threshold "
            f"({CIRCLE_CONFIDENCE_THRESHOLD})."
        )
        return None

    print(
        f"Best fiber center detected at ({best_cx}, {best_cy}) with radius {best_r}px. "
        f"Confidence: {best_conf:.2f} (from {source})."
    )
    return (best_cx, best_cy), float(best_r)

if __name__ == '__main__':
    import sys
    from preprocess_image import preprocess_image
    if len(sys.argv) != 2:
        print("Usage: python find_fiber_center.py <image_path>")
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
        print(f"Found center: {center}, radius: {radius}")
        # Draw the circle on the original image
        output_image = image.copy()
        cv2.circle(output_image, center, int(radius), (0, 255, 0), 2)
        cv2.imwrite("fiber_center_detected.jpg", output_image)
        print("Saved image with detected center to fiber_center_detected.jpg")

