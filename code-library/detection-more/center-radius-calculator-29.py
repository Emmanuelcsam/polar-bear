from typing import Dict, Optional, Tuple, List
import cv2
import numpy as np

from log_message import log_message
from inspector_config import InspectorConfig
from preprocess_image import preprocess_image
from load_single_image import load_single_image
from pathlib import Path

def find_fiber_center_and_radius(
    processed_images: Dict[str, np.ndarray], 
    config: InspectorConfig
) -> Optional[Tuple[Tuple[int, int], float]]:
    """
    Robustly finds the primary circular feature (assumed cladding) center and radius.
    
    Args:
        processed_images: Dictionary of preprocessed grayscale images.
        config: An InspectorConfig object with Hough Circle parameters.
        
    Returns:
        A tuple (center_xy, radius_px) or None if no reliable circle is found.
    """
    log_message("Starting fiber center and radius detection...")
    
    if not processed_images or 'original_gray' not in processed_images:
        log_message("Processed images dictionary is empty or missing 'original_gray'.", level="ERROR")
        return None

    all_detected_circles: List[Tuple[int, int, int, float, str]] = []
    original_gray = processed_images['original_gray']
    h, w = original_gray.shape[:2]
    img_center_x, img_center_y = w // 2, h // 2

    # Define min/max radius expectations for selection scoring
    expected_min_r = 0.10 * min(h, w)
    expected_max_r = 0.35 * min(h, w)

    # Define parameters for Hough transform itself
    min_dist_circles = int(min(h, w) * config.HOUGH_MIN_DIST_FACTOR)
    min_r_hough = int(min(h, w) * config.HOUGH_MIN_RADIUS_FACTOR)
    max_r_hough = int(min(h, w) * config.HOUGH_MAX_RADIUS_FACTOR)
    log_message(f"Hough params: minDist={min_dist_circles}, minR={min_r_hough}, maxR={max_r_hough}")

    images_to_check = ['clahe_enhanced', 'gaussian_blurred', 'bilateral_filtered']
    for image_key in images_to_check:
        img_to_process = processed_images.get(image_key)
        if img_to_process is None:
            continue

        for dp in config.HOUGH_DP_VALUES:
            for param1 in config.HOUGH_PARAM1_VALUES:
                for param2 in config.HOUGH_PARAM2_VALUES:
                    try:
                        circles = cv2.HoughCircles(
                            img_to_process, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist_circles,
                            param1=param1, param2=param2, minRadius=min_r_hough, maxRadius=max_r_hough
                        )
                        if circles is not None:
                            for i in circles[0, :]:
                                cx, cy, r = int(i[0]), int(i[1]), int(i[2])
                                if r == 0: continue

                                # --- Confidence Score Calculation ---
                                dist_to_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
                                max_dist = np.sqrt((w/2)**2 + (h/2)**2)
                                center_prox_score = max(0, 1.0 - (dist_to_center / max_dist))

                                radius_score = 0.0
                                if expected_min_r <= r <= expected_max_r:
                                    radius_score = 1.0
                                elif r < expected_min_r:
                                    radius_score = max(0, 1.0 - (expected_min_r - r) / expected_min_r)
                                else: # r > expected_max_r
                                    radius_score = max(0, 1.0 - (r - expected_max_r) / expected_max_r)

                                accumulator_score = param2 / max(config.HOUGH_PARAM2_VALUES)
                                
                                confidence = (center_prox_score * 0.4) + (radius_score * 0.4) + (accumulator_score * 0.2)
                                all_detected_circles.append((cx, cy, r, confidence, f"{image_key}_dp{dp}_p1{param1}_p2{param2}"))
                    except Exception as e:
                        log_message(f"Error in HoughCircles on {image_key}: {e}", level="WARNING")

    if not all_detected_circles:
        log_message("No circles detected by any Hough Transform attempt.", level="WARNING")
        return None

    all_detected_circles.sort(key=lambda x: x[3], reverse=True)
    log_message(f"Top 3 circle candidates: {all_detected_circles[:3]}", level="DEBUG")

    best_cx, best_cy, best_r, best_conf, src_params = all_detected_circles[0]

    if best_conf < config.CIRCLE_CONFIDENCE_THRESHOLD:
        log_message(f"Best circle confidence ({best_conf:.2f}) is below threshold ({config.CIRCLE_CONFIDENCE_THRESHOLD}).", level="WARNING")
        return None

    log_message(f"Best fiber center detected at ({best_cx}, {best_cy}) with radius {best_r}px. Confidence: {best_conf:.2f} (from {src_params}).")
    return (best_cx, best_cy), float(best_r)

if __name__ == '__main__':
    # Example of how to use the find_fiber_center_and_radius function
    
    # 1. Setup: Load a config and an image, then preprocess it
    conf = InspectorConfig()
    image_path = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    
    print(f"--- Loading and preprocessing image: {image_path} ---")
    bgr_image = load_single_image(image_path)
    
    if bgr_image is not None:
        preprocessed_bundle = preprocess_image(bgr_image, conf)
        
        if preprocessed_bundle:
            # 2. Run the center detection function
            print("\n--- Finding fiber center and radius ---")
            result = find_fiber_center_and_radius(preprocessed_bundle, conf)
            
            # 3. Verify and display the output
            if result:
                center, radius = result
                print(f"\nSuccess! Found center at {center} with radius {radius:.2f} pixels.")
                
                # Draw the result on the original image for visual confirmation
                annotated_image = bgr_image.copy()
                cv2.circle(annotated_image, center, int(radius), (0, 255, 0), 2) # Green circle for cladding
                cv2.circle(annotated_image, center, 2, (0, 0, 255), 3) # Red dot for center
                
                output_filename = "modularized_scripts/z_test_output_center_detection.png"
                cv2.imwrite(output_filename, annotated_image)
                print(f"Saved annotated image to '{output_filename}' for verification.")
            else:
                print("\nCould not find a reliable fiber center in the image.")
        else:
            print("Preprocessing failed, cannot run center detection.")
    else:
        print(f"Could not load the image at {image_path}.")
