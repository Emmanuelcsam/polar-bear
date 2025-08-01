"""
Main frame processing module that orchestrates the entire defect detection pipeline.
"""

import time
import cv2
import logging
from detection.preprocessing import preprocess_image
from detection.ssim_detector import compute_ssim_difference
from detection.scratch_detector import detect_scratches
from detection.blob_detector import detect_blobs
from detection.circle_detector import detect_circles


def process_frame(live_frame, ref_img_processed):
    """
    Orchestrates the entire defect detection pipeline for a single frame.
    """
    start_time = time.perf_counter()
    
    if live_frame is None or ref_img_processed is None:
        return None, None, None

    # --- Preprocessing ---
    live_img_processed = preprocess_image(live_frame)
    
    # --- Difference Analysis ---
    diff_mask, ssim_score = compute_ssim_difference(ref_img_processed, 
                                                   live_img_processed)
    
    all_detections = []
    if diff_mask is not None:
        logging.debug(f"SSIM score: {ssim_score:.4f}. Potential defects found, "
                    f"running detectors.")
        
        # --- Run Specialized Detectors ---
        scratch_detections = detect_scratches(live_img_processed)
        all_detections.extend(scratch_detections)
        
        blob_detections = detect_blobs(diff_mask)
        all_detections.extend(blob_detections)
        
        circle_detections = detect_circles(live_img_processed, diff_mask)
        all_detections.extend(circle_detections)

    else:
        logging.debug(f"SSIM score: {ssim_score:.4f}. No significant difference.")
        
    # --- Visualization ---
    annotated_frame = live_frame.copy()
    for det in all_detections:
        det_type = det["type"]
        if det_type == "Scratch" or det_type == "Blob":
            x, y, w, h = det["location"]
            # Red for scratch, Yellow for blob
            color = (0, 0, 255) if det_type == "Scratch" else (0, 255, 255)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated_frame, det_type, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif det_type == "Circle":
            center_x, center_y, radius = det["location"]
            color = (255, 0, 0)  # Blue for circle
            cv2.circle(annotated_frame, (center_x, center_y), radius, color, 2)
            cv2.circle(annotated_frame, (center_x, center_y), 2, color, 3)  # Center dot
            # Fix overflow error by ensuring coordinates are within bounds
            text_x = max(0, center_x - radius)
            text_y = max(0, center_y - radius - 10)
            cv2.putText(annotated_frame, det_type, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add FPS and status info to the frame
    end_time = time.perf_counter()
    processing_time = (end_time - start_time) * 1000
    fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Defects: {len(all_detections)}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    logging.debug(f"Frame processing took {processing_time:.2f} ms. "
                f"Found {len(all_detections)} defects.")
    
    return annotated_frame, all_detections, diff_mask 