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
    start_time = time.perf_counter()  # Record timestamp for performance measurement
    
    # Guard clause: ensure both input images are valid before processing
    if live_frame is None or ref_img_processed is None:
        return None, None, None  # Return tuple of None values to maintain consistent return format

    # --- Preprocessing ---
    live_img_processed = preprocess_image(live_frame)  # Normalize, denoise, and standardize the live camera frame
    
    # --- Difference Analysis ---
    # Compare processed live frame against reference to identify potential defect regions
    diff_mask, ssim_score = compute_ssim_difference(ref_img_processed, 
                                                   live_img_processed)
    
    all_detections = []  # Initialize list to accumulate all detected defects from different algorithms
    # Only run defect detectors if SSIM analysis found significant differences
    if diff_mask is not None:
        # Log the structural similarity score for debugging and threshold tuning
        logging.debug(f"SSIM score: {ssim_score:.4f}. Potential defects found, "
                    f"running detectors.")
        
        # --- Run Specialized Detectors ---
        scratch_detections = detect_scratches(live_img_processed)  # Find linear defects using edge detection
        all_detections.extend(scratch_detections)  # Add scratch results to master detection list
        
        blob_detections = detect_blobs(diff_mask)  # Find irregular shaped defects in difference regions
        all_detections.extend(blob_detections)  # Add blob results to master detection list
        
        circle_detections = detect_circles(live_img_processed, diff_mask)  # Find circular defects using Hough transform
        all_detections.extend(circle_detections)  # Add circle results to master detection list

    else:
        # Log when no significant differences found, indicating clean frame
        logging.debug(f"SSIM score: {ssim_score:.4f}. No significant difference.")
        
    # --- Visualization ---
    annotated_frame = live_frame.copy()  # Create independent copy to avoid modifying original frame
    # Iterate through each detected defect to draw visual annotations
    for det in all_detections:
        det_type = det["type"]  # Extract defect classification (Scratch, Blob, or Circle)
        # Handle rectangular defects (scratches and blobs) with bounding boxes
        if det_type == "Scratch" or det_type == "Blob":
            x, y, w, h = det["location"]  # Unpack bounding box coordinates and dimensions
            # Set color coding: red for scratches (critical), yellow for blobs (moderate)
            color = (0, 0, 255) if det_type == "Scratch" else (0, 255, 255)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)  # Draw bounding rectangle with 2px thickness
            # Place defect type label above the bounding box
            cv2.putText(annotated_frame, det_type, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Handle circular defects with center point and radius annotations
        elif det_type == "Circle":
            center_x, center_y, radius = det["location"]  # Unpack circle parameters
            color = (255, 0, 0)  # Blue color for circular defects
            cv2.circle(annotated_frame, (center_x, center_y), radius, color, 2)  # Draw outer circle boundary
            cv2.circle(annotated_frame, (center_x, center_y), 2, color, 3)  # Draw center point marker
            # Calculate text position ensuring it stays within image boundaries
            text_x = max(0, center_x - radius)  # Prevent negative x coordinate
            text_y = max(0, center_y - radius - 10)  # Prevent negative y coordinate with offset
            # Place defect type label near the circle
            cv2.putText(annotated_frame, det_type, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add FPS and status info to the frame
    end_time = time.perf_counter()  # Capture end timestamp for performance calculation
    processing_time = (end_time - start_time) * 1000  # Convert elapsed time to milliseconds for readability
    # Calculate frames per second with division by zero protection
    fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    # Display real-time performance metrics in top-left corner
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text with 2px thickness
    # Display total defect count for immediate quality assessment
    cv2.putText(annotated_frame, f"Defects: {len(all_detections)}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text positioned below FPS
    
    # Log detailed performance metrics for debugging and optimization
    logging.debug(f"Frame processing took {processing_time:.2f} ms. "
                f"Found {len(all_detections)} defects.")
    
    # Return complete results: annotated visualization, detection data, and difference mask
    return annotated_frame, all_detections, diff_mask 