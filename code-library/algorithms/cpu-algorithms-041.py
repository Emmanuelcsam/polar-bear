#!/usr/bin/env python3
"""
LEI (Linear Enhancement Inspector) Scratch Detection
==================================================
Standalone implementation of the LEI algorithm for linear defect (scratch) detection
in fiber optic end-face images. This algorithm uses oriented linear filters to detect
scratches at multiple orientations and scales.

Reference: Based on research in linear defect detection using directional filtering.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple
from pathlib import Path
import argparse
import math


def create_linear_kernel(length: int, angle_deg: float, thickness: int = 1) -> np.ndarray:
    """
    Create a linear kernel for scratch detection at a specific angle.
    
    Args:
        length: Length of the linear kernel
        angle_deg: Angle in degrees (0-180)
        thickness: Thickness of the line (default: 1)
        
    Returns:
        Linear kernel as float32 array
    """
    if length <= 0:
        raise ValueError("Kernel length must be positive")
    
    # Create empty kernel
    kernel = np.zeros((length, length), dtype=np.float32)
    
    # Draw vertical line through center
    center = length // 2
    start_y = max(0, center - thickness // 2)
    end_y = min(length, center + thickness // 2 + 1)
    
    for y in range(start_y, end_y):
        cv2.line(kernel, (center, 0), (center, length - 1), 1.0, thickness=1)
    
    # Rotate kernel to desired angle
    if angle_deg != 0:
        center_point = (float(length - 1) / 2.0, float(length - 1) / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center_point, float(angle_deg), 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (length, length), 
                               flags=cv2.INTER_LINEAR)
    
    # Normalize kernel
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel = kernel / kernel_sum
    
    return kernel


def lei_scratch_detection(
    image: np.ndarray,
    kernel_lengths: List[int] = [11, 17, 23],
    angle_step_deg: int = 15,
    dual_branch_width: int = 2,
    response_threshold: float = 0.1
) -> np.ndarray:
    """
    Apply LEI (Linear Enhancement Inspector) algorithm for scratch detection.
    
    This algorithm creates linear kernels at multiple orientations and scales,
    then applies them to detect linear features (scratches) in the image.
    
    Args:
        image: Input grayscale image
        kernel_lengths: List of kernel lengths for multi-scale detection
        angle_step_deg: Angular step for orientation search (degrees)
        dual_branch_width: Width for dual-branch enhancement
        response_threshold: Threshold for response normalization
        
    Returns:
        Float32 response map showing scratch likelihood
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
        
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")
    
    # Ensure image is float32 for processing
    if image.dtype != np.float32:
        image_f32 = image.astype(np.float32) / 255.0
    else:
        image_f32 = image.copy()
    
    h, w = image_f32.shape
    max_response = np.zeros((h, w), dtype=np.float32)
    
    logging.debug(f"Applying LEI with kernels {kernel_lengths}, angle_step={angle_step_deg}")
    
    # Try different kernel lengths for multi-scale detection
    for length in kernel_lengths:
        if length <= 0:
            continue
            
        logging.debug(f"Processing kernel length: {length}")
        
        # Search for scratches at different orientations
        for angle_deg in range(0, 180, angle_step_deg):
            # Create linear kernel for current orientation
            kernel = create_linear_kernel(length, angle_deg)
            
            # Apply convolution to detect linear features
            response = cv2.filter2D(image_f32, cv2.CV_32F, kernel)
            
            # Dual-branch enhancement (if configured)
            if dual_branch_width > 0:
                # Create perpendicular kernel for contrast enhancement
                perp_angle = (angle_deg + 90) % 180
                perp_kernel = create_linear_kernel(dual_branch_width * 2 + 1, perp_angle)
                perp_response = cv2.filter2D(image_f32, cv2.CV_32F, perp_kernel)
                
                # Enhance response by subtracting perpendicular response
                response = response - 0.5 * perp_response
            
            # Keep maximum response across all orientations
            max_response = np.maximum(max_response, response)
    
    # Normalize response
    if np.max(max_response) > response_threshold:
        max_response = max_response / np.max(max_response)
    
    logging.debug(f"LEI max response: {np.max(max_response):.4f}")
    return max_response


def advanced_scratch_detection(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 15,
    min_line_length: int = 10,
    max_line_gap: int = 5
) -> np.ndarray:
    """
    Advanced scratch detection using Canny edge detection + Hough line transform.
    
    Args:
        image: Input grayscale image
        canny_low: Low threshold for Canny edge detection
        canny_high: High threshold for Canny edge detection
        hough_threshold: Accumulator threshold for Hough line detection
        min_line_length: Minimum line length for Hough transform
        max_line_gap: Maximum gap between line segments
        
    Returns:
        Binary mask with detected line segments
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    
    # Ensure input is uint8
    if image.dtype != np.uint8:
        normalized = np.zeros_like(image, dtype=np.uint8)
        cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX)
        image = normalized
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, canny_low, canny_high, apertureSize=3)
    
    # Initialize mask for detected line segments
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Detect line segments using probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi / 180, 
        threshold=hough_threshold,
        minLineLength=min_line_length, 
        maxLineGap=max_line_gap
    )
    
    # Draw detected lines on mask
    if lines is not None:
        logging.debug(f"Detected {len(lines)} line segments")
        for line_seg in lines:
            x1, y1, x2, y2 = line_seg[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=2)
    else:
        logging.debug("No line segments detected")
    
    return mask


def combine_scratch_detections(
    lei_response: np.ndarray,
    hough_mask: np.ndarray,
    lei_threshold: float = 0.3,
    combination_method: str = "union"
) -> np.ndarray:
    """
    Combine LEI response map with Hough line detection results.
    
    Args:
        lei_response: LEI response map (float32, 0-1)
        hough_mask: Hough line detection mask (uint8, 0/255)
        lei_threshold: Threshold for LEI response
        combination_method: "union", "intersection", or "weighted"
        
    Returns:
        Combined binary mask
    """
    # Threshold LEI response to create binary mask
    lei_binary = (lei_response > lei_threshold).astype(np.uint8) * 255
    
    if combination_method == "union":
        # Logical OR of both methods
        combined = cv2.bitwise_or(lei_binary, hough_mask)
    elif combination_method == "intersection":
        # Logical AND of both methods
        combined = cv2.bitwise_and(lei_binary, hough_mask)
    elif combination_method == "weighted":
        # Weighted combination
        lei_weight = 0.7
        hough_weight = 0.3
        combined = cv2.addWeighted(lei_binary, lei_weight, hough_mask, hough_weight, 0)
        combined = (combined > 127).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")
    
    return combined


def classify_scratches(
    mask: np.ndarray,
    min_aspect_ratio: float = 3.0,
    min_length_px: int = 10
) -> Tuple[List[dict], np.ndarray]:
    """
    Classify detected features as scratches based on geometric properties.
    
    Args:
        mask: Binary mask with detected features
        min_aspect_ratio: Minimum aspect ratio to classify as scratch
        min_length_px: Minimum length in pixels
        
    Returns:
        Tuple of (scratch_list, classified_mask)
    """
    # Find connected components
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )
    
    scratches = []
    classified_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        # Get component statistics
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Calculate aspect ratio
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        length = max(w, h)
        
        # Classify as scratch if meets criteria
        if aspect_ratio >= min_aspect_ratio and length >= min_length_px:
            # Extract component mask
            component_mask = (labels_img == i).astype(np.uint8)
            
            # Find contour for more detailed analysis
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour
                contour = max(contours, key=cv2.contourArea)
                
                # Fit rotated rectangle for better measurements
                if len(contour) >= 5:  # Minimum points for fitEllipse
                    rotated_rect = cv2.minAreaRect(contour)
                    (cx, cy), (rect_w, rect_h), angle = rotated_rect
                    
                    # Use rotated rectangle dimensions
                    actual_length = max(rect_w, rect_h)
                    actual_width = min(rect_w, rect_h)
                    actual_aspect_ratio = actual_length / (actual_width + 1e-6)
                    
                    scratch_info = {
                        'id': f'scratch_{i}',
                        'centroid': (float(cx), float(cy)),
                        'area': int(area),
                        'length': float(actual_length),
                        'width': float(actual_width),
                        'aspect_ratio': float(actual_aspect_ratio),
                        'angle': float(angle),
                        'bbox': (int(x), int(y), int(w), int(h)),
                        'contour': contour
                    }
                    
                    scratches.append(scratch_info)
                    classified_mask[labels_img == i] = 255
    
    logging.info(f"Classified {len(scratches)} scratches from {num_labels-1} components")
    return scratches, classified_mask


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for scratch detection."""
    image_path_obj = Path(image_path)
    
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path_obj), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Apply histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Apply slight Gaussian blur to reduce noise
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return denoised


def visualize_scratch_results(
    original: np.ndarray, 
    lei_response: np.ndarray, 
    hough_mask: np.ndarray,
    combined_mask: np.ndarray,
    scratches: List[dict],
    save_path: Optional[str] = None
):
    """Visualize scratch detection results."""
    # Create color version of original
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Draw scratch bounding boxes and labels
    for scratch in scratches:
        x, y, w, h = scratch['bbox']
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label
        label = f"L:{scratch['length']:.1f} AR:{scratch['aspect_ratio']:.1f}"
        cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (0, 255, 0), 1)
    
    # Create comparison visualization
    lei_vis = cv2.normalize(lei_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lei_vis = cv2.cvtColor(lei_vis, cv2.COLOR_GRAY2BGR)
    hough_vis = cv2.cvtColor(hough_mask, cv2.COLOR_GRAY2BGR)
    combined_vis = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    
    # Stack visualizations
    top_row = np.hstack([cv2.cvtColor(original, cv2.COLOR_GRAY2BGR), lei_vis])
    bottom_row = np.hstack([hough_vis, combined_vis])
    comparison = np.vstack([top_row, bottom_row])
    
    # Add final result
    final_comparison = np.hstack([comparison, 
                                 np.vstack([result, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)])])
    
    if save_path:
        cv2.imwrite(save_path, final_comparison)
        print(f"Results saved to: {save_path}")
    
    return final_comparison


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="LEI Scratch Detection")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--kernel-lengths", nargs="+", type=int, default=[11, 17, 23],
                       help="List of kernel lengths (default: 11 17 23)")
    parser.add_argument("--angle-step", type=int, default=15,
                       help="Angular step in degrees (default: 15)")
    parser.add_argument("--lei-threshold", type=float, default=0.3,
                       help="LEI response threshold (default: 0.3)")
    parser.add_argument("--min-aspect-ratio", type=float, default=3.0,
                       help="Minimum aspect ratio for scratches (default: 3.0)")
    parser.add_argument("--combination", choices=["union", "intersection", "weighted"],
                       default="union", help="Method to combine LEI and Hough (default: union)")
    parser.add_argument("--output", "-o", help="Output path for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    try:
        # Load and preprocess image
        logging.info(f"Loading image: {args.input_image}")
        image = load_and_preprocess_image(args.input_image)
        logging.info(f"Image loaded successfully: {image.shape}")
        
        # Apply LEI detection
        logging.info("Applying LEI scratch detection")
        lei_response = lei_scratch_detection(
            image,
            kernel_lengths=args.kernel_lengths,
            angle_step_deg=args.angle_step
        )
        
        # Apply Hough line detection
        logging.info("Applying Hough line detection")
        hough_mask = advanced_scratch_detection(image)
        
        # Combine results
        logging.info(f"Combining results using {args.combination} method")
        combined_mask = combine_scratch_detections(
            lei_response, 
            hough_mask, 
            lei_threshold=args.lei_threshold,
            combination_method=args.combination
        )
        
        # Classify scratches
        logging.info("Classifying detected features")
        scratches, classified_mask = classify_scratches(
            combined_mask,
            min_aspect_ratio=args.min_aspect_ratio
        )
        
        # Print results
        logging.info(f"Detection complete:")
        logging.info(f"  Total features detected: {np.sum(combined_mask > 0)} pixels")
        logging.info(f"  Classified scratches: {len(scratches)}")
        
        for i, scratch in enumerate(scratches):
            logging.info(f"    Scratch {i+1}: length={scratch['length']:.1f}px, "
                        f"AR={scratch['aspect_ratio']:.1f}, angle={scratch['angle']:.1f}°")
        
        # Visualize and save results
        if args.output:
            comparison = visualize_scratch_results(
                image, lei_response, hough_mask, combined_mask, scratches, args.output
            )
            logging.info(f"Results saved to: {args.output}")
        else:
            # Display results
            comparison = visualize_scratch_results(
                image, lei_response, hough_mask, combined_mask, scratches
            )
            cv2.imshow("Scratch Detection Results", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
