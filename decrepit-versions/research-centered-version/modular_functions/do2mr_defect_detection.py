#!/usr/bin/env python3
"""
DO2MR (Difference of Min-Max Ranking) Defect Detection
====================================================
Standalone implementation of the DO2MR algorithm for region-based defect detection
in fiber optic end-face images. This algorithm is particularly effective at detecting
local contrast variations that indicate surface defects.

Reference: Based on research in fiber optic inspection using morphological operations.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Union
from pathlib import Path
import argparse


def do2mr_defect_detection(
    image: np.ndarray,
    kernel_size: int = 5,
    gamma: float = 1.5,
    min_defect_area_px: int = 5,
    apply_morphology: bool = True
) -> np.ndarray:
    """
    Apply Difference of Min-Max Ranking (DO2MR) algorithm for defect detection.
    
    This algorithm applies morphological erosion and dilation to find local
    minimum and maximum values, then computes the difference to highlight
    regions with high local contrast variations (defects).
    
    Args:
        image: Input grayscale image (uint8)
        kernel_size: Size of morphological kernel (must be odd)
        gamma: Threshold multiplier for statistical thresholding
        min_defect_area_px: Minimum area in pixels for valid defects
        apply_morphology: Whether to apply post-processing morphological operations
        
    Returns:
        Binary mask (0/255) where 255 indicates detected defects
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
        
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")
        
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd and >= 3")
    
    # Ensure image is grayscale uint8
    if image.dtype != np.uint8:
        # Normalize to 0-255 range and convert to uint8
        normalized = np.zeros_like(image, dtype=np.uint8)
        cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX)
        image = normalized
    
    logging.debug(f"Applying DO2MR with kernel_size={kernel_size}, gamma={gamma}")
    
    # Create rectangular structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply erosion to find minimum values in local neighborhoods
    min_filtered = cv2.erode(image, kernel)
    
    # Apply dilation to find maximum values in local neighborhoods
    max_filtered = cv2.dilate(image, kernel)
    
    # Calculate residual image showing local contrast variations
    residual = cv2.subtract(max_filtered, min_filtered)
    
    # Statistical thresholding
    # Extract only non-zero values from residual for statistics calculation
    zone_vals = residual[residual > 0]
    
    # Check if any values exist in the zone to avoid division by zero
    if zone_vals.size == 0:
        logging.warning("No contrast variations found in image")
        return np.zeros_like(image, dtype=np.uint8)
    
    # Calculate statistics for adaptive thresholding
    mean_res = np.mean(zone_vals.astype(np.float32))
    std_res = np.std(zone_vals.astype(np.float32))
    
    # Initialize binary mask for defect regions
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Apply statistical threshold: pixels with values > mean + gamma*std are defects
    threshold_val = mean_res + (gamma * std_res)
    mask[residual.astype(np.float32) > threshold_val] = 255
    
    if apply_morphology:
        # Apply median blur to remove salt-and-pepper noise from binary mask
        mask = cv2.medianBlur(mask, 3)
        
        # Apply morphological opening to remove small isolated noise regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Remove small components below minimum area threshold
        if min_defect_area_px > 0:
            # Find connected components
            num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8, ltype=cv2.CV_32S
            )
            
            # Filter out small components
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_defect_area_px:
                    mask[labels_img == i] = 0
    
    logging.debug(f"DO2MR detected {np.sum(mask > 0)} defect pixels")
    return mask


def multiscale_do2mr_detection(
    image: np.ndarray,
    scales: list = [0.5, 0.75, 1.0, 1.25, 1.5],
    base_kernel_size: int = 5,
    gamma: float = 1.5
) -> np.ndarray:
    """
    Apply DO2MR detection at multiple scales for enhanced detection.
    
    Args:
        image: Input grayscale image
        scales: List of scale factors to apply
        base_kernel_size: Base kernel size (will be scaled)
        gamma: Threshold multiplier
        
    Returns:
        Combined binary mask from all scales
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    
    # Initialize accumulator for combining detections across scales
    combined_mask = np.zeros_like(image, dtype=np.uint8)
    
    for scale in scales:
        if scale <= 0:
            continue
            
        # Scale kernel size
        kernel_size = max(3, int(base_kernel_size * scale))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd
            
        logging.debug(f"Processing scale {scale} with kernel_size {kernel_size}")
        
        # Apply DO2MR at current scale
        scale_mask = do2mr_defect_detection(
            image, 
            kernel_size=kernel_size,
            gamma=gamma,
            apply_morphology=False  # Apply morphology only at the end
        )
        
        # Combine with previous results using logical OR
        combined_mask = cv2.bitwise_or(combined_mask, scale_mask)
    
    # Apply final morphological cleanup
    combined_mask = cv2.medianBlur(combined_mask, 3)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    
    return combined_mask


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for DO2MR detection."""
    image_path_obj = Path(image_path)
    
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path_obj), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Apply Gaussian blur for noise reduction
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return denoised


def visualize_results(original: np.ndarray, mask: np.ndarray, save_path: Optional[str] = None):
    """Visualize DO2MR detection results."""
    # Create result visualization
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Overlay defects in red
    result[mask > 0] = [0, 0, 255]  # Red for defects
    
    # Create side-by-side comparison
    comparison = np.hstack([
        cv2.cvtColor(original, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        result
    ])
    
    if save_path:
        cv2.imwrite(save_path, comparison)
        print(f"Results saved to: {save_path}")
    
    return comparison


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="DO2MR Defect Detection")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--kernel-size", type=int, default=5, 
                       help="Morphological kernel size (default: 5)")
    parser.add_argument("--gamma", type=float, default=1.5,
                       help="Threshold multiplier (default: 1.5)")
    parser.add_argument("--multiscale", action="store_true",
                       help="Use multiscale detection")
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
        
        # Apply DO2MR detection
        if args.multiscale:
            logging.info("Applying multiscale DO2MR detection")
            mask = multiscale_do2mr_detection(image, gamma=args.gamma)
        else:
            logging.info("Applying single-scale DO2MR detection")
            mask = do2mr_defect_detection(
                image, 
                kernel_size=args.kernel_size,
                gamma=args.gamma
            )
        
        # Count defects
        defect_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        defect_percentage = (defect_pixels / total_pixels) * 100
        
        logging.info(f"Detection complete:")
        logging.info(f"  Defect pixels: {defect_pixels}")
        logging.info(f"  Total pixels: {total_pixels}")
        logging.info(f"  Defect coverage: {defect_percentage:.2f}%")
        
        # Visualize and save results
        if args.output:
            comparison = visualize_results(image, mask, args.output)
            logging.info(f"Results saved to: {args.output}")
        else:
            # Display results using OpenCV
            comparison = visualize_results(image, mask)
            cv2.imshow("DO2MR Results (Original | Mask | Overlay)", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
