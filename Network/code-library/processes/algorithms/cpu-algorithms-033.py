#!/usr/bin/env python3
"""
DO2MR Defect Detection Module
=============================
Standalone module for DO2MR (Difference of Min-Max Ranking) region-based defect detection.
Extracted from the Advanced Fiber Optic End Face Defect Detection System.

Author: Modularized by AI
Date: July 9, 2025
Version: 1.0
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import argparse
import sys
from pathlib import Path


class DO2MRDetector:
    """
    A class for DO2MR (Difference of Min-Max Ranking) defect detection.
    This method detects region-based defects like dirt, pits, and contamination.
    """
    
    def __init__(self,
                 kernel_sizes: List[Tuple[int, int]] = [(5, 5), (9, 9), (13, 13)],
                 gamma_values: List[float] = [2.0, 2.5, 3.0],
                 median_blur_kernel_size: int = 5,
                 morph_open_kernel_size: Tuple[int, int] = (3, 3),
                 min_votes_ratio: float = 0.3):
        """
        Initialize the DO2MR detector with configuration parameters.
        
        Args:
            kernel_sizes: List of structuring element sizes for min/max filtering
            gamma_values: List of sensitivity parameters for thresholding
            median_blur_kernel_size: Kernel size for median blur on residual
            morph_open_kernel_size: Kernel for morphological opening
            min_votes_ratio: Minimum ratio of votes required for defect confirmation
        """
        self.kernel_sizes = kernel_sizes
        self.gamma_values = gamma_values
        self.median_blur_kernel_size = median_blur_kernel_size
        self.morph_open_kernel_size = morph_open_kernel_size
        self.min_votes_ratio = min_votes_ratio
    
    def detect_defects(self, image: np.ndarray, 
                      mask: Optional[np.ndarray] = None,
                      region_name: str = "image") -> Optional[np.ndarray]:
        """
        Detects region-based defects using DO2MR method.
        
        Args:
            image: Input grayscale image
            mask: Optional binary mask to restrict detection area
            region_name: Name of the region being processed (for logging)
            
        Returns:
            Binary mask of detected defects, or None if error occurs
        """
        if image is None:
            print("ERROR: Input image is None")
            return None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply mask if provided
        if mask is not None:
            masked_image = cv2.bitwise_and(gray, gray, mask=mask)
        else:
            masked_image = gray
            mask = np.ones_like(gray, dtype=np.uint8) * 255
        
        print(f"INFO: Starting DO2MR detection for region '{region_name}'")
        
        H, W = gray.shape
        vote_map = np.zeros((H, W), dtype=np.float32)
        total_passes = 0
        
        # Iterate over configured kernel sizes
        for kernel_size in self.kernel_sizes:
            print(f"INFO: Processing with kernel size {kernel_size}")
            
            # Create structuring element
            struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            
            # Apply minimum filter (erosion) and maximum filter (dilation)
            min_filtered = cv2.erode(masked_image, struct_element)
            max_filtered = cv2.dilate(masked_image, struct_element)
            
            # Calculate residual image (difference between max and min)
            residual = cv2.subtract(max_filtered, min_filtered)
            
            # Apply median blur to reduce noise
            if self.median_blur_kernel_size > 0:
                residual = cv2.medianBlur(residual, self.median_blur_kernel_size)
            
            # Iterate over gamma values for thresholding
            for gamma in self.gamma_values:
                # Calculate threshold using mean and std of residual within mask
                masked_residual_values = residual[mask > 0]
                if masked_residual_values.size == 0:
                    print(f"WARNING: Empty mask for kernel={kernel_size}, gamma={gamma}")
                    continue
                
                mean_val = float(np.mean(masked_residual_values))
                std_val = float(np.std(masked_residual_values))
                threshold_val = np.clip(mean_val + gamma * std_val, 0, 255)
                
                # Apply threshold to create binary defect mask
                _, defect_mask = cv2.threshold(residual, threshold_val, 255, cv2.THRESH_BINARY)
                defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=mask)
                
                # Apply morphological opening to remove noise
                if (self.morph_open_kernel_size[0] > 0 and 
                    self.morph_open_kernel_size[1] > 0):
                    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                           self.morph_open_kernel_size)
                    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, open_kernel)
                
                # Accumulate votes
                vote_map += (defect_mask.astype(np.float32) / 255.0)
                total_passes += 1
        
        if total_passes == 0:
            print(f"WARNING: No passes executed for region '{region_name}'")
            return None
        
        # Final defect map based on votes
        min_votes_required = max(1, int(total_passes * self.min_votes_ratio))
        final_mask = (vote_map >= min_votes_required).astype(np.uint8) * 255
        
        num_defects = cv2.connectedComponents(final_mask)[0] - 1
        print(f"INFO: DO2MR detection complete. Found {num_defects} potential defects")
        
        return final_mask
    
    def visualize_detection(self, original_image: np.ndarray,
                          defect_mask: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualizes the detected defects on the original image.
        
        Args:
            original_image: Original input image
            defect_mask: Binary mask of detected defects
            mask: Optional region mask
            
        Returns:
            Image with defects highlighted
        """
        # Convert to color if grayscale
        if len(original_image.shape) == 2:
            vis_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = original_image.copy()
        
        # Highlight defects in red
        if defect_mask is not None:
            vis_img[defect_mask > 0] = [0, 0, 255]  # Red color for defects
        
        # Draw region boundary if mask provided
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)  # Green boundary
        
        return vis_img


def main():
    """
    Main function for standalone execution.
    """
    parser = argparse.ArgumentParser(description="DO2MR Defect Detection Module")
    parser.add_argument("input_path", help="Path to input image")
    parser.add_argument("--output_dir", default="do2mr_output",
                       help="Output directory")
    parser.add_argument("--mask_path", help="Path to region mask image (optional)")
    parser.add_argument("--gamma", type=float, nargs='+', default=[2.0, 2.5, 3.0],
                       help="Gamma values for thresholding")
    parser.add_argument("--kernel_sizes", type=int, nargs='+', default=[5, 9, 13],
                       help="Kernel sizes for morphological operations")
    parser.add_argument("--min_votes_ratio", type=float, default=0.3,
                       help="Minimum votes ratio for defect confirmation")
    
    args = parser.parse_args()
    
    # Load input image
    image_path = Path(args.input_path)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)
        
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        sys.exit(1)
        
    print(f"INFO: Loaded image {image_path} with shape {image.shape}")
    
    # Load mask if provided
    mask = None
    if args.mask_path:
        mask_path = Path(args.mask_path)
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            print(f"INFO: Loaded mask {mask_path}")
        else:
            print(f"WARNING: Mask file not found: {mask_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    kernel_sizes = [(k, k) for k in args.kernel_sizes]
    detector = DO2MRDetector(
        kernel_sizes=kernel_sizes,
        gamma_values=args.gamma,
        min_votes_ratio=args.min_votes_ratio
    )
    
    # Detect defects
    print("INFO: Starting DO2MR defect detection...")
    defect_mask = detector.detect_defects(image, mask, "input_region")
    
    if defect_mask is None:
        print("ERROR: Defect detection failed")
        sys.exit(1)
    
    # Visualize results
    vis_img = detector.visualize_detection(image, defect_mask, mask)
    
    # Save outputs
    base_name = image_path.stem
    
    # Save defect mask
    mask_output_path = output_dir / f"{base_name}_do2mr_defects.jpg"
    cv2.imwrite(str(mask_output_path), defect_mask)
    print(f"INFO: Defect mask saved to {mask_output_path}")
    
    # Save visualization
    vis_output_path = output_dir / f"{base_name}_do2mr_visualization.jpg"
    cv2.imwrite(str(vis_output_path), vis_img)
    print(f"INFO: Visualization saved to {vis_output_path}")
    
    # Count and report defects
    num_components, labels = cv2.connectedComponents(defect_mask)
    num_defects = num_components - 1  # Subtract background
    
    stats_path = output_dir / f"{base_name}_do2mr_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"DO2MR Defect Detection Results\n")
        f.write(f"==============================\n")
        f.write(f"Input image: {image_path}\n")
        f.write(f"Number of detected defects: {num_defects}\n")
        f.write(f"Kernel sizes used: {args.kernel_sizes}\n")
        f.write(f"Gamma values used: {args.gamma}\n")
        f.write(f"Minimum votes ratio: {args.min_votes_ratio}\n")
    
    print(f"INFO: Statistics saved to {stats_path}")
    print(f"INFO: DO2MR detection complete. Found {num_defects} defects.")


if __name__ == "__main__":
    main()
