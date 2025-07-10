#!/usr/bin/env python3
"""
LEI Scratch Detection Module
============================
Standalone module for LEI (Linear Enhancement Inspector) scratch detection.
Extracted from the Advanced Fiber Optic End Face Defect Detection System.

Author: Modularized by AI
Date: July 9, 2025
Version: 1.0
"""

import cv2
import numpy as np
from typing import List, Optional
import argparse
import sys
from pathlib import Path


class LEIScratchDetector:
    """
    A class for LEI (Linear Enhancement Inspector) scratch detection.
    This method detects linear features and scratches using directional filtering.
    """
    
    def __init__(self,
                 kernel_lengths: List[int] = [11, 17, 23],
                 angle_step: int = 15,
                 threshold_factor: float = 2.0,
                 morph_close_kernel_size: tuple = (5, 5),
                 min_scratch_area: int = 15):
        """
        Initialize the LEI scratch detector with configuration parameters.
        
        Args:
            kernel_lengths: List of linear kernel lengths for multi-scale detection
            angle_step: Angular resolution for rotation (in degrees)
            threshold_factor: Factor for thresholding response map
            morph_close_kernel_size: Kernel size for morphological closing
            min_scratch_area: Minimum area for valid scratch detection
        """
        self.kernel_lengths = kernel_lengths
        self.angle_step = angle_step
        self.threshold_factor = threshold_factor
        self.morph_close_kernel_size = morph_close_kernel_size
        self.min_scratch_area = min_scratch_area
    
    def detect_scratches(self, image: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        region_name: str = "image") -> Optional[np.ndarray]:
        """
        Detects linear scratches using LEI method.
        
        Args:
            image: Input grayscale image
            mask: Optional binary mask to restrict detection area
            region_name: Name of the region being processed (for logging)
            
        Returns:
            Binary mask of detected scratches, or None if error occurs
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
        
        print(f"INFO: Starting LEI scratch detection for region '{region_name}'")
        
        # Enhance contrast with histogram equalization
        enhanced_image = cv2.equalizeHist(masked_image)
        enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=mask)
        
        H, W = gray.shape
        max_response_map = np.zeros((H, W), dtype=np.float32)
        
        # Iterate over different kernel lengths for multi-scale detection
        for kernel_length in self.kernel_lengths:
            print(f"INFO: Processing with kernel length {kernel_length}")
            
            # Iterate through angles from 0 to 180 degrees
            for angle_deg in range(0, 180, self.angle_step):
                # Create linear structuring element
                line_kernel = np.zeros((kernel_length, kernel_length), dtype=np.uint8)
                
                # Create a line through the center
                center = kernel_length // 2
                if angle_deg == 0:  # Horizontal line
                    line_kernel[center, :] = 1
                elif angle_deg == 90:  # Vertical line
                    line_kernel[:, center] = 1
                else:  # Diagonal line
                    # Calculate line endpoints
                    angle_rad = np.radians(angle_deg)
                    for i in range(kernel_length):
                        for j in range(kernel_length):
                            # Check if point is on the line
                            dx = j - center
                            dy = i - center
                            # Distance from point to line through center with given angle
                            dist = abs(dx * np.sin(angle_rad) - dy * np.cos(angle_rad))
                            if dist < 0.5:  # Tolerance for discrete pixels
                                line_kernel[i, j] = 1
                
                # Normalize kernel
                if np.sum(line_kernel) > 0:
                    line_kernel_float = line_kernel.astype(np.float32) / np.sum(line_kernel)
                else:
                    continue
                
                # Apply convolution
                try:
                    response = cv2.filter2D(enhanced_image.astype(np.float32), 
                                          cv2.CV_32F, line_kernel_float, 
                                          borderType=cv2.BORDER_REPLICATE)
                    # Update maximum response
                    max_response_map = np.maximum(max_response_map, response)
                except Exception as e:
                    print(f"WARNING: Filter failed for length={kernel_length}, "
                         f"angle={angle_deg}: {e}")
                    continue
        
        # Normalize response map to 0-255 range
        if np.max(max_response_map) > 0:
            cv2.normalize(max_response_map, max_response_map, 0, 255, cv2.NORM_MINMAX)
        response_8u = max_response_map.astype(np.uint8)
        
        # Threshold using adaptive method
        zone_vals = response_8u[mask > 0]
        if zone_vals.size == 0:
            print(f"WARNING: Empty mask for region '{region_name}'")
            return None
        
        mean_val = float(np.mean(zone_vals))
        std_val = float(np.std(zone_vals))
        threshold_val = mean_val + self.threshold_factor * std_val
        threshold_val = float(np.clip(threshold_val, 0, 255))
        
        _, scratch_mask = cv2.threshold(response_8u, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Apply morphological closing to connect broken scratch segments
        if (self.morph_close_kernel_size[0] > 0 and 
            self.morph_close_kernel_size[1] > 0):
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                   self.morph_close_kernel_size)
            scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # Restrict to mask region
        scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=mask)
        
        # Filter by minimum area
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
            scratch_mask, connectivity=8)
        final_mask = np.zeros_like(scratch_mask)
        
        valid_scratches = 0
        for i in range(1, num_components):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_scratch_area:
                final_mask[labels == i] = 255
                valid_scratches += 1
        
        print(f"INFO: LEI detection complete. Found {valid_scratches} valid scratches")
        
        return final_mask
    
    def visualize_detection(self, original_image: np.ndarray,
                          scratch_mask: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualizes the detected scratches on the original image.
        
        Args:
            original_image: Original input image
            scratch_mask: Binary mask of detected scratches
            mask: Optional region mask
            
        Returns:
            Image with scratches highlighted
        """
        # Convert to color if grayscale
        if len(original_image.shape) == 2:
            vis_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = original_image.copy()
        
        # Highlight scratches in magenta
        if scratch_mask is not None:
            vis_img[scratch_mask > 0] = [255, 0, 255]  # Magenta color for scratches
        
        # Draw region boundary if mask provided
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)  # Green boundary
        
        return vis_img


def main():
    """
    Main function for standalone execution.
    """
    parser = argparse.ArgumentParser(description="LEI Scratch Detection Module")
    parser.add_argument("input_path", help="Path to input image")
    parser.add_argument("--output_dir", default="lei_output",
                       help="Output directory")
    parser.add_argument("--mask_path", help="Path to region mask image (optional)")
    parser.add_argument("--kernel_lengths", type=int, nargs='+', default=[11, 17, 23],
                       help="Kernel lengths for linear detection")
    parser.add_argument("--angle_step", type=int, default=15,
                       help="Angle step in degrees")
    parser.add_argument("--threshold_factor", type=float, default=2.0,
                       help="Threshold factor for response map")
    parser.add_argument("--min_scratch_area", type=int, default=15,
                       help="Minimum area for valid scratch")
    
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
    detector = LEIScratchDetector(
        kernel_lengths=args.kernel_lengths,
        angle_step=args.angle_step,
        threshold_factor=args.threshold_factor,
        min_scratch_area=args.min_scratch_area
    )
    
    # Detect scratches
    print("INFO: Starting LEI scratch detection...")
    scratch_mask = detector.detect_scratches(image, mask, "input_region")
    
    if scratch_mask is None:
        print("ERROR: Scratch detection failed")
        sys.exit(1)
    
    # Visualize results
    vis_img = detector.visualize_detection(image, scratch_mask, mask)
    
    # Save outputs
    base_name = image_path.stem
    
    # Save scratch mask
    mask_output_path = output_dir / f"{base_name}_lei_scratches.jpg"
    cv2.imwrite(str(mask_output_path), scratch_mask)
    print(f"INFO: Scratch mask saved to {mask_output_path}")
    
    # Save visualization
    vis_output_path = output_dir / f"{base_name}_lei_visualization.jpg"
    cv2.imwrite(str(vis_output_path), vis_img)
    print(f"INFO: Visualization saved to {vis_output_path}")
    
    # Count and report scratches
    num_components, labels = cv2.connectedComponents(scratch_mask)
    num_scratches = num_components - 1  # Subtract background
    
    # Calculate scratch statistics
    if num_scratches > 0:
        contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum(cv2.contourArea(c) for c in contours)
        avg_area = total_area / num_scratches if num_scratches > 0 else 0
    else:
        total_area = 0
        avg_area = 0
    
    stats_path = output_dir / f"{base_name}_lei_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"LEI Scratch Detection Results\n")
        f.write(f"=============================\n")
        f.write(f"Input image: {image_path}\n")
        f.write(f"Number of detected scratches: {num_scratches}\n")
        f.write(f"Total scratch area: {total_area:.1f} pixels\n")
        f.write(f"Average scratch area: {avg_area:.1f} pixels\n")
        f.write(f"Kernel lengths used: {args.kernel_lengths}\n")
        f.write(f"Angle step: {args.angle_step} degrees\n")
        f.write(f"Threshold factor: {args.threshold_factor}\n")
        f.write(f"Minimum scratch area: {args.min_scratch_area}\n")
    
    print(f"INFO: Statistics saved to {stats_path}")
    print(f"INFO: LEI detection complete. Found {num_scratches} scratches.")


if __name__ == "__main__":
    main()
