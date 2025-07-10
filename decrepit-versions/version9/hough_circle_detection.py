#!/usr/bin/env python3
"""
Hough Circle Detection Module
=============================
Standalone module for robust circular feature detection using Hough Circle Transform.
Extracted from the Advanced Fiber Optic End Face Defect Detection System.

Author: Modularized by AI
Date: July 9, 2025
Version: 1.0
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import argparse
import sys
from pathlib import Path


class HoughCircleDetector:
    """
    A class for robust circle detection using Hough Circle Transform.
    """
    
    def __init__(self,
                 dp_values: List[float] = [1.0, 1.2, 1.5],
                 min_dist_factor: float = 0.1,
                 param1_values: List[int] = [50, 70, 100],
                 param2_values: List[int] = [25, 30, 40],
                 min_radius_factor: float = 0.05,
                 max_radius_factor: float = 0.6,
                 confidence_threshold: float = 0.3):
        """
        Initialize the circle detector with configuration parameters.
        
        Args:
            dp_values: List of inverse ratio of accumulator resolution values
            min_dist_factor: Minimum distance factor between circle centers
            param1_values: List of upper Canny thresholds
            param2_values: List of accumulator thresholds
            min_radius_factor: Minimum radius factor relative to image size
            max_radius_factor: Maximum radius factor relative to image size
            confidence_threshold: Minimum confidence for valid circle
        """
        self.dp_values = dp_values
        self.min_dist_factor = min_dist_factor
        self.param1_values = param1_values
        self.param2_values = param2_values
        self.min_radius_factor = min_radius_factor
        self.max_radius_factor = max_radius_factor
        self.confidence_threshold = confidence_threshold
    
    def detect_circles(self, image: np.ndarray, 
                      processed_images: Optional[Dict[str, np.ndarray]] = None) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Detects the primary circular feature using Hough Circle Transform.
        
        Args:
            image: Input grayscale image
            processed_images: Optional dict of preprocessed images for multi-stage detection
            
        Returns:
            Tuple of ((center_x, center_y), radius) or None if no reliable circle found
        """
        if image is None:
            print("ERROR: Input image is None")
            return None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        h, w = gray.shape[:2]
        min_dist = int(min(h, w) * self.min_dist_factor)
        min_r = int(min(h, w) * self.min_radius_factor)
        max_r = int(min(h, w) * self.max_radius_factor)
        
        print(f"INFO: Image size: {w}x{h}, radius range: {min_r}-{max_r}")
        
        candidates = []
        
        # Use processed images if available, otherwise just the input
        images_to_test = {}
        if processed_images:
            images_to_test = {k: v for k, v in processed_images.items() 
                            if k in ['gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced']}
        
        if not images_to_test:
            images_to_test = {'original': gray}
        
        # Try each processing stage
        for key, img in images_to_test.items():
            if img is None:
                continue
                
            print(f"INFO: Testing Hough circles on {key} image")
            
            for dp in self.dp_values:
                for param1 in self.param1_values:
                    for param2 in self.param2_values:
                        try:
                            circles = cv2.HoughCircles(
                                img, cv2.HOUGH_GRADIENT,
                                dp=dp,
                                minDist=min_dist,
                                param1=param1,
                                param2=param2,
                                minRadius=min_r,
                                maxRadius=max_r
                            )
                            
                            if circles is not None:
                                # Handle different OpenCV versions - circles can be 3D array
                                if len(circles.shape) == 3:
                                    circles = np.round(circles[0, :]).astype("int")
                                else:
                                    circles = np.round(circles).astype("int")
                                for c in circles:
                                    cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                                    # Score based on distance from center and radius
                                    dist_center = np.sqrt((cx - w//2)**2 + (cy - h//2)**2)
                                    norm_r = r / max_r if max_r > 0 else 0
                                    confidence = (param2 / max(self.param2_values)) * 0.5 \
                                               + 0.5 * norm_r \
                                               - 0.2 * (dist_center/(min(h,w)/2))
                                    confidence = max(0.0, min(1.0, confidence))
                                    candidates.append((cx, cy, r, confidence, key))
                                    
                        except Exception as e:
                            print(f"WARNING: HoughCircles error on {key} "
                                 f"(dp={dp},p1={param1},p2={param2}): {e}")
        
        if not candidates:
            print("WARNING: No circles found by HoughCircles")
            return None
        
        # Select best candidate
        candidates.sort(key=lambda x: x[3], reverse=True)
        best_cx, best_cy, best_r, best_conf, best_key = candidates[0]
        
        if best_conf < self.confidence_threshold:
            print(f"WARNING: Best circle confidence ({best_conf:.2f}) below threshold "
                 f"({self.confidence_threshold})")
            return None
        
        print(f"INFO: Selected circle at ({best_cx},{best_cy}) radius={best_r}px, "
              f"confidence={best_conf:.2f}, from '{best_key}'")
        
        return (best_cx, best_cy), float(best_r)
    
    def visualize_detection(self, image: np.ndarray, 
                          circle_result: Optional[Tuple[Tuple[int, int], float]]) -> np.ndarray:
        """
        Visualizes the detected circle on the image.
        
        Args:
            image: Input image
            circle_result: Result from detect_circles
            
        Returns:
            Image with circle drawn on it
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = image.copy()
            
        if circle_result is not None:
            (cx, cy), radius = circle_result
            # Draw the circle
            cv2.circle(vis_img, (cx, cy), int(radius), (0, 255, 0), 2)
            # Draw the center
            cv2.circle(vis_img, (cx, cy), 2, (0, 0, 255), 3)
            # Add text
            cv2.putText(vis_img, f"R={int(radius)}px", (cx-50, cy-int(radius)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_img


def main():
    """
    Main function for standalone execution.
    """
    parser = argparse.ArgumentParser(description="Hough Circle Detection Module")
    parser.add_argument("input_path", help="Path to input image")
    parser.add_argument("--output_dir", default="circle_detection_output",
                       help="Output directory")
    parser.add_argument("--min_radius_factor", type=float, default=0.05,
                       help="Minimum radius factor (0-1)")
    parser.add_argument("--max_radius_factor", type=float, default=0.6,
                       help="Maximum radius factor (0-1)")
    parser.add_argument("--confidence_threshold", type=float, default=0.3,
                       help="Confidence threshold (0-1)")
    parser.add_argument("--preprocess", action="store_true",
                       help="Apply preprocessing before detection")
    
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = HoughCircleDetector(
        min_radius_factor=args.min_radius_factor,
        max_radius_factor=args.max_radius_factor,
        confidence_threshold=args.confidence_threshold
    )
    
    # Apply preprocessing if requested
    processed_images = None
    if args.preprocess:
        print("INFO: Applying preprocessing...")
        # Simple preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        processed_images = {
            'gaussian_blurred': cv2.GaussianBlur(gray, (7, 7), 2),
            'bilateral_filtered': cv2.bilateralFilter(gray, 9, 75, 75),
            'clahe_enhanced': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        }
    
    # Detect circles
    print("INFO: Starting circle detection...")
    result = detector.detect_circles(image, processed_images)
    
    if result is None:
        print("WARNING: No circles detected")
    else:
        (cx, cy), radius = result
        print(f"INFO: Detected circle - Center: ({cx}, {cy}), Radius: {radius:.1f}px")
    
    # Visualize result
    vis_img = detector.visualize_detection(image, result)
    
    # Save output
    base_name = image_path.stem
    output_path = output_dir / f"{base_name}_circle_detection.jpg"
    cv2.imwrite(str(output_path), vis_img)
    print(f"INFO: Visualization saved to {output_path}")
    
    # Save circle parameters
    if result is not None:
        (cx, cy), radius = result
        params_path = output_dir / f"{base_name}_circle_params.txt"
        with open(params_path, 'w') as f:
            f.write(f"Center: ({cx}, {cy})\n")
            f.write(f"Radius: {radius:.1f} pixels\n")
        print(f"INFO: Circle parameters saved to {params_path}")


if __name__ == "__main__":
    main()
