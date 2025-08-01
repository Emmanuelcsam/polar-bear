#!/usr/bin/env python3
"""
Bright Core Extractor - Standalone Module
Extracted from fiber optic defect detection system
Detects bright fiber cores using local contrast validation
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class BrightCoreExtractor:
    """
    Performs bright core detection on fiber optic images using local contrast validation.
    """
    
    # Detection parameters
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    MEDIAN_BLUR_KERNEL = 5
    HOUGH_DP = 1
    HOUGH_MIN_DIST = 100
    HOUGH_PARAM1 = 150
    HOUGH_PARAM2 = 25
    MIN_RADIUS = 15
    MAX_RADIUS = 150
    
    # Local contrast validation parameters
    LOCAL_CONTRAST_FACTOR = 1.15  # Inner region must be this much brighter
    OUTER_RING_WIDTH = 15  # Width of outer ring for comparison
    
    def __init__(self, image_path: str, debug_mode: bool = False):
        """
        Initialize the bright core extractor.
        
        Args:
            image_path (str): Path to the input image
            debug_mode (bool): Whether to save debug images
        """
        self.image_path = Path(image_path)
        self.debug_mode = debug_mode
        
        # Load image
        self.image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise IOError(f"Could not read image at {self.image_path}")
        
        self.output_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.center = None
        self.radius = None
        self.segmented_mask = None
        self.results = {}
    
    def detect_bright_core(self) -> Dict[str, Any]:
        """
        Execute the bright core detection pipeline.
        
        Returns:
            dict: Detection results with center, radius, and confidence
        """
        result = {
            'method': 'bright_core_extractor',
            'image_path': str(self.image_path),
            'success': False,
            'center': None,
            'core_radius': None,
            'cladding_radius': None,
            'confidence': 0.0
        }
        
        try:
            print(f"Processing {self.image_path.name}...")
            
            # Preprocess image
            processed_image = cv2.GaussianBlur(self.image, self.GAUSSIAN_BLUR_KERNEL, 0)
            processed_image = cv2.medianBlur(processed_image, self.MEDIAN_BLUR_KERNEL)
            
            # Find circles using Hough transform
            circles = self._find_circles(processed_image)
            if circles is None:
                result['error'] = 'No circles detected'
                return result
            
            # Validate circles using local contrast
            for circle in circles:
                is_valid, debug_data = self._validate_with_local_contrast(circle)
                if is_valid:
                    print(f"Validated circle via local contrast at {self.center}, R={self.radius}px.")
                    
                    # Create precise mask and analyze
                    self._create_precise_mask()
                    self._analyze_final_segment()
                    
                    # Estimate cladding radius (typically 2.5-3x core radius)
                    estimated_cladding_radius = int(self.radius * 2.5) if self.radius else None
                    
                    result.update({
                        'success': True,
                        'center': self.center,
                        'core_radius': self.radius,
                        'cladding_radius': estimated_cladding_radius,
                        'confidence': self._calculate_confidence(debug_data),
                        'analysis_results': self.results
                    })
                    
                    return result
                else:
                    # Save debug image if validation fails
                    if self.debug_mode:
                        self._save_debug_image(circle, debug_data)
            
            # No valid circles found
            result['error'] = f'Found {len(circles)} circles, but none passed validation'
            
        except Exception as e:
            result['error'] = f'Detection failed: {str(e)}'
        
        return result
    
    def _find_circles(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect circles using Circular Hough Transform.
        
        Args:
            image (np.ndarray): Preprocessed grayscale image
            
        Returns:
            np.ndarray or None: Array of detected circles or None if none found
        """
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, 
            dp=self.HOUGH_DP, 
            minDist=self.HOUGH_MIN_DIST,
            param1=self.HOUGH_PARAM1, 
            param2=self.HOUGH_PARAM2,
            minRadius=self.MIN_RADIUS, 
            maxRadius=self.MAX_RADIUS
        )
        
        return np.uint16(np.around(circles[0, :])) if circles is not None else None
    
    def _validate_with_local_contrast(self, circle: np.ndarray) -> Tuple[bool, Dict]:
        """
        Validate a circle by comparing internal brightness to outer surroundings.
        
        Args:
            circle (np.ndarray): Circle parameters [x, y, radius]
            
        Returns:
            tuple: (is_valid, debug_data)
        """
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        
        # Create inner mask (the core)
        inner_mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(inner_mask, (x, y), r, 255, -1)
        
        # Create outer ring mask (around the core)
        outer_mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), r + self.OUTER_RING_WIDTH, 255, -1)
        outer_mask = cv2.subtract(outer_mask, inner_mask)
        
        # Calculate mean intensities
        inner_pixels = self.image[inner_mask > 0]
        outer_pixels = self.image[outer_mask > 0]
        
        if len(inner_pixels) == 0 or len(outer_pixels) == 0:
            return False, {}
        
        inner_mean = np.mean(inner_pixels)
        outer_mean = np.mean(outer_pixels)
        
        debug_data = {
            'inner_mean': inner_mean,
            'outer_mean': outer_mean,
            'contrast_ratio': inner_mean / outer_mean if outer_mean > 0 else 0,
            'inner_std': np.std(inner_pixels),
            'outer_std': np.std(outer_pixels),
            'inner_mask': inner_mask,
            'outer_mask': outer_mask
        }
        
        # Validate: inner region must be sufficiently brighter
        if inner_mean > outer_mean * self.LOCAL_CONTRAST_FACTOR:
            self.center = (x, y)
            self.radius = r
            return True, debug_data
        
        return False, debug_data
    
    def _calculate_confidence(self, debug_data: Dict) -> float:
        """
        Calculate confidence score based on contrast and consistency.
        
        Args:
            debug_data (dict): Validation debug data
            
        Returns:
            float: Confidence score between 0 and 1
        """
        contrast_ratio = debug_data.get('contrast_ratio', 0)
        inner_std = debug_data.get('inner_std', 0)
        outer_std = debug_data.get('outer_std', 0)
        
        # Contrast score (higher contrast = higher confidence)
        contrast_score = min(1.0, (contrast_ratio - 1.0) * 2)
        
        # Consistency score (lower std dev = higher confidence)
        consistency_score = max(0.0, 1.0 - (inner_std + outer_std) / 100.0)
        
        # Combined confidence
        confidence = (contrast_score * 0.7 + consistency_score * 0.3)
        return max(0.0, min(1.0, confidence))
    
    def _save_debug_image(self, circle: np.ndarray, debug_data: Dict):
        """
        Save diagnostic image when validation fails.
        
        Args:
            circle (np.ndarray): Circle parameters
            debug_data (dict): Debug information
        """
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        
        debug_img = self.output_image.copy()
        
        # Overlay masks with color tints
        inner_mask = debug_data.get('inner_mask', np.zeros_like(self.image))
        outer_mask = debug_data.get('outer_mask', np.zeros_like(self.image))
        
        # Green tint for inner region
        debug_img[inner_mask > 0] = (debug_img[inner_mask > 0] * 0.5 + 
                                    np.array([0, 128, 0], dtype=np.uint8))
        
        # Red tint for outer ring
        debug_img[outer_mask > 0] = (debug_img[outer_mask > 0] * 0.5 + 
                                    np.array([0, 0, 128], dtype=np.uint8))
        
        # Draw the rejected circle in yellow
        cv2.circle(debug_img, (x, y), r, (0, 255, 255), 2)
        
        # Add debug information text
        inner_mean = debug_data.get('inner_mean', 0)
        outer_mean = debug_data.get('outer_mean', 0)
        threshold = outer_mean * self.LOCAL_CONTRAST_FACTOR
        
        cv2.putText(debug_img, f"Inner Mean: {inner_mean:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Outer Mean: {outer_mean:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Threshold: {threshold:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Save debug image
        debug_path = self.image_path.parent / f"{self.image_path.stem}_DEBUG.png"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"Saved debug image to {debug_path}")
    
    def _create_precise_mask(self):
        """Generate precise segmentation mask using validated circle as ROI."""
        if self.center is None or self.radius is None:
            return
        
        # Create ROI mask
        roi_mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.circle(roi_mask, self.center, self.radius, 255, -1)
        
        # Apply ROI to image
        roi_pixels = cv2.bitwise_and(self.image, self.image, mask=roi_mask)
        
        # Apply Otsu thresholding within ROI
        _, self.segmented_mask = cv2.threshold(
            roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    
    def _analyze_final_segment(self):
        """Compute statistics on the final segmented fiber area."""
        if self.segmented_mask is None:
            return
        
        fiber_pixels = self.image[self.segmented_mask > 0]
        if len(fiber_pixels) == 0:
            return
        
        # Find contours
        contours, _ = cv2.findContours(
            self.segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return
        
        # Analyze largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Calculate metrics
        circularity = (4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0
        (center_x, center_y), effective_radius = cv2.minEnclosingCircle(largest_contour)
        
        self.results = {
            'center_x': int(center_x),
            'center_y': int(center_y),
            'effective_radius': float(effective_radius),
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'mean_intensity': float(np.mean(fiber_pixels)),
            'std_intensity': float(np.std(fiber_pixels)),
            'min_intensity': int(np.min(fiber_pixels)),
            'max_intensity': int(np.max(fiber_pixels))
        }
    
    def save_visualization(self, output_dir: str):
        """
        Save visualization of the detection results.
        
        Args:
            output_dir (str): Directory to save outputs
        """
        if not self.results:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Draw contours and circle on output image
        if self.segmented_mask is not None:
            contours, _ = cv2.findContours(
                self.segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(self.output_image, contours, -1, (255, 0, 0), 2)
        
        if self.center and self.radius:
            # Draw detected circle
            cv2.circle(self.output_image, self.center, self.radius, (0, 255, 0), 2)
            cv2.drawMarker(self.output_image, self.center, (0, 0, 255), 
                          cv2.MARKER_CROSS, 10, 2)
        
        # Save images
        base_name = self.image_path.stem
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_analysis.png"), 
                   self.output_image)
        
        if self.segmented_mask is not None:
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), 
                       self.segmented_mask)


def bright_core_detection(image_path, output_dir='bright_core_output', debug_mode=False):
    """
    Standalone function for bright core detection.
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory to save results
        debug_mode (bool): Whether to save debug images
        
    Returns:
        dict: Detection results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run detection
        extractor = BrightCoreExtractor(image_path, debug_mode=debug_mode)
        result = extractor.detect_bright_core()
        
        # Save visualization if successful
        if result['success']:
            extractor.save_visualization(output_dir)
        
        # Save results
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        result_path = os.path.join(output_dir, f'{base_filename}_bright_core_result.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4, cls=NumpyEncoder)
        
        return result
        
    except Exception as e:
        return {
            'method': 'bright_core_extractor',
            'image_path': image_path,
            'success': False,
            'error': f'Processing failed: {str(e)}',
            'center': None,
            'core_radius': None,
            'cladding_radius': None,
            'confidence': 0.0
        }


def main():
    """Command line interface for bright core detection"""
    parser = argparse.ArgumentParser(description='Bright Core Detection for Fiber Optic Images')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output-dir', default='bright_core_output',
                       help='Output directory (default: bright_core_output)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (saves diagnostic images)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run detection
    result = bright_core_detection(
        image_path=args.image_path,
        output_dir=args.output_dir,
        debug_mode=args.debug
    )
    
    # Print results
    if args.verbose:
        print(json.dumps(result, indent=2, cls=NumpyEncoder))
    else:
        if result['success']:
            print(f"✓ Bright core detection successful!")
            print(f"  Center: {result['center']}")
            print(f"  Core radius: {result['core_radius']}")
            print(f"  Estimated cladding radius: {result['cladding_radius']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            
            # Show analysis results if available
            analysis = result.get('analysis_results', {})
            if analysis:
                print(f"  Circularity: {analysis.get('circularity', 0):.3f}")
                print(f"  Mean intensity: {analysis.get('mean_intensity', 0):.1f}")
        else:
            print(f"✗ Bright core detection failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
