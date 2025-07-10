#!/usr/bin/env python3
"""
Bright Core Extractor Module
Detects bright fiber cores using local contrast validation and Hough circle detection.
Optimized for robust fiber optic core detection.
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


class BrightCoreExtractor:
    """
    Detects bright fiber cores using circular Hough transform and local contrast validation
    """
    
    def __init__(self, 
                 gaussian_kernel: Tuple[int, int] = (5, 5),
                 median_kernel: int = 5,
                 hough_dp: int = 1,
                 hough_min_dist: int = 100,
                 hough_param1: int = 150,
                 hough_param2: int = 25,
                 min_radius: int = 15,
                 max_radius: int = 150,
                 contrast_factor: float = 1.15,
                 outer_ring_width: int = 15):
        """
        Initialize the bright core extractor
        
        Args:
            gaussian_kernel: Gaussian blur kernel size
            median_kernel: Median blur kernel size
            hough_dp: Hough transform dp parameter
            hough_min_dist: Minimum distance between circle centers
            hough_param1: First method-specific parameter for Hough
            hough_param2: Second method-specific parameter for Hough
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            contrast_factor: Local contrast validation factor
            outer_ring_width: Width of outer ring for contrast comparison
        """
        self.gaussian_kernel = gaussian_kernel
        self.median_kernel = median_kernel
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.contrast_factor = contrast_factor
        self.outer_ring_width = outer_ring_width
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to enhance circle detection
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image
        """
        # Apply Gaussian blur
        processed = cv2.GaussianBlur(image, self.gaussian_kernel, 0)
        
        # Apply median blur to reduce noise
        processed = cv2.medianBlur(processed, self.median_kernel)
        
        return processed
    
    def find_circles(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect circles using Hough circle transform
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Array of detected circles or None if no circles found
        """
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        if circles is not None:
            return np.uint16(np.around(circles[0, :]))
        return None
    
    def validate_circle_contrast(self, image: np.ndarray, circle: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a circle using local contrast comparison
        
        Args:
            image: Original grayscale image
            circle: Circle parameters [x, y, radius]
            
        Returns:
            Tuple of (is_valid, debug_data)
        """
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        
        # Create inner and outer masks
        inner_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(inner_mask, (x, y), r, 255, -1)
        
        outer_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), r + self.outer_ring_width, 255, -1)
        outer_mask = cv2.subtract(outer_mask, inner_mask)
        
        # Calculate mean intensities
        inner_pixels = image[inner_mask > 0]
        outer_pixels = image[outer_mask > 0]
        
        debug_data = {
            'circle': (x, y, r),
            'inner_pixels_count': len(inner_pixels),
            'outer_pixels_count': len(outer_pixels),
            'inner_mean': 0.0,
            'outer_mean': 0.0,
            'contrast_ratio': 0.0,
            'required_ratio': self.contrast_factor
        }
        
        # Check if we have pixels in both regions
        if len(inner_pixels) == 0 or len(outer_pixels) == 0:
            debug_data['error'] = 'Insufficient pixels in inner or outer region'
            return False, debug_data
        
        inner_mean = np.mean(inner_pixels)
        outer_mean = np.mean(outer_pixels)
        
        debug_data['inner_mean'] = float(inner_mean)
        debug_data['outer_mean'] = float(outer_mean)
        debug_data['contrast_ratio'] = float(inner_mean / outer_mean) if outer_mean > 0 else 0.0
        
        # Check if inner area is sufficiently brighter than outer ring
        is_valid = inner_mean > outer_mean * self.contrast_factor
        
        return is_valid, debug_data
    
    def create_segmentation_mask(self, image: np.ndarray, center: Tuple[int, int], 
                                radius: int) -> np.ndarray:
        """
        Create a precise segmentation mask for the detected core
        
        Args:
            image: Original image
            center: Circle center (x, y)
            radius: Circle radius
            
        Returns:
            Binary segmentation mask
        """
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        return mask
    
    def analyze_core_properties(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Analyze properties of the detected core region
        
        Args:
            image: Original image
            mask: Core segmentation mask
            
        Returns:
            Dictionary of core properties
        """
        core_pixels = image[mask > 0]
        
        properties = {
            'area_pixels': int(np.sum(mask > 0)),
            'mean_intensity': float(np.mean(core_pixels)) if len(core_pixels) > 0 else 0.0,
            'std_intensity': float(np.std(core_pixels)) if len(core_pixels) > 0 else 0.0,
            'min_intensity': int(np.min(core_pixels)) if len(core_pixels) > 0 else 0,
            'max_intensity': int(np.max(core_pixels)) if len(core_pixels) > 0 else 0,
        }
        
        # Calculate additional statistics
        if len(core_pixels) > 0:
            properties['median_intensity'] = float(np.median(core_pixels))
            properties['intensity_range'] = int(properties['max_intensity'] - properties['min_intensity'])
        
        return properties
    
    def extract_bright_core(self, image_path: str, debug_mode: bool = False) -> Dict[str, Any]:
        """
        Main function to extract bright core from fiber optic image
        
        Args:
            image_path: Path to input image
            debug_mode: Whether to generate debug visualizations
            
        Returns:
            Dictionary containing extraction results
        """
        result = {
            'method': 'bright_core_extractor',
            'image_path': image_path,
            'success': False,
            'center': None,
            'core_radius': None,
            'confidence': 0.0,
            'circles_detected': 0,
            'validation_attempts': 0
        }
        
        # Load image
        if not Path(image_path).exists():
            result['error'] = f"Image file not found: {image_path}"
            return result
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            result['error'] = f"Could not read image: {image_path}"
            return result
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Find circles
        circles = self.find_circles(processed_image)
        if circles is None:
            result['error'] = 'No circles detected'
            return result
        
        result['circles_detected'] = len(circles)
        
        # Validate each circle
        best_circle = None
        best_validation_data = None
        best_confidence = 0.0
        
        for i, circle in enumerate(circles):
            result['validation_attempts'] += 1
            
            is_valid, validation_data = self.validate_circle_contrast(image, circle)
            
            if is_valid:
                # Calculate confidence based on contrast ratio
                confidence = min(1.0, validation_data['contrast_ratio'] / self.contrast_factor)
                
                if confidence > best_confidence:
                    best_circle = circle
                    best_validation_data = validation_data
                    best_confidence = confidence
            
            # Save debug image if requested and validation failed
            if debug_mode and not is_valid:
                self._save_debug_image(image, circle, validation_data, image_path, i)
        
        # Set results if we found a valid circle
        if best_circle is not None:
            x, y, r = best_circle
            result['success'] = True
            result['center'] = (int(x), int(y))
            result['core_radius'] = int(r)
            result['confidence'] = float(best_confidence)
            result['validation_data'] = best_validation_data
            
            # Create segmentation mask and analyze properties
            mask = self.create_segmentation_mask(image, (int(x), int(y)), int(r))
            result['core_properties'] = self.analyze_core_properties(image, mask)
            
            # Save successful detection if debug mode
            if debug_mode:
                self._save_success_image(image, best_circle, image_path)
        else:
            result['error'] = f'Found {len(circles)} circles but none passed local contrast validation'
        
        return result
    
    def _save_debug_image(self, image: np.ndarray, circle: np.ndarray, 
                         validation_data: Dict[str, Any], image_path: str, index: int):
        """Save debug visualization for failed validation"""
        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        x, y, r = validation_data['circle']
        
        # Draw inner and outer regions
        inner_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(inner_mask, (x, y), r, 255, -1)
        
        outer_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), r + self.outer_ring_width, 255, -1)
        outer_mask = cv2.subtract(outer_mask, inner_mask)
        
        # Color the regions
        debug_img[inner_mask > 0] = debug_img[inner_mask > 0] * 0.5 + np.array([0, 128, 0], dtype=np.uint8)
        debug_img[outer_mask > 0] = debug_img[outer_mask > 0] * 0.5 + np.array([0, 0, 128], dtype=np.uint8)
        
        # Draw circle
        cv2.circle(debug_img, (x, y), r, (0, 255, 255), 2)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_img, f"Inner: {validation_data['inner_mean']:.2f}", 
                   (10, 30), font, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Outer: {validation_data['outer_mean']:.2f}", 
                   (10, 60), font, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Ratio: {validation_data['contrast_ratio']:.2f}", 
                   (10, 90), font, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Required: {validation_data['required_ratio']:.2f}", 
                   (10, 120), font, 0.6, (0, 255, 255), 2)
        
        # Save debug image
        debug_path = Path(image_path).parent / f"{Path(image_path).stem}_debug_{index}.png"
        cv2.imwrite(str(debug_path), debug_img)
    
    def _save_success_image(self, image: np.ndarray, circle: np.ndarray, image_path: str):
        """Save visualization of successful detection"""
        result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        x, y, r = circle
        cv2.circle(result_img, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.circle(result_img, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Add text
        cv2.putText(result_img, f"Core detected: R={r}px", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save result image
        result_path = Path(image_path).parent / f"{Path(image_path).stem}_core_detected.png"
        cv2.imwrite(str(result_path), result_img)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Bright Core Extractor')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with visualizations')
    parser.add_argument('--min-radius', type=int, default=15,
                       help='Minimum circle radius')
    parser.add_argument('--max-radius', type=int, default=150,
                       help='Maximum circle radius')
    parser.add_argument('--contrast-factor', type=float, default=1.15,
                       help='Local contrast factor for validation')
    parser.add_argument('--hough-param2', type=int, default=25,
                       help='Hough transform param2 (lower = more sensitive)')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = BrightCoreExtractor(
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        contrast_factor=args.contrast_factor,
        hough_param2=args.hough_param2
    )
    
    # Extract bright core
    result = extractor.extract_bright_core(args.image_path, debug_mode=args.debug)
    
    # Print results
    print(f"Core extraction {'successful' if result['success'] else 'failed'}")
    if result['success']:
        print(f"Center: {result['center']}")
        print(f"Radius: {result['core_radius']} pixels")
        print(f"Confidence: {result['confidence']:.3f}")
        if 'core_properties' in result:
            props = result['core_properties']
            print(f"Core area: {props['area_pixels']} pixels")
            print(f"Mean intensity: {props['mean_intensity']:.2f}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print(f"Circles detected: {result['circles_detected']}")
    print(f"Validation attempts: {result['validation_attempts']}")


if __name__ == "__main__":
    main()
