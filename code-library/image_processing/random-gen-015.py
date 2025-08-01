#!/usr/bin/env python3
"""
Hough Transform Fiber Separation Module
Uses Hough circle detection to identify fiber core and cladding regions.
Includes adaptive parameter tuning for low-contrast images.
"""

import cv2
import numpy as np
import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


class HoughFiberSeparator:
    """
    Separates fiber optic images into core and cladding using Hough circle detection
    """
    
    def __init__(self,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8),
                 canny_thresh1: int = 50,
                 canny_thresh2: int = 150,
                 gaussian_kernel: Tuple[int, int] = (3, 3),
                 hough_dp: int = 1,
                 hough_min_dist: int = 50,
                 hough_param1: int = 200,
                 hough_param2: int = 20,
                 cladding_min_radius: int = 100,
                 cladding_max_radius: int = 500,
                 core_min_radius: int = 20,
                 core_max_radius: int = 80):
        """
        Initialize the Hough fiber separator
        
        Args:
            clahe_clip_limit: CLAHE contrast limiting threshold
            clahe_tile_size: CLAHE tile grid size
            canny_thresh1: Lower threshold for Canny edge detection
            canny_thresh2: Upper threshold for Canny edge detection
            gaussian_kernel: Gaussian blur kernel size
            hough_dp: Inverse ratio of accumulator resolution to image resolution
            hough_min_dist: Minimum distance between detected circle centers
            hough_param1: Upper threshold for edge detection in Hough transform
            hough_param2: Accumulator threshold for center detection
            cladding_min_radius: Minimum radius for cladding detection
            cladding_max_radius: Maximum radius for cladding detection
            core_min_radius: Minimum radius for core detection
            core_max_radius: Maximum radius for core detection
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.gaussian_kernel = gaussian_kernel
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.cladding_min_radius = cladding_min_radius
        self.cladding_max_radius = cladding_max_radius
        self.core_min_radius = core_min_radius
        self.core_max_radius = core_max_radius
    
    def preprocess_image(self, image: np.ndarray, apply_blur_after_canny: bool = False) -> Dict[str, np.ndarray]:
        """
        Preprocess image with CLAHE enhancement and Canny edge detection
        
        Args:
            image: Input grayscale image
            apply_blur_after_canny: Whether to apply blur after Canny
            
        Returns:
            Dictionary containing processed images
        """
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_size)
        clahe_image = clahe.apply(image)
        
        # Apply Gaussian blur before Canny to reduce noise
        img_blurred = cv2.GaussianBlur(clahe_image, self.gaussian_kernel, 0)
        
        # Apply Canny edge detection
        canny_image = cv2.Canny(img_blurred, self.canny_thresh1, self.canny_thresh2, apertureSize=3)
        
        # Optional blur after Canny
        if apply_blur_after_canny:
            canny_image = cv2.GaussianBlur(canny_image, (5, 5), 0)
        
        return {
            'original': image,
            'clahe': clahe_image,
            'blurred': img_blurred,
            'canny': canny_image
        }
    
    def extract_circular_region(self, image: np.ndarray, center_x: int, center_y: int, 
                               radius: int, invert: bool = False) -> np.ndarray:
        """
        Extract or mask circular region from image
        
        Args:
            image: Input grayscale image
            center_x: Circle center x coordinate
            center_y: Circle center y coordinate
            radius: Circle radius
            invert: If True, mask inside circle; if False, mask outside circle
            
        Returns:
            Image with circular extraction/masking applied
        """
        result = image.copy()
        rows, cols = result.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:rows, :cols]
        
        # Calculate distance from center
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        if invert:
            # Mask inside circle (for creating cladding mask)
            mask = distances <= radius
        else:
            # Mask outside circle (for extracting circular region)
            mask = distances > radius
        
        result[mask] = 0
        
        return result
    
    def detect_cladding_circle(self, processed_images: Dict[str, np.ndarray]) -> Optional[Tuple[int, int, int]]:
        """
        Detect cladding circle using Hough transform
        
        Args:
            processed_images: Dictionary of preprocessed images
            
        Returns:
            Tuple of (center_x, center_y, radius) or None if not found
        """
        canny_image = processed_images['canny']
        
        # Primary detection attempt
        circles = cv2.HoughCircles(
            canny_image,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.cladding_min_radius,
            maxRadius=self.cladding_max_radius
        )
        
        # Fallback with relaxed parameters
        if circles is None:
            circles = cv2.HoughCircles(
                canny_image,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_dp,
                minDist=self.hough_min_dist,
                param1=self.hough_param1,
                param2=15,  # Lower threshold for faint circles
                minRadius=self.cladding_min_radius,
                maxRadius=self.cladding_max_radius
            )
        
        if circles is not None:
            # Return the first (most confident) circle
            circle = circles[0][0]
            return int(circle[0]), int(circle[1]), int(circle[2])
        
        return None
    
    def detect_core_circle(self, cladding_region: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect core circle within cladding region
        
        Args:
            cladding_region: Extracted cladding region image
            
        Returns:
            Tuple of (center_x, center_y, radius) or None if not found
        """
        # Enhance cladding region for core detection
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_size)
        enhanced_cladding = clahe.apply(cladding_region)
        
        # Apply Canny edge detection
        canny_cladding = cv2.Canny(enhanced_cladding, self.canny_thresh1, self.canny_thresh2)
        canny_blurred = cv2.GaussianBlur(canny_cladding, (7, 7), 0)
        
        # Primary core detection
        circles = cv2.HoughCircles(
            canny_blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=10,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.core_min_radius,
            maxRadius=self.core_max_radius
        )
        
        # Fallback with adjusted parameters for low-contrast cores
        if circles is None:
            # Lower Canny thresholds and smaller blur
            canny_adjusted = cv2.Canny(enhanced_cladding, 30, 100)
            canny_adjusted_blurred = cv2.GaussianBlur(canny_adjusted, (5, 5), 0)
            
            circles = cv2.HoughCircles(
                canny_adjusted_blurred,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_dp,
                minDist=10,
                param1=self.hough_param1,
                param2=15,  # Lower threshold
                minRadius=self.core_min_radius,
                maxRadius=min(60, self.core_max_radius)  # Smaller max radius
            )
        
        if circles is not None:
            # Return the first (most confident) circle
            circle = circles[0][0]
            return int(circle[0]), int(circle[1]), int(circle[2])
        
        return None
    
    def create_masks(self, image_shape: Tuple[int, int], 
                    cladding_center: Tuple[int, int], cladding_radius: int,
                    core_center: Optional[Tuple[int, int]] = None, 
                    core_radius: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Create binary masks for different fiber regions
        
        Args:
            image_shape: Shape of the original image (height, width)
            cladding_center: Center of cladding circle
            cladding_radius: Radius of cladding circle
            core_center: Center of core circle (optional)
            core_radius: Radius of core circle (optional)
            
        Returns:
            Dictionary of binary masks
        """
        height, width = image_shape
        masks = {}
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Cladding mask (inside cladding circle)
        clad_x, clad_y = cladding_center
        cladding_distances = np.sqrt((x_coords - clad_x)**2 + (y_coords - clad_y)**2)
        masks['cladding'] = (cladding_distances <= cladding_radius).astype(np.uint8) * 255
        
        # Core mask (if core detected)
        if core_center and core_radius:
            core_x, core_y = core_center
            core_distances = np.sqrt((x_coords - core_x)**2 + (y_coords - core_y)**2)
            masks['core'] = (core_distances <= core_radius).astype(np.uint8) * 255
        else:
            masks['core'] = np.zeros((height, width), dtype=np.uint8)
        
        # Ferrule mask (outside cladding)
        masks['ferrule'] = ((cladding_distances > cladding_radius).astype(np.uint8) * 255)
        
        return masks
    
    def estimate_core_from_cladding(self, cladding_center: Tuple[int, int], 
                                   cladding_radius: int) -> Tuple[Tuple[int, int], int]:
        """
        Estimate core position and size based on cladding
        
        Args:
            cladding_center: Center of detected cladding
            cladding_radius: Radius of detected cladding
            
        Returns:
            Tuple of (estimated_core_center, estimated_core_radius)
        """
        # Assume core is centered and roughly 1/6 the diameter of cladding
        estimated_core_radius = max(self.core_min_radius, cladding_radius // 6)
        estimated_core_radius = min(estimated_core_radius, self.core_max_radius)
        
        return cladding_center, estimated_core_radius
    
    def separate_fiber(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Main function to separate fiber into core and cladding regions
        
        Args:
            image_path: Path to input fiber image
            output_dir: Optional directory to save intermediate results
            
        Returns:
            Dictionary containing separation results
        """
        result = {
            'method': 'hough_separation',
            'image_path': image_path,
            'success': False,
            'center': None,
            'core_radius': None,
            'cladding_radius': None,
            'confidence': 0.0,
            'core_detected': False,
            'cladding_detected': False
        }
        
        # Validate input
        if not Path(image_path).exists():
            result['error'] = f"File not found: '{image_path}'"
            return result
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            result['error'] = f"Could not read image from '{image_path}'"
            return result
        
        try:
            # Preprocess image
            processed_images = self.preprocess_image(image)
            
            # Detect cladding circle
            cladding_result = self.detect_cladding_circle(processed_images)
            if cladding_result is None:
                result['error'] = "Could not detect cladding circle"
                return result
            
            cladding_x, cladding_y, cladding_radius = cladding_result
            result['cladding_detected'] = True
            result['center'] = (cladding_x, cladding_y)
            result['cladding_radius'] = cladding_radius
            
            # Extract cladding region for core detection
            cladding_region = self.extract_circular_region(
                image, cladding_x, cladding_y, cladding_radius, invert=False
            )
            
            # Detect core circle
            core_result = self.detect_core_circle(cladding_region)
            
            if core_result is not None:
                core_x, core_y, core_radius = core_result
                result['core_detected'] = True
                result['core_radius'] = core_radius
                # Core coordinates are relative to original image
                result['core_center'] = (core_x, core_y)
                result['confidence'] = 0.8  # High confidence when both detected
            else:
                # Estimate core based on cladding
                estimated_center, estimated_radius = self.estimate_core_from_cladding(
                    (cladding_x, cladding_y), cladding_radius
                )
                result['core_center'] = estimated_center
                result['core_radius'] = estimated_radius
                result['core_estimated'] = True
                result['confidence'] = 0.4  # Lower confidence for estimated core
            
            result['success'] = True
            
            # Create masks
            core_center = result.get('core_center', (cladding_x, cladding_y))
            masks = self.create_masks(
                image.shape,
                (cladding_x, cladding_y),
                cladding_radius,
                core_center,
                result['core_radius']
            )
            result['masks'] = masks
            
            # Save intermediate results if output directory specified
            if output_dir:
                self._save_results(image, result, processed_images, output_dir, image_path)
                
        except Exception as e:
            result['error'] = f"Processing error: {str(e)}"
        
        return result
    
    def _save_results(self, original_image: np.ndarray, result: Dict[str, Any],
                     processed_images: Dict[str, np.ndarray], 
                     output_dir: str, image_path: str):
        """Save intermediate processing results"""
        os.makedirs(output_dir, exist_ok=True)
        base_filename = Path(image_path).stem
        
        # Save processed images
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_clahe.png"), 
                   processed_images['clahe'])
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_canny.png"), 
                   processed_images['canny'])
        
        # Save masks if available
        if 'masks' in result:
            for mask_name, mask in result['masks'].items():
                cv2.imwrite(os.path.join(output_dir, f"{base_filename}_mask_{mask_name}.png"), mask)
        
        # Save result visualization
        vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # Draw cladding circle
        if result['cladding_detected']:
            center = result['center']
            radius = result['cladding_radius']
            cv2.circle(vis_image, center, radius, (0, 255, 0), 2)
            cv2.circle(vis_image, center, 3, (0, 255, 0), -1)
        
        # Draw core circle
        if result.get('core_center') and result.get('core_radius'):
            core_center = result['core_center']
            core_radius = result['core_radius']
            color = (255, 0, 0) if result.get('core_detected') else (0, 0, 255)
            cv2.circle(vis_image, core_center, core_radius, color, 2)
            cv2.circle(vis_image, core_center, 2, color, -1)
        
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_result.png"), vis_image)
        
        # Save JSON result
        result_copy = result.copy()
        result_copy.pop('masks', None)  # Remove masks from JSON (too large)
        with open(os.path.join(output_dir, f"{base_filename}_hough_result.json"), 'w') as f:
            json.dump(result_copy, f, indent=4)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Hough Transform Fiber Separation')
    parser.add_argument('image_path', help='Path to input fiber image')
    parser.add_argument('--output-dir', default='output_hough',
                       help='Output directory for results')
    parser.add_argument('--canny-low', type=int, default=50,
                       help='Lower Canny threshold')
    parser.add_argument('--canny-high', type=int, default=150,
                       help='Upper Canny threshold')
    parser.add_argument('--hough-param2', type=int, default=20,
                       help='Hough param2 (accumulator threshold)')
    parser.add_argument('--min-dist', type=int, default=50,
                       help='Minimum distance between circle centers')
    parser.add_argument('--cladding-min-radius', type=int, default=100,
                       help='Minimum cladding radius')
    parser.add_argument('--cladding-max-radius', type=int, default=500,
                       help='Maximum cladding radius')
    
    args = parser.parse_args()
    
    # Create separator
    separator = HoughFiberSeparator(
        canny_thresh1=args.canny_low,
        canny_thresh2=args.canny_high,
        hough_param2=args.hough_param2,
        hough_min_dist=args.min_dist,
        cladding_min_radius=args.cladding_min_radius,
        cladding_max_radius=args.cladding_max_radius
    )
    
    # Process image
    result = separator.separate_fiber(args.image_path, args.output_dir)
    
    # Print results
    print(f"Fiber separation {'successful' if result['success'] else 'failed'}")
    if result['success']:
        print(f"Cladding center: {result['center']}")
        print(f"Cladding radius: {result['cladding_radius']} pixels")
        if result.get('core_center'):
            core_status = "detected" if result.get('core_detected') else "estimated"
            print(f"Core center: {result['core_center']} ({core_status})")
            print(f"Core radius: {result['core_radius']} pixels")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
