#!/usr/bin/env python3
"""
Hough Circle Segmentation Module
Advanced circle detection for fiber optic core/cladding segmentation.
Extracted and optimized from separation.py BasicFallbackSegmenter.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class HoughCircleSegmenter:
    """
    Advanced circle detection using Hough transforms for fiber optic segmentation.
    Detects core and cladding circles with multiple fallback strategies.
    """
    
    def __init__(self, 
                 min_circle_ratio: float = 0.1,
                 max_circle_ratio: float = 0.8,
                 core_to_cladding_ratio: float = 0.2,
                 gaussian_blur_kernel: int = 5):
        """
        Initialize the segmenter with configurable parameters.
        
        Args:
            min_circle_ratio: Minimum circle radius as fraction of image size
            max_circle_ratio: Maximum circle radius as fraction of image size
            core_to_cladding_ratio: Expected ratio of core to cladding radius
            gaussian_blur_kernel: Kernel size for Gaussian blur preprocessing
        """
        self.min_circle_ratio = min_circle_ratio
        self.max_circle_ratio = max_circle_ratio
        self.core_to_cladding_ratio = core_to_cladding_ratio
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal circle detection.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 0)
        
        return blurred
    
    def detect_circles_adaptive(self, gray: np.ndarray) -> Optional[List[Tuple[int, int, int]]]:
        """
        Detect circles using adaptive Hough parameters.
        
        Args:
            gray: Preprocessed grayscale image
            
        Returns:
            List of circles as (x, y, radius) tuples, or None if no circles found
        """
        h, w = gray.shape
        min_radius = int(min(h, w) * self.min_circle_ratio)
        max_radius = int(min(h, w) * self.max_circle_ratio)
        min_dist = max(50, min(h, w) // 4)
        
        # Try multiple parameter combinations
        param_sets = [
            {'dp': 1, 'param1': 50, 'param2': 30},
            {'dp': 1, 'param1': 100, 'param2': 50},
            {'dp': 2, 'param1': 50, 'param2': 30},
            {'dp': 1, 'param1': 50, 'param2': 20},
            {'dp': 1.5, 'param1': 75, 'param2': 40}
        ]
        
        for params in param_sets:
            try:
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=params['dp'],
                    minDist=min_dist,
                    param1=params['param1'],
                    param2=params['param2'],
                    minRadius=min_radius,
                    maxRadius=max_radius
                )
                
                if circles is not None:
                    circles_array = np.round(circles[0]).astype(int)
                    circles_list = []
                    for i in range(circles_array.shape[0]):
                        x, y, r = circles_array[i]
                        circles_list.append((int(x), int(y), int(r)))
                    self.logger.info(f"Found {len(circles_list)} circles with params: {params}")
                    return circles_list
                    
            except Exception as e:
                self.logger.warning(f"HoughCircles failed with params {params}: {e}")
                continue
        
        return None
    
    def filter_and_sort_circles(self, circles: List[Tuple[int, int, int]], 
                               image_shape: Tuple[int, int]) -> List[Tuple[int, int, int]]:
        """
        Filter invalid circles and sort by size.
        
        Args:
            circles: List of detected circles
            image_shape: (height, width) of the image
            
        Returns:
            Filtered and sorted circles (largest first)
        """
        h, w = image_shape
        valid_circles = []
        
        for x, y, r in circles:
            # Check if circle is within image bounds
            if (r < x < w - r) and (r < y < h - r):
                # Check if radius is reasonable
                if self.min_circle_ratio * min(h, w) <= r <= self.max_circle_ratio * min(h, w):
                    valid_circles.append((x, y, r))
        
        # Sort by radius (largest first)
        valid_circles.sort(key=lambda circle: circle[2], reverse=True)
        return valid_circles
    
    def assign_core_cladding(self, circles: List[Tuple[int, int, int]]) -> Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
        """
        Assign detected circles to core and cladding based on size and position.
        
        Args:
            circles: List of valid circles sorted by size
            
        Returns:
            Tuple of (cladding_circle, core_circle) or (None, None) if assignment fails
        """
        if not circles:
            return None, None
        
        if len(circles) == 1:
            # Only one circle found - assume it's cladding, estimate core
            cladding = circles[0]
            core_radius = int(cladding[2] * self.core_to_cladding_ratio)
            core = (cladding[0], cladding[1], core_radius)
            return cladding, core
        
        # Multiple circles - find best core/cladding pair
        cladding = circles[0]  # Largest circle is likely cladding
        
        # Look for core circle
        best_core = None
        best_score = 0
        
        for candidate in circles[1:]:
            cx, cy, cr = candidate
            clad_x, clad_y, clad_r = cladding
            
            # Calculate distance between centers
            center_dist = np.sqrt((cx - clad_x)**2 + (cy - clad_y)**2)
            
            # Calculate radius ratio
            radius_ratio = cr / clad_r
            
            # Score based on concentricity and size ratio
            concentricity_score = max(0, 1 - center_dist / (clad_r * 0.3))
            size_score = max(0, 1 - abs(radius_ratio - self.core_to_cladding_ratio) / self.core_to_cladding_ratio)
            
            total_score = concentricity_score * size_score
            
            if total_score > best_score:
                best_score = total_score
                best_core = candidate
        
        # If no good core found, estimate one
        if best_core is None or best_score < 0.3:
            core_radius = int(cladding[2] * self.core_to_cladding_ratio)
            best_core = (cladding[0], cladding[1], core_radius)
        
        return cladding, best_core
    
    def create_geometric_fallback(self, image_shape: Tuple[int, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Create fallback circles based on image geometry when detection fails.
        
        Args:
            image_shape: (height, width) of the image
            
        Returns:
            Tuple of (cladding_circle, core_circle)
        """
        h, w = image_shape
        center_x, center_y = w // 2, h // 2
        
        # Estimate cladding as fraction of image size
        cladding_radius = min(h, w) // 3
        core_radius = int(cladding_radius * self.core_to_cladding_ratio)
        
        cladding = (center_x, center_y, cladding_radius)
        core = (center_x, center_y, core_radius)
        
        return cladding, core
    
    def create_masks(self, core_circle: Tuple[int, int, int], 
                    cladding_circle: Tuple[int, int, int], 
                    image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Create binary masks for core, cladding, and ferrule regions.
        
        Args:
            core_circle: (x, y, radius) of core circle
            cladding_circle: (x, y, radius) of cladding circle
            image_shape: (height, width) of the image
            
        Returns:
            Dictionary containing 'core', 'cladding', and 'ferrule' masks
        """
        h, w = image_shape
        
        # Create coordinate grids
        y_grid, x_grid = np.ogrid[:h, :w]
        
        # Core mask
        core_x, core_y, core_r = core_circle
        core_dist = np.sqrt((x_grid - core_x)**2 + (y_grid - core_y)**2)
        core_mask = (core_dist <= core_r).astype(np.uint8)
        
        # Cladding mask
        clad_x, clad_y, clad_r = cladding_circle
        clad_dist = np.sqrt((x_grid - clad_x)**2 + (y_grid - clad_y)**2)
        cladding_mask = ((core_dist > core_r) & (clad_dist <= clad_r)).astype(np.uint8)
        
        # Ferrule mask (everything outside cladding)
        ferrule_mask = (clad_dist > clad_r).astype(np.uint8)
        
        return {
            'core': core_mask,
            'cladding': cladding_mask,
            'ferrule': ferrule_mask
        }
    
    def segment_image(self, image: np.ndarray) -> Dict:
        """
        Perform complete circle-based segmentation of an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Preprocess image
            gray = self.preprocess_image(image)
            
            # Detect circles
            circles = self.detect_circles_adaptive(gray)
            
            if circles:
                # Filter and sort circles
                valid_circles = self.filter_and_sort_circles(circles, gray.shape)
                
                if valid_circles:
                    # Assign core and cladding
                    cladding, core = self.assign_core_cladding(valid_circles)
                    confidence = 0.7  # Good detection
                    method = 'hough_circles_detected'
                else:
                    # Use geometric fallback
                    cladding, core = self.create_geometric_fallback(gray.shape)
                    confidence = 0.3
                    method = 'hough_circles_geometric_fallback'
            else:
                # No circles detected - use geometric fallback
                cladding, core = self.create_geometric_fallback(gray.shape)
                confidence = 0.1
                method = 'hough_circles_no_detection_fallback'
            
            # Create masks (ensure we have valid circles)
            if core is None or cladding is None:
                # Use geometric fallback
                cladding, core = self.create_geometric_fallback(gray.shape)
                confidence = 0.1
                method = 'hough_circles_fallback_none'
            
            masks = self.create_masks(core, cladding, gray.shape)
            
            # Prepare result
            result = {
                'success': True,
                'center': (float(cladding[0]), float(cladding[1])),
                'core_radius': float(core[2]),
                'cladding_radius': float(cladding[2]),
                'confidence': confidence,
                'method': method,
                'masks': masks,
                'circles_detected': len(circles) if circles else 0,
                'core_circle': core,
                'cladding_circle': cladding
            }
            
            self.logger.info(f"Segmentation successful: {method}, confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Segmentation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'hough_circles_error'
            }
    
    def segment_from_file(self, image_path: str) -> Dict:
        """
        Segment an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': f'Could not load image: {image_path}',
                    'method': 'hough_circles_load_error'
                }
            
            return self.segment_image(image)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'hough_circles_file_error'
            }


def visualize_segmentation(image: np.ndarray, result: Dict, save_path: Optional[str] = None):
    """
    Visualize segmentation results.
    
    Args:
        image: Original image
        result: Segmentation result dictionary
        save_path: Optional path to save visualization
    """
    if not result.get('success'):
        print(f"Cannot visualize failed segmentation: {result.get('error', 'Unknown error')}")
        return
    
    # Create visualization
    vis_img = image.copy()
    
    # Draw circles
    core_circle = result.get('core_circle')
    cladding_circle = result.get('cladding_circle')
    
    if cladding_circle:
        cv2.circle(vis_img, (cladding_circle[0], cladding_circle[1]), cladding_circle[2], (0, 255, 255), 2)
        cv2.putText(vis_img, 'Cladding', (cladding_circle[0] - 30, cladding_circle[1] + cladding_circle[2] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if core_circle:
        cv2.circle(vis_img, (core_circle[0], core_circle[1]), core_circle[2], (0, 255, 0), 2)
        cv2.putText(vis_img, 'Core', (core_circle[0] - 15, core_circle[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add method and confidence info
    method = result.get('method', 'unknown')
    confidence = result.get('confidence', 0)
    cv2.putText(vis_img, f'Method: {method}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_img, f'Confidence: {confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"Visualization saved to: {save_path}")
    else:
        cv2.imshow('Hough Circle Segmentation', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Test the HoughCircleSegmenter functionality."""
    print("Testing HoughCircleSegmenter...")
    
    # Create test segmenter
    segmenter = HoughCircleSegmenter()
    
    # Ask for image path
    image_path = input("Enter path to test image (or press Enter to create synthetic): ").strip()
    
    if not image_path:
        print("Creating synthetic fiber image...")
        # Create synthetic fiber image
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Cladding (outer circle)
        cv2.circle(img, (200, 200), 150, (100, 100, 100), -1)
        # Core (inner circle)
        cv2.circle(img, (200, 200), 30, (200, 200, 200), -1)
        # Add some noise
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        test_path = "synthetic_fiber.png"
        cv2.imwrite(test_path, img)
        image_path = test_path
        print(f"Created synthetic fiber image: {test_path}")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Test segmentation
    result = segmenter.segment_from_file(image_path)
    
    print("\nSegmentation Results:")
    print(f"Success: {result.get('success')}")
    print(f"Method: {result.get('method')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    
    if result.get('success'):
        print(f"Center: {result.get('center')}")
        print(f"Core radius: {result.get('core_radius'):.1f}")
        print(f"Cladding radius: {result.get('cladding_radius'):.1f}")
        print(f"Circles detected: {result.get('circles_detected', 0)}")
        
        # Load image for visualization
        img = cv2.imread(image_path)
        visualize_segmentation(img, result, "segmentation_result.png")
    else:
        print(f"Error: {result.get('error')}")


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    main()
