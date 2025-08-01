#!/usr/bin/env python3
"""
Robust Mask Generation Module
============================

This module provides comprehensive fiber optic region detection using multiple
fallback methods for robust fiber core, cladding, and ferrule segmentation.

Author: Modular Analysis Team
Version: 1.0
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_closing, binary_opening
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
from scipy.ndimage import distance_transform_edt


class RobustMaskGenerator:
    """
    Comprehensive fiber optic mask generation using multiple fallback methods.
    """
    
    def __init__(self, cladding_core_ratio: float = 125.0/9.0, 
                 ferrule_buffer_ratio: float = 1.2):
        """
        Initialize the mask generator.
        
        Args:
            cladding_core_ratio: Ratio of cladding to core diameter (typically 125/9 for SMF)
            ferrule_buffer_ratio: Buffer ratio for ferrule detection
        """
        self.cladding_core_ratio = cladding_core_ratio
        self.ferrule_buffer_ratio = ferrule_buffer_ratio
        self.logger = logging.getLogger(__name__)
    
    def generate_masks(self, image: np.ndarray) -> Tuple[Optional[Dict[str, np.ndarray]], 
                                                         Optional[Dict[str, any]]]:
        """
        Generate fiber region masks using multiple fallback methods.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (masks_dict, localization_info) or (None, None) if failed
        """
        if not self._validate_input(image):
            return None, None
        
        # Try each method in order of reliability
        methods = [
            ("adaptive_threshold", self._adaptive_threshold_method),
            ("hough_circles", self._hough_circles_method),
            ("watershed", self._watershed_method),
            ("template_matching", self._template_matching_method),
            ("contour_analysis", self._contour_analysis_method),
            ("gradient_based", self._gradient_based_method),
            ("distance_transform", self._distance_transform_method),
            ("estimated", self._estimated_method)
        ]
        
        for method_name, method_func in methods:
            try:
                masks, localization = method_func(image)
                if masks is not None and localization is not None:
                    if self._validate_masks(masks, image.shape):
                        localization['method'] = method_name
                        self.logger.info(f"Successfully used {method_name} for mask generation")
                        return masks, localization
            except Exception as e:
                self.logger.warning(f"{method_name} failed: {str(e)}")
                continue
        
        self.logger.error("All mask generation methods failed")
        return None, None
    
    def _validate_input(self, image: np.ndarray) -> bool:
        """Validate input image."""
        if image is None or image.size == 0:
            self.logger.error("Invalid input image")
            return False
        if len(image.shape) != 2:
            self.logger.error("Input must be grayscale image")
            return False
        return True
    
    def _adaptive_threshold_method(self, image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate masks using adaptive thresholding."""
        # Multiple adaptive threshold parameters
        params_list = [
            (51, 10), (71, 15), (91, 20), (31, 5)
        ]
        
        best_result = None
        best_circularity = 0
        
        for block_size, c_value in params_list:
            try:
                # Ensure odd block size
                if block_size % 2 == 0:
                    block_size += 1
                
                adaptive_thresh = cv2.adaptiveThreshold(
                    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, block_size, c_value
                )
                
                # Clean up with morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
                
                # Find contours
                contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                # Find the most circular contour
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < image.shape[0] * image.shape[1] * 0.01:
                        continue
                    
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > best_circularity and circularity > 0.3:
                            (cx, cy), cr = cv2.minEnclosingCircle(contour)
                            cx, cy, cr = int(cx), int(cy), int(cr)
                            
                            if self._validate_circle(cx, cy, cr, image.shape):
                                best_circularity = circularity
                                best_result = (cx, cy, cr, circularity)
            
            except Exception as e:
                self.logger.debug(f"Adaptive threshold params {block_size}, {c_value} failed: {e}")
                continue
        
        if best_result is not None:
            cx, cy, cr, circularity = best_result
            core_r = max(1, int(cr / self.cladding_core_ratio))
            masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
            localization = {
                "center": (cx, cy),
                "cladding_radius_px": cr,
                "core_radius_px": core_r,
                "circularity": circularity
            }
            return masks, localization
        
        return None, None
    
    def _hough_circles_method(self, image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate masks using Hough circle detection."""
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        # Multiple parameter sets for robustness
        param_sets = [
            (50, 30, 1.0), (100, 50, 1.0), (30, 20, 1.2), 
            (80, 40, 1.5), (60, 35, 1.0), (70, 45, 1.0)
        ]
        
        for param1, param2, dp in param_sets:
            try:
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp=dp,
                    minDist=image.shape[0]//2,
                    param1=param1, param2=param2,
                    minRadius=int(image.shape[0] * 0.15),
                    maxRadius=int(image.shape[0] * 0.45)
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        cx, cy, cr = circle
                        if self._validate_circle(cx, cy, cr, image.shape):
                            core_r = max(1, int(cr / self.cladding_core_ratio))
                            masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
                            localization = {
                                "center": (int(cx), int(cy)),
                                "cladding_radius_px": int(cr),
                                "core_radius_px": core_r
                            }
                            return masks, localization
            except Exception as e:
                self.logger.debug(f"Hough params {param1}, {param2}, {dp} failed: {e}")
                continue
        
        return None, None
    
    def _watershed_method(self, image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate masks using watershed segmentation."""
        try:
            # Create markers using distance transform
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Clean binary image
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Distance transform
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Find local maxima as markers
            from skimage.feature import peak_local_maxima
            local_maxima = peak_local_maxima(dist_transform, min_distance=20, threshold_abs=0.3*dist_transform.max())
            
            if len(local_maxima) > 0:
                markers = np.zeros_like(dist_transform, dtype=np.int32)
                for i, (y, x) in enumerate(local_maxima):
                    markers[y, x] = i + 1
                
                # Apply watershed
                labels = watershed(-dist_transform, markers, mask=binary)
                
                # Find the largest component (likely the fiber)
                unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
                if len(counts) > 0:
                    largest_label = unique_labels[np.argmax(counts)]
                    fiber_mask = (labels == largest_label).astype(np.uint8) * 255
                    
                    # Find contour and enclosing circle
                    contours, _ = cv2.findContours(fiber_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        (cx, cy), cr = cv2.minEnclosingCircle(largest_contour)
                        cx, cy, cr = int(cx), int(cy), int(cr)
                        
                        if self._validate_circle(cx, cy, cr, image.shape):
                            core_r = max(1, int(cr / self.cladding_core_ratio))
                            masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
                            localization = {
                                "center": (cx, cy),
                                "cladding_radius_px": cr,
                                "core_radius_px": core_r
                            }
                            return masks, localization
        
        except Exception as e:
            self.logger.debug(f"Watershed method failed: {e}")
        
        return None, None
    
    def _template_matching_method(self, image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate masks using template matching."""
        try:
            # Create circular templates of different sizes
            h, w = image.shape
            template_sizes = [w//6, w//5, w//4, w//3]
            
            best_match = None
            best_score = 0
            
            for size in template_sizes:
                if size < 20:  # Too small
                    continue
                
                # Create circular template
                template = np.zeros((size*2, size*2), dtype=np.uint8)
                cv2.circle(template, (size, size), size-2, 255, -1)
                
                # Template matching
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    cx = max_loc[0] + size
                    cy = max_loc[1] + size
                    cr = size
                    best_match = (cx, cy, cr)
            
            if best_match is not None and best_score > 0.3:
                cx, cy, cr = best_match
                if self._validate_circle(cx, cy, cr, image.shape):
                    core_r = max(1, int(cr / self.cladding_core_ratio))
                    masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
                    localization = {
                        "center": (cx, cy),
                        "cladding_radius_px": cr,
                        "core_radius_px": core_r
                    }
                    return masks, localization
        
        except Exception as e:
            self.logger.debug(f"Template matching failed: {e}")
        
        return None, None
    
    def _contour_analysis_method(self, image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate masks using contour analysis."""
        try:
            # Multiple threshold methods
            threshold_methods = [
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                cv2.THRESH_TRIANGLE,
                cv2.THRESH_TOZERO + cv2.THRESH_OTSU
            ]
            
            best_circle = None
            best_score = 0
            
            for thresh_method in threshold_methods:
                try:
                    _, binary = cv2.threshold(image, 0, 255, thresh_method)
                    
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < 100:  # Too small
                            continue
                        
                        # Check circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            
                            if circularity > 0.4:  # Reasonably circular
                                (cx, cy), cr = cv2.minEnclosingCircle(contour)
                                cx, cy, cr = int(cx), int(cy), int(cr)
                                
                                if self._validate_circle(cx, cy, cr, image.shape):
                                    score = circularity * area  # Combined score
                                    if score > best_score:
                                        best_score = score
                                        best_circle = (cx, cy, cr, circularity)
                
                except Exception as e:
                    continue
            
            if best_circle is not None:
                cx, cy, cr, circularity = best_circle
                core_r = max(1, int(cr / self.cladding_core_ratio))
                masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
                localization = {
                    "center": (cx, cy),
                    "cladding_radius_px": cr,
                    "core_radius_px": core_r,
                    "circularity": circularity
                }
                return masks, localization
        
        except Exception as e:
            self.logger.debug(f"Contour analysis failed: {e}")
        
        return None, None
    
    def _gradient_based_method(self, image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate masks using gradient-based edge detection."""
        try:
            # Compute gradients
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize and threshold
            gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, binary = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the most circular contour
                best_contour = None
                best_circularity = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 100:
                        continue
                    
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > best_circularity:
                            best_circularity = circularity
                            best_contour = contour
                
                if best_contour is not None and best_circularity > 0.3:
                    (cx, cy), cr = cv2.minEnclosingCircle(best_contour)
                    cx, cy, cr = int(cx), int(cy), int(cr)
                    
                    if self._validate_circle(cx, cy, cr, image.shape):
                        core_r = max(1, int(cr / self.cladding_core_ratio))
                        masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
                        localization = {
                            "center": (cx, cy),
                            "cladding_radius_px": cr,
                            "core_radius_px": core_r,
                            "circularity": best_circularity
                        }
                        return masks, localization
        
        except Exception as e:
            self.logger.debug(f"Gradient-based method failed: {e}")
        
        return None, None
    
    def _distance_transform_method(self, image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate masks using distance transform."""
        try:
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Distance transform
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Find the point with maximum distance (center of largest circle)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
            
            if max_val > 10:  # Reasonable radius
                cx, cy = max_loc
                cr = int(max_val * 0.9)  # Slightly smaller than max distance
                
                if self._validate_circle(cx, cy, cr, image.shape):
                    core_r = max(1, int(cr / self.cladding_core_ratio))
                    masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
                    localization = {
                        "center": (cx, cy),
                        "cladding_radius_px": cr,
                        "core_radius_px": core_r
                    }
                    return masks, localization
        
        except Exception as e:
            self.logger.debug(f"Distance transform method failed: {e}")
        
        return None, None
    
    def _estimated_method(self, image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Last resort: use estimated center and radius."""
        self.logger.warning("Using estimated fiber location - results may be inaccurate")
        
        h, w = image.shape
        cx, cy = w // 2, h // 2
        cr = min(h, w) // 3
        core_r = max(1, int(cr / self.cladding_core_ratio))
        
        masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
        localization = {
            "center": (cx, cy),
            "cladding_radius_px": cr,
            "core_radius_px": core_r
        }
        return masks, localization
    
    def _validate_circle(self, cx: int, cy: int, cr: int, image_shape: Tuple[int, int]) -> bool:
        """Validate that the detected circle is reasonable."""
        h, w = image_shape
        
        # Check if center is within image
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return False
        
        # Check if radius is reasonable
        if cr < min(h, w) * 0.1 or cr > min(h, w) * 0.45:
            return False
        
        # Check if circle fits within image
        if (cx - cr < 0 or cx + cr >= w or 
            cy - cr < 0 or cy + cr >= h):
            return False
        
        return True
    
    def _validate_masks(self, masks: Dict[str, np.ndarray], image_shape: Tuple[int, int]) -> bool:
        """Validate generated masks."""
        if not masks:
            return False
        
        required_masks = ['Core', 'Cladding', 'Ferrule', 'Fiber']
        for mask_name in required_masks:
            if mask_name not in masks:
                return False
            
            mask = masks[mask_name]
            if mask.shape != image_shape:
                return False
            
            if mask_name == 'Core' and np.sum(mask) < 10:  # Core too small
                return False
        
        return True
    
    def _create_fiber_masks(self, shape: Tuple[int, int], cx: int, cy: int, 
                           core_r: int, cladding_r: int) -> Dict[str, np.ndarray]:
        """Create fiber region masks."""
        h, w = shape[:2]
        
        # Core mask
        core_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core_mask, (cx, cy), core_r, 255, -1)
        
        # Full cladding mask (including core)
        cladding_mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(cladding_mask_full, (cx, cy), cladding_r, 255, -1)
        
        # Cladding only (excluding core)
        cladding_mask = cv2.subtract(cladding_mask_full, core_mask)
        
        # Ferrule mask (everything outside cladding)
        ferrule_mask = np.ones((h, w), dtype=np.uint8) * 255
        cv2.circle(ferrule_mask, (cx, cy), 
                  int(cladding_r * self.ferrule_buffer_ratio), 0, -1)
        
        # Fiber mask (core + cladding)
        fiber_mask = cv2.add(core_mask, cladding_mask)
        
        return {
            "Core": core_mask,
            "Cladding": cladding_mask,
            "Ferrule": ferrule_mask,
            "Fiber": fiber_mask
        }


def test_robust_mask_generation():
    """Test the robust mask generation with synthetic data."""
    print("Testing Robust Mask Generation Module")
    print("=" * 50)
    
    # Create a synthetic fiber image
    def create_test_fiber_image(size=400):
        image = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        
        # Add background
        image.fill(200)
        
        # Add ferrule (darker)
        cv2.circle(image, center, size//3, 160, -1)
        
        # Add cladding
        cv2.circle(image, center, size//4, 120, -1)
        
        # Add core
        cv2.circle(image, center, size//20, 80, -1)
        
        # Add some noise
        noise = np.random.normal(0, 5, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    # Test with different image conditions
    test_cases = [
        ("normal", create_test_fiber_image(400)),
        ("small", create_test_fiber_image(200)),
        ("large", create_test_fiber_image(600)),
    ]
    
    generator = RobustMaskGenerator()
    
    for case_name, test_image in test_cases:
        print(f"\nTesting {case_name} image...")
        
        masks, localization = generator.generate_masks(test_image)
        
        if masks is not None and localization is not None:
            print(f"✓ Success with method: {localization['method']}")
            print(f"  Center: {localization['center']}")
            print(f"  Cladding radius: {localization['cladding_radius_px']}px")
            print(f"  Core radius: {localization['core_radius_px']}px")
            print(f"  Generated masks: {list(masks.keys())}")
            
            # Visualize results
            visualize_masks(test_image, masks, localization, f"mask_test_{case_name}.png")
        else:
            print(f"✗ Failed to generate masks for {case_name} image")


def visualize_masks(image: np.ndarray, masks: Dict[str, np.ndarray], 
                   localization: Dict, save_path: str = None):
    """Visualize the generated masks."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image with circle overlay
    axes[0, 0].imshow(image, cmap='gray')
    cx, cy = localization['center']
    cladding_r = localization['cladding_radius_px']
    core_r = localization['core_radius_px']
    
    circle1 = plt.Circle((cx, cy), cladding_r, fill=False, color='red', linewidth=2)
    circle2 = plt.Circle((cx, cy), core_r, fill=False, color='blue', linewidth=2)
    axes[0, 0].add_patch(circle1)
    axes[0, 0].add_patch(circle2)
    axes[0, 0].set_title(f"Original + Detection\\nMethod: {localization.get('method', 'unknown')}")
    axes[0, 0].axis('off')
    
    # Individual masks
    mask_names = ['Core', 'Cladding', 'Ferrule', 'Fiber']
    colors = ['Blues', 'Greens', 'Reds', 'Purples']
    
    for i, (mask_name, cmap) in enumerate(zip(mask_names, colors)):
        if mask_name in masks:
            row, col = divmod(i + 1, 3)
            if i == 3:  # Fiber mask
                row, col = 1, 2
            axes[row, col].imshow(masks[mask_name], cmap=cmap)
            axes[row, col].set_title(f"{mask_name} Mask")
            axes[row, col].axis('off')
    
    # Combined visualization
    axes[1, 1].imshow(image, cmap='gray', alpha=0.7)
    for mask_name, color in zip(['Core', 'Cladding'], ['red', 'green']):
        if mask_name in masks:
            mask_overlay = np.ma.masked_where(masks[mask_name] == 0, masks[mask_name])
            axes[1, 1].imshow(mask_overlay, alpha=0.5, cmap=color+'s')
    axes[1, 1].set_title("Combined Overlay")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def demo_robust_mask_generation():
    """Demonstrate the robust mask generation with various scenarios."""
    print("Robust Mask Generation Demo")
    print("=" * 40)
    
    # Test the module
    test_robust_mask_generation()
    
    print("\\nDemo completed!")
    print("Generated files: mask_test_*.png")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    demo_robust_mask_generation()
