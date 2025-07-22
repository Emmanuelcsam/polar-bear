#!/usr/bin/env python3
"""
Fiber Localization Module
========================
Standalone implementation for locating fiber boundaries (cladding and core)
in fiber optic end-face images using multiple detection methods.

Features:
- HoughCircles detection for cladding boundary
- Template matching for enhanced detection
- Adaptive thresholding with contour fitting
- Circle fitting using least squares optimization
- Core detection within cladding region
- Robust multi-method fallback system
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import argparse
import json

# Try to import circle fitting library
try:
    import circle_fit as cf
    CIRCLE_FIT_AVAILABLE = True
except ImportError:
    CIRCLE_FIT_AVAILABLE = False
    logging.warning("circle-fit library not available. Using fallback methods.")


class FiberLocalizer:
    """Class for locating fiber boundaries in end-face images."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fiber localizer.
        
        Args:
            config: Configuration dictionary with localization parameters
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            "hough_dp": 1.2,
            "hough_min_dist_factor": 0.15,
            "hough_param1": 70,
            "hough_param2": 35,
            "hough_min_radius_factor": 0.08,
            "hough_max_radius_factor": 0.45,
            "adaptive_thresh_block_size": 31,
            "adaptive_thresh_C": 5,
            "template_match_threshold": 0.6,
            "core_search_factor": 0.7  # Core is typically 70% of cladding radius
        }
    
    def locate_fiber_structure(
        self,
        processed_image: np.ndarray,
        original_image: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Locate fiber cladding and core boundaries.
        
        Args:
            processed_image: Preprocessed grayscale image
            original_image: Original grayscale image for core detection
            
        Returns:
            Dictionary with localization results or None if failed
        """
        if processed_image is None or processed_image.size == 0:
            raise ValueError("Input image is empty or None")
        
        if len(processed_image.shape) != 2:
            raise ValueError("Input must be a grayscale image")
        
        h, w = processed_image.shape
        min_img_dim = min(h, w)
        
        # Calculate HoughCircles parameters
        hough_params = self._calculate_hough_parameters(min_img_dim)
        
        localization_result = {}
        
        # Method 1: HoughCircles detection
        cladding_found = self._detect_cladding_hough(processed_image, hough_params, localization_result)
        
        # Method 2: Template matching (if HoughCircles failed)
        if not cladding_found:
            logging.info("Attempting template matching for cladding detection")
            cladding_found = self._detect_cladding_template(processed_image, min_img_dim, localization_result)
        
        # Method 3: Adaptive thresholding + contour fitting (final fallback)
        if not cladding_found:
            logging.info("Attempting adaptive thresholding + contour fitting")
            image_for_thresh = original_image if original_image is not None else processed_image
            cladding_found = self._detect_cladding_contour(image_for_thresh, hough_params, localization_result)
        
        # Method 4: Circle fitting optimization (if available and other methods found candidates)
        if cladding_found and CIRCLE_FIT_AVAILABLE:
            self._refine_with_circle_fit(processed_image, localization_result)
        
        # Detect core within cladding region
        if cladding_found:
            self._detect_core(processed_image, localization_result)
        
        return localization_result if cladding_found else None
    
    def _calculate_hough_parameters(self, min_img_dim: int) -> Dict[str, Any]:
        """Calculate HoughCircles parameters based on image size."""
        config = self.config
        
        return {
            "dp": config["hough_dp"],
            "min_dist": int(min_img_dim * config["hough_min_dist_factor"]),
            "param1": config["hough_param1"],
            "param2": config["hough_param2"],
            "min_radius": int(min_img_dim * config["hough_min_radius_factor"]),
            "max_radius": int(min_img_dim * config["hough_max_radius_factor"])
        }
    
    def _detect_cladding_hough(
        self,
        image: np.ndarray,
        hough_params: Dict[str, Any],
        result: Dict[str, Any]
    ) -> bool:
        """
        Detect cladding using HoughCircles transform.
        
        Args:
            image: Input grayscale image
            hough_params: HoughCircles parameters
            result: Dictionary to store results
            
        Returns:
            True if cladding detected, False otherwise
        """
        logging.debug(f"HoughCircles parameters: {hough_params}")
        
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=hough_params["dp"],
            minDist=hough_params["min_dist"],
            param1=hough_params["param1"],
            param2=hough_params["param2"],
            minRadius=hough_params["min_radius"],
            maxRadius=hough_params["max_radius"]
        )
        
        if circles is not None:
            circles_int = np.uint16(np.around(circles))
            logging.info(f"HoughCircles detected {circles.shape[1]} circle(s)")
            
            # Select best circle using heuristics
            best_circle = self._select_best_circle(circles_int[0, :], image.shape)
            
            if best_circle is not None:
                cx, cy, r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
                result['cladding_center_xy'] = (cx, cy)
                result['cladding_radius_px'] = float(r)
                result['localization_method'] = 'HoughCircles'
                result['confidence'] = self._calculate_circle_confidence(image, cx, cy, r)
                
                logging.info(f"Cladding (Hough): Center=({cx},{cy}), Radius={r}px")
                return True
        
        logging.warning("HoughCircles found no suitable circles")
        return False
    
    def _detect_cladding_template(
        self,
        image: np.ndarray,
        min_img_dim: int,
        result: Dict[str, Any]
    ) -> bool:
        """
        Detect cladding using template matching.
        
        Args:
            image: Input grayscale image
            min_img_dim: Minimum image dimension
            result: Dictionary to store results
            
        Returns:
            True if cladding detected, False otherwise
        """
        if image.shape[0] < 100 or image.shape[1] < 100:
            return False
        
        # Create circular template
        template_radius = int(min_img_dim * 0.3)
        template = np.zeros((template_radius * 2, template_radius * 2), dtype=np.uint8)
        cv2.circle(template, (template_radius, template_radius), template_radius, (255,), -1)
        
        best_match_val = 0
        best_match_loc = None
        best_match_scale = 1.0
        
        # Try different scales
        for scale in np.linspace(0.5, 1.5, 11):
            scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
            
            if (scaled_template.shape[0] > image.shape[0] or 
                scaled_template.shape[1] > image.shape[1]):
                continue
            
            # Perform template matching
            match_result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match_result)
            
            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_match_scale = scale
        
        # Check if match quality is sufficient
        threshold = self.config["template_match_threshold"]
        if best_match_val > threshold and best_match_loc is not None:
            detected_radius = int(template_radius * best_match_scale)
            detected_center = (best_match_loc[0] + detected_radius, 
                             best_match_loc[1] + detected_radius)
            
            result['cladding_center_xy'] = detected_center
            result['cladding_radius_px'] = float(detected_radius)
            result['localization_method'] = 'TemplateMatching'
            result['confidence'] = float(best_match_val)
            
            logging.info(f"Cladding (Template): Center={detected_center}, "
                        f"Radius={detected_radius}px, Confidence={best_match_val:.3f}")
            return True
        
        return False
    
    def _detect_cladding_contour(
        self,
        image: np.ndarray,
        hough_params: Dict[str, Any],
        result: Dict[str, Any]
    ) -> bool:
        """
        Detect cladding using adaptive thresholding and contour fitting.
        
        Args:
            image: Input grayscale image
            hough_params: Parameters for size validation
            result: Dictionary to store results
            
        Returns:
            True if cladding detected, False otherwise
        """
        config = self.config
        block_size = config["adaptive_thresh_block_size"]
        C = config["adaptive_thresh_C"]
        
        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        # Apply adaptive thresholding
        thresh_img = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
        
        # Morphological operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Fill holes
        try:
            from scipy import ndimage
            closed_binary = (closed // 255).astype(np.uint8)
            filled = ndimage.binary_fill_holes(closed_binary)
            filled = filled.astype(np.uint8) * 255
        except ImportError:
            logging.warning("scipy not available, skipping hole filling")
            filled = closed
        except Exception:
            filled = closed
        
        # Opening to clean up
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Filter and select best contour
        valid_contours = self._filter_contours(list(contours), hough_params)
        
        if valid_contours:
            best_contour = max(valid_contours, key=cv2.contourArea)
            
            # Fit circle to contour
            (cx, cy), radius = cv2.minEnclosingCircle(best_contour)
            
            result['cladding_center_xy'] = (float(cx), float(cy))
            result['cladding_radius_px'] = float(radius)
            result['localization_method'] = 'ContourFitting'
            result['confidence'] = self._calculate_contour_confidence(best_contour)
            
            logging.info(f"Cladding (Contour): Center=({cx:.1f},{cy:.1f}), Radius={radius:.1f}px")
            return True
        
        return False
    
    def _refine_with_circle_fit(self, image: np.ndarray, result: Dict[str, Any]):
        """
        Refine circle detection using least squares circle fitting.
        
        Args:
            image: Input grayscale image
            result: Dictionary with existing results to refine
        """
        if not CIRCLE_FIT_AVAILABLE:
            return
        
        try:
            cx, cy = result['cladding_center_xy']
            radius = result['cladding_radius_px']
            
            # Extract edge points around the detected circle
            edge_points = self._extract_edge_points(image, cx, cy, radius)
            
            if len(edge_points) > 10:  # Need sufficient points
                # Fit circle using least squares
                xc, yc, r, _ = cf.least_squares_circle(edge_points)
                
                # Update results if improvement is significant
                if abs(r - radius) < radius * 0.2:  # Within 20% of original
                    result['cladding_center_xy'] = (float(xc), float(yc))
                    result['cladding_radius_px'] = float(r)
                    result['localization_method'] += '+CircleFit'
                    
                    logging.info(f"Circle refined: Center=({xc:.1f},{yc:.1f}), Radius={r:.1f}px")
        
        except Exception as e:
            logging.debug(f"Circle fitting failed: {e}")
    
    def _detect_core(self, image: np.ndarray, result: Dict[str, Any]):
        """
        Detect fiber core within the cladding region.
        
        Args:
            image: Input grayscale image
            result: Dictionary with cladding results
        """
        if 'cladding_center_xy' not in result:
            return
        
        cx, cy = result['cladding_center_xy']
        cladding_radius = result['cladding_radius_px']
        
        # Core is typically much smaller than cladding
        max_core_radius = int(cladding_radius * self.config["core_search_factor"])
        min_core_radius = max(3, int(cladding_radius * 0.1))
        
        # Create mask for core search region
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(cladding_radius * 0.8), (255,), -1)
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, mask)
        
        # Try HoughCircles for core detection
        core_circles = cv2.HoughCircles(
            masked_image,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=int(cladding_radius * 0.5),
            param1=50,
            param2=20,
            minRadius=min_core_radius,
            maxRadius=max_core_radius
        )
        
        if core_circles is not None:
            # Select closest core to cladding center
            cores = np.uint16(np.around(core_circles))
            best_core = None
            min_dist = float('inf')
            
            for core in cores[0, :]:  # type: ignore
                core_cx, core_cy, core_r = core
                dist = np.sqrt((core_cx - cx)**2 + (core_cy - cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_core = core
            
            if best_core is not None:
                core_cx, core_cy, core_r = best_core
                result['core_center_xy'] = (int(core_cx), int(core_cy))
                result['core_radius_px'] = float(core_r)
                
                logging.info(f"Core detected: Center=({core_cx},{core_cy}), Radius={core_r}px")
        else:
            # Fallback: Assume core at cladding center with typical size
            estimated_core_radius = cladding_radius * 0.1  # Typical single-mode core
            result['core_center_xy'] = (int(cx), int(cy))
            result['core_radius_px'] = float(estimated_core_radius)
            result['core_estimated'] = True
            
            logging.info(f"Core estimated: Center=({cx:.0f},{cy:.0f}), "
                        f"Radius={estimated_core_radius:.1f}px (estimated)")
    
    def _select_best_circle(self, circles: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Select the best circle from HoughCircles results."""
        if len(circles) == 0:
            return None
        
        h, w = image_shape
        img_center_x, img_center_y = w // 2, h // 2
        
        best_circle = None
        best_score = -1
        
        for circle in circles:
            cx, cy, r = circle
            
            # Calculate distance from image center
            dist_to_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
            
            # Scoring: prefer larger circles closer to center
            center_score = 1.0 / (1.0 + dist_to_center / min(h, w))
            size_score = r / max(h, w)
            combined_score = center_score * 0.7 + size_score * 0.3
            
            if combined_score > best_score:
                best_score = combined_score
                best_circle = circle
        
        return best_circle
    
    def _filter_contours(self, contours: List[np.ndarray], hough_params: Dict[str, Any]) -> List[np.ndarray]:
        """Filter contours based on size and circularity."""
        valid_contours = []
        min_area = np.pi * (hough_params["min_radius"]**2) * 0.3
        max_area = np.pi * (hough_params["max_radius"]**2) * 2.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if not (min_area < area < max_area):
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity < 0.5:  # Must be reasonably circular
                continue
            
            valid_contours.append(contour)
        
        return valid_contours
    
    def _calculate_circle_confidence(self, image: np.ndarray, cx: int, cy: int, r: int) -> float:
        """Calculate confidence score for detected circle."""
        # Simple confidence based on edge strength along circle
        angles = np.linspace(0, 2*np.pi, 36)
        edge_strengths = []
        
        for angle in angles:
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                # Calculate local gradient magnitude
                gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(gx[y, x]**2 + gy[y, x]**2)
                edge_strengths.append(magnitude)
        
        return float(np.mean(edge_strengths)) / 255.0 if edge_strengths else 0.0
    
    def _calculate_contour_confidence(self, contour: np.ndarray) -> float:
        """Calculate confidence score for contour-based detection."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter**2)
        return float(min(circularity, 1.0))
    
    def _extract_edge_points(self, image: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
        """Extract edge points around detected circle for circle fitting."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Extract points within annular region around detected circle
        y_indices, x_indices = np.where(edges > 0)
        points = np.column_stack([x_indices, y_indices])
        
        # Filter points within radius range
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        valid_mask = (distances > radius * 0.8) & (distances < radius * 1.2)
        
        return points[valid_mask]


def preprocess_for_localization(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for fiber localization.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Preprocessed image
    """
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Apply Gaussian blur for noise reduction
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return denoised


def visualize_localization_results(
    image: np.ndarray,
    localization_result: Dict[str, Any],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize fiber localization results.
    
    Args:
        image: Original grayscale image
        localization_result: Dictionary with localization results
        save_path: Optional path to save visualization
        
    Returns:
        Annotated image
    """
    # Create color version
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()
    
    # Draw cladding circle
    if 'cladding_center_xy' in localization_result:
        cx, cy = localization_result['cladding_center_xy']
        radius = int(localization_result['cladding_radius_px'])
        
        # Draw cladding boundary
        cv2.circle(result, (int(cx), int(cy)), radius, (0, 255, 0), 2)
        cv2.circle(result, (int(cx), int(cy)), 3, (0, 255, 0), -1)
        
        # Add label
        method = localization_result.get('localization_method', 'Unknown')
        confidence = localization_result.get('confidence', 0.0)
        label = f"Cladding ({method}): R={radius}px, C={confidence:.2f}"
        cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw core circle
    if 'core_center_xy' in localization_result:
        core_cx, core_cy = localization_result['core_center_xy']
        core_radius = int(localization_result['core_radius_px'])
        
        # Draw core boundary
        cv2.circle(result, (int(core_cx), int(core_cy)), core_radius, (0, 0, 255), 2)
        cv2.circle(result, (int(core_cx), int(core_cy)), 2, (0, 0, 255), -1)
        
        # Add label
        core_label = f"Core: R={core_radius}px"
        if localization_result.get('core_estimated', False):
            core_label += " (estimated)"
        cv2.putText(result, core_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, result)
        logging.info(f"Visualization saved to: {save_path}")
    
    return result


def main():
    """Main function for standalone testing."""
    parser = argparse.ArgumentParser(description="Fiber Localization")
    parser.add_argument("input_image", help="Path to input fiber image")
    parser.add_argument("--config", help="Path to JSON configuration file")
    parser.add_argument("--output", "-o", help="Output path for results (JSON)")
    parser.add_argument("--visualize", help="Path to save visualization image")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    try:
        # Load configuration
        config = None
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Load image
        logging.info(f"Loading image: {args.input_image}")
        image = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {args.input_image}")
        
        logging.info(f"Image loaded: {image.shape}")
        
        # Preprocess image
        processed_image = preprocess_for_localization(image)
        
        # Initialize localizer
        localizer = FiberLocalizer(config)
        
        # Perform localization
        logging.info("Performing fiber localization...")
        result = localizer.locate_fiber_structure(processed_image, image)
        
        if result:
            logging.info("Localization successful:")
            
            if 'cladding_center_xy' in result:
                cx, cy = result['cladding_center_xy']
                radius = result['cladding_radius_px']
                method = result.get('localization_method', 'Unknown')
                confidence = result.get('confidence', 0.0)
                
                logging.info(f"  Cladding: Center=({cx:.1f},{cy:.1f}), "
                           f"Radius={radius:.1f}px, Method={method}, Confidence={confidence:.3f}")
            
            if 'core_center_xy' in result:
                core_cx, core_cy = result['core_center_xy']
                core_radius = result['core_radius_px']
                estimated = " (estimated)" if result.get('core_estimated', False) else ""
                
                logging.info(f"  Core: Center=({core_cx:.1f},{core_cy:.1f}), "
                           f"Radius={core_radius:.1f}px{estimated}")
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                logging.info(f"Results saved to: {args.output}")
            
            # Create visualization
            if args.visualize:
                visualize_localization_results(image, result, args.visualize)
            else:
                # Display results
                vis = visualize_localization_results(image, result)
                cv2.imshow("Fiber Localization Results", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        else:
            logging.error("Fiber localization failed")
            return 1
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
