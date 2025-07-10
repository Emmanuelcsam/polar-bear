#!/usr/bin/env python3
"""
Gradient-Based Fiber Segmentation Module
Universal fiber optic endface segmentation using multiple robust methods.
Does not rely on Hough circles - uses gradients, brightness analysis, and morphology.
"""

import cv2
import numpy as np
import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import warnings

# Optional scipy imports
try:
    from scipy import ndimage
    from scipy.signal import find_peaks, savgol_filter
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: SciPy not available. Some gradient-based functions will use fallbacks.")

warnings.filterwarnings('ignore')


class GradientFiberSegmenter:
    """
    Robust fiber optic segmentation using gradient analysis and multiple detection methods
    """
    
    def __init__(self,
                 clahe_clip_limit: float = 3.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8),
                 brightness_percentile: int = 85,
                 small_object_threshold: int = 100,
                 gaussian_sigma: float = 2.0,
                 gradient_threshold: float = 10.0):
        """
        Initialize the gradient fiber segmenter
        
        Args:
            clahe_clip_limit: CLAHE contrast limiting threshold
            clahe_tile_size: CLAHE tile grid size
            brightness_percentile: Percentile for brightness thresholding
            small_object_threshold: Minimum size for object filtering
            gaussian_sigma: Sigma for Gaussian smoothing
            gradient_threshold: Threshold for gradient-based edge detection
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.brightness_percentile = brightness_percentile
        self.small_object_threshold = small_object_threshold
        self.gaussian_sigma = gaussian_sigma
        self.gradient_threshold = gradient_threshold
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to enhance fiber features
        
        Args:
            image: Input grayscale image
            
        Returns:
            Enhanced image
        """
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, 
            tileGridSize=self.clahe_tile_size
        )
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def find_brightness_centroid(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Find center using brightness-weighted centroid
        
        Args:
            image: Enhanced grayscale image
            
        Returns:
            Tuple of (center_point, confidence)
        """
        try:
            # Use top percentile of brightness
            threshold = np.percentile(image, self.brightness_percentile)
            bright_mask = image > threshold
            
            # Convert to proper boolean type
            bright_mask = bright_mask.astype(bool)
            
            # Remove small objects if mask is valid
            if bright_mask.any():
                try:
                    from skimage.morphology import remove_small_objects
                    cleaned_mask = remove_small_objects(bright_mask, min_size=self.small_object_threshold)
                    if cleaned_mask is not None and cleaned_mask.any():
                        bright_mask = cleaned_mask
                except ImportError:
                    # If skimage not available, use OpenCV morphology
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    bright_mask_uint8 = (bright_mask * 255).astype(np.uint8)
                    bright_mask_uint8 = cv2.morphologyEx(bright_mask_uint8, cv2.MORPH_OPEN, kernel)
                    bright_mask = bright_mask_uint8 > 0
                except Exception:
                    # Continue with original mask if filtering fails
                    pass
            
            # Convert to uint8 for cv2.moments
            bright_mask_uint8 = (bright_mask * 255).astype(np.uint8)
            
            if np.sum(bright_mask_uint8) == 0:
                return None, 0.0
            
            # Calculate centroid using moments
            M = cv2.moments(bright_mask_uint8)
            if M['m00'] == 0:
                return None, 0.0
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Calculate confidence based on compactness of bright region
            y_coords, x_coords = np.where(bright_mask)
            if len(x_coords) > 0:
                spread = np.std(np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2))
                confidence = 1.0 / (1.0 + spread / 100)
            else:
                confidence = 0.5
            
            return (cx, cy), confidence
            
        except Exception as e:
            print(f"Warning: Brightness centroid calculation failed: {e}")
            return None, 0.0
    
    def find_gradient_center(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Find center using circular gradient optimization
        
        Args:
            image: Enhanced grayscale image
            
        Returns:
            Tuple of (center_point, confidence)
        """
        try:
            # Apply Gaussian smoothing
            smoothed = ndimage.gaussian_filter(image.astype(np.float32), sigma=self.gaussian_sigma)
            
            # Calculate gradients
            grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Find points with significant gradients
            significant_gradients = grad_magnitude > self.gradient_threshold
            y_coords, x_coords = np.where(significant_gradients)
            
            if len(x_coords) < 10:  # Need sufficient points
                return None, 0.0
            
            # Initial guess - image center
            h, w = image.shape
            initial_center = (w // 2, h // 2)
            
            # Objective function: minimize sum of radial gradient inconsistency
            def objective(center):
                cx, cy = center
                
                # Calculate radial distances
                distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
                
                # Calculate expected radial gradients (pointing outward for bright center)
                expected_grad_x = (x_coords - cx) / (distances + 1e-8)
                expected_grad_y = (y_coords - cy) / (distances + 1e-8)
                
                # Get actual gradients at these points
                actual_grad_x = grad_x[y_coords, x_coords]
                actual_grad_y = grad_y[y_coords, x_coords]
                
                # Normalize actual gradients
                actual_magnitude = np.sqrt(actual_grad_x**2 + actual_grad_y**2) + 1e-8
                actual_grad_x_norm = actual_grad_x / actual_magnitude
                actual_grad_y_norm = actual_grad_y / actual_magnitude
                
                # Calculate alignment (dot product)
                alignment = (expected_grad_x * actual_grad_x_norm + 
                           expected_grad_y * actual_grad_y_norm)
                
                # Minimize negative alignment (maximize alignment)
                return -np.mean(alignment)
            
            # Optimize center position
            bounds = [(0, w-1), (0, h-1)]
            result = minimize(objective, initial_center, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                center = (int(result.x[0]), int(result.x[1]))
                confidence = min(1.0, -result.fun)  # Convert back to positive
                return center, max(0.0, confidence)
            else:
                return None, 0.0
                
        except Exception as e:
            print(f"Warning: Gradient center calculation failed: {e}")
            return None, 0.0
    
    def find_morphological_center(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Find center using morphological operations
        
        Args:
            image: Enhanced grayscale image
            
        Returns:
            Tuple of (center_point, confidence)
        """
        try:
            # Apply threshold to create binary image
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, 0.0
            
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate moments
            M = cv2.moments(largest_contour)
            if M['m00'] == 0:
                return None, 0.0
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Calculate confidence based on contour properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                # Circularity measure (1.0 = perfect circle)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                confidence = min(1.0, circularity)
            else:
                confidence = 0.5
            
            return (cx, cy), confidence
            
        except Exception as e:
            print(f"Warning: Morphological center calculation failed: {e}")
            return None, 0.0
    
    def find_edge_based_center(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Find center using edge analysis
        
        Args:
            image: Enhanced grayscale image
            
        Returns:
            Tuple of (center_point, confidence)
        """
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Find edge points
            y_coords, x_coords = np.where(edges > 0)
            
            if len(x_coords) < 10:
                return None, 0.0
            
            # Try to find circular patterns by analyzing edge point distances
            h, w = image.shape
            best_center = None
            best_score = 0
            
            # Sample potential centers
            for cy in range(h // 4, 3 * h // 4, h // 8):
                for cx in range(w // 4, 3 * w // 4, w // 8):
                    
                    # Calculate distances from this center to all edge points
                    distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
                    
                    # Look for peaks in distance histogram (indicating circular structures)
                    hist, bins = np.histogram(distances, bins=50)
                    
                    # Find peaks in histogram
                    try:
                        peaks, _ = find_peaks(hist, height=len(x_coords) * 0.01)
                        
                        if len(peaks) >= 1:  # At least one prominent circle
                            # Score based on peak prominence
                            peak_heights = hist[peaks]
                            score = np.max(peak_heights) / len(x_coords)
                            
                            if score > best_score:
                                best_score = score
                                best_center = (cx, cy)
                    except:
                        continue
            
            if best_center is not None:
                confidence = min(1.0, best_score * 10)  # Scale confidence
                return best_center, confidence
            else:
                return None, 0.0
                
        except Exception as e:
            print(f"Warning: Edge-based center calculation failed: {e}")
            return None, 0.0
    
    def find_center_multi_method(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Find center using multiple methods and combine results
        
        Args:
            image: Enhanced grayscale image
            
        Returns:
            Combined center coordinates
        """
        centers = []
        weights = []
        
        # Method 1: Brightness-weighted centroid
        center1, conf1 = self.find_brightness_centroid(image)
        if center1 is not None and conf1 > 0:
            centers.append(center1)
            weights.append(conf1)
        
        # Method 2: Circular gradient optimization
        center2, conf2 = self.find_gradient_center(image)
        if center2 is not None and conf2 > 0:
            centers.append(center2)
            weights.append(conf2)
        
        # Method 3: Morphological center
        center3, conf3 = self.find_morphological_center(image)
        if center3 is not None and conf3 > 0:
            centers.append(center3)
            weights.append(conf3)
        
        # Method 4: Edge-based center estimation
        center4, conf4 = self.find_edge_based_center(image)
        if center4 is not None and conf4 > 0:
            centers.append(center4)
            weights.append(conf4)
        
        if not centers:
            # Fallback to image center
            h, w = image.shape
            return (w // 2, h // 2)
        else:
            # Weighted average of all detected centers
            centers = np.array(centers)
            weights = np.array(weights)
            weights = weights / weights.sum()
            center = np.average(centers, axis=0, weights=weights).astype(int)
            return tuple(center)
    
    def estimate_radii_from_gradients(self, image: np.ndarray, center: Tuple[int, int]) -> Tuple[Optional[int], Optional[int]]:
        """
        Estimate core and cladding radii using gradient analysis
        
        Args:
            image: Enhanced grayscale image
            center: Center coordinates
            
        Returns:
            Tuple of (core_radius, cladding_radius)
        """
        try:
            cx, cy = center
            h, w = image.shape
            
            # Create radial profile
            max_radius = min(w, h) // 2
            radii = np.arange(1, max_radius)
            intensities = []
            
            for r in radii:
                # Sample points on circle at this radius
                angles = np.linspace(0, 2 * np.pi, max(8, int(2 * np.pi * r)))
                x_coords = cx + r * np.cos(angles)
                y_coords = cy + r * np.sin(angles)
                
                # Keep points within image bounds
                valid_mask = ((x_coords >= 0) & (x_coords < w) & 
                             (y_coords >= 0) & (y_coords < h))
                
                if np.any(valid_mask):
                    x_coords = x_coords[valid_mask].astype(int)
                    y_coords = y_coords[valid_mask].astype(int)
                    
                    # Average intensity at this radius
                    avg_intensity = np.mean(image[y_coords, x_coords])
                    intensities.append(avg_intensity)
                else:
                    intensities.append(0)
            
            intensities = np.array(intensities)
            
            if len(intensities) < 10:
                return None, None
            
            # Smooth the profile
            try:
                from scipy.signal import savgol_filter
                if len(intensities) >= 5:
                    smoothed = savgol_filter(intensities, 
                                           min(11, len(intensities) // 2 * 2 + 1), 2)
                else:
                    smoothed = intensities
            except:
                smoothed = intensities
            
            # Find significant changes (potential boundaries)
            gradient = np.gradient(smoothed)
            
            # Find peaks and valleys in gradient (boundaries)
            try:
                from scipy.signal import find_peaks
                
                # Find negative peaks (decreasing intensity - potential core boundary)
                neg_peaks, _ = find_peaks(-gradient, height=np.std(gradient) * 0.5)
                
                # Find positive peaks (increasing intensity - potential cladding boundary)
                pos_peaks, _ = find_peaks(gradient, height=np.std(gradient) * 0.5)
                
                core_radius = None
                cladding_radius = None
                
                # Core typically appears as first significant negative gradient
                if len(neg_peaks) > 0:
                    core_radius = int(radii[neg_peaks[0]])
                
                # Cladding boundary might be where intensity starts dropping again
                if len(pos_peaks) > 0:
                    # Look for the last significant positive peak
                    cladding_radius = int(radii[pos_peaks[-1]])
                elif len(neg_peaks) > 1:
                    # Or second negative peak
                    cladding_radius = int(radii[neg_peaks[-1]])
                
                # Sanity checks
                if core_radius and cladding_radius:
                    if core_radius >= cladding_radius:
                        # Swap if needed
                        core_radius, cladding_radius = cladding_radius, core_radius
                    
                    # Ensure reasonable ratios
                    if cladding_radius / core_radius > 10:
                        core_radius = cladding_radius // 6  # Typical ratio
                
                return core_radius, cladding_radius
                
            except:
                # Fallback estimation
                # Assume core is around 1/6 of image width, cladding is 1/3
                core_radius = min(w, h) // 12
                cladding_radius = min(w, h) // 6
                return core_radius, cladding_radius
                
        except Exception as e:
            print(f"Warning: Radii estimation failed: {e}")
            return None, None
    
    def segment_fiber(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Main function to segment fiber using gradient-based methods
        
        Args:
            image_path: Path to input fiber image
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary containing segmentation results
        """
        result = {
            'method': 'gradient_approach',
            'image_path': image_path,
            'success': False,
            'center': None,
            'core_radius': None,
            'cladding_radius': None,
            'confidence': 0.0,
            'methods_used': []
        }
        
        # Validate input
        if not Path(image_path).exists():
            result['error'] = f"Image not found: {image_path}"
            return result
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            result['error'] = f"Could not read image: {image_path}"
            return result
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        try:
            # Preprocess image
            enhanced = self.preprocess_image(gray)
            
            # Find center using multiple methods
            center = self.find_center_multi_method(enhanced)
            result['center'] = center
            
            # Estimate radii
            core_radius, cladding_radius = self.estimate_radii_from_gradients(enhanced, center)
            
            if core_radius is not None:
                result['core_radius'] = core_radius
                result['success'] = True
                result['confidence'] = 0.6
            
            if cladding_radius is not None:
                result['cladding_radius'] = cladding_radius
                result['confidence'] = 0.8
            
            # If no radii found, use estimates
            if core_radius is None and cladding_radius is None:
                h, w = gray.shape
                result['core_radius'] = min(w, h) // 12
                result['cladding_radius'] = min(w, h) // 6
                result['success'] = True
                result['confidence'] = 0.3
            
            # Save results if output directory specified
            if output_dir and result['success']:
                self._save_results(image, result, enhanced, output_dir, image_path)
                
        except Exception as e:
            result['error'] = f"Processing error: {str(e)}"
        
        return result
    
    def _save_results(self, original_image: np.ndarray, result: Dict[str, Any],
                     enhanced_image: np.ndarray, output_dir: str, image_path: str):
        """Save segmentation results"""
        os.makedirs(output_dir, exist_ok=True)
        base_filename = Path(image_path).stem
        
        # Save enhanced image
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_enhanced.png"), enhanced_image)
        
        # Create visualization
        if len(original_image.shape) == 3:
            vis_image = original_image.copy()
        else:
            vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        center = result['center']
        core_radius = result.get('core_radius')
        cladding_radius = result.get('cladding_radius')
        
        # Draw center
        cv2.circle(vis_image, center, 3, (0, 255, 0), -1)
        
        # Draw core
        if core_radius:
            cv2.circle(vis_image, center, core_radius, (255, 0, 0), 2)
            cv2.putText(vis_image, f"Core: {core_radius}px", 
                       (center[0] - 50, center[1] - core_radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw cladding
        if cladding_radius:
            cv2.circle(vis_image, center, cladding_radius, (0, 255, 0), 2)
            cv2.putText(vis_image, f"Cladding: {cladding_radius}px", 
                       (center[0] - 70, center[1] - cladding_radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_result.png"), vis_image)
        
        # Save JSON result
        with open(os.path.join(output_dir, f"{base_filename}_gradient_result.json"), 'w') as f:
            json.dump(result, f, indent=4)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Gradient-Based Fiber Segmentation')
    parser.add_argument('image_path', help='Path to input fiber image')
    parser.add_argument('--output-dir', default='output_gradient',
                       help='Output directory for results')
    parser.add_argument('--clahe-clip', type=float, default=3.0,
                       help='CLAHE clip limit')
    parser.add_argument('--brightness-percentile', type=int, default=85,
                       help='Percentile for brightness thresholding')
    parser.add_argument('--gradient-threshold', type=float, default=10.0,
                       help='Threshold for gradient-based edge detection')
    
    args = parser.parse_args()
    
    # Create segmenter
    segmenter = GradientFiberSegmenter(
        clahe_clip_limit=args.clahe_clip,
        brightness_percentile=args.brightness_percentile,
        gradient_threshold=args.gradient_threshold
    )
    
    # Process image
    result = segmenter.segment_fiber(args.image_path, args.output_dir)
    
    # Print results
    print(f"Fiber segmentation {'successful' if result['success'] else 'failed'}")
    if result['success']:
        print(f"Center: {result['center']}")
        if result.get('core_radius'):
            print(f"Core radius: {result['core_radius']} pixels")
        if result.get('cladding_radius'):
            print(f"Cladding radius: {result['cladding_radius']} pixels")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
