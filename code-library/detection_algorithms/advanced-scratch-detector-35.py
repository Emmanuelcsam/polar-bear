#!/usr/bin/env python3
"""
Advanced Scratch Detection Module
=================================
Standalone module for detecting scratches and linear defects
using multiple advanced techniques.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available, using OpenCV alternatives")

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available")

class AdvancedScratchDetector:
    """Advanced scratch detection using multiple techniques."""
    
    def __init__(self):
        self.methods = {
            'gradient': self._gradient_based_detection,
            'gabor': self._gabor_based_detection,
            'hessian': self._hessian_based_detection,
            'morphological': self._morphological_detection,
            'frequency': self._frequency_based_detection,
            'steerable': self._steerable_filter_detection,
            'frangi': self._frangi_vesselness_detection
        }
        
    def detect_scratches(self, image: np.ndarray, zone_mask: Optional[np.ndarray] = None,
                        methods: List[str] = ['gradient', 'gabor', 'hessian'],
                        fusion_weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Detect scratches using multiple methods.
        
        Args:
            image: Input grayscale image
            zone_mask: Optional zone mask
            methods: List of detection methods to use
            fusion_weights: Weights for method fusion
            
        Returns:
            Final scratch mask and individual method results
        """
        logger.info(f"Running scratch detection with methods: {methods}")
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Default weights
        if fusion_weights is None:
            fusion_weights = {method: 1.0 for method in methods}
        
        # Apply zone mask if provided
        if zone_mask is not None:
            masked_image = image * (zone_mask.astype(np.float32) / 255.0)
        else:
            masked_image = image
        
        # Run selected methods
        individual_results = {}
        combined_map = np.zeros_like(image, dtype=np.float32)
        
        for method in methods:
            if method in self.methods:
                try:
                    result = self.methods[method](masked_image)
                    individual_results[method] = result
                    
                    # Add weighted contribution
                    weight = fusion_weights.get(method, 1.0)
                    if result is not None:
                        combined_map += result.astype(np.float32) * weight
                    
                    logger.info(f"{method} detection completed")
                    
                except Exception as e:
                    logger.error(f"Error in {method} scratch detection: {e}")
                    individual_results[method] = np.zeros_like(image, dtype=np.uint8)
                    continue
        
        # Normalize combined map
        total_weight = sum(fusion_weights.get(m, 1.0) for m in methods if m in individual_results and individual_results[m] is not None)
        if total_weight > 0:
            combined_map /= total_weight
            combined_map = np.clip(combined_map, 0, 1)
        
        # Apply zone mask to final result
        if zone_mask is not None:
            combined_map = combined_map * (zone_mask.astype(np.float32) / 255.0)
        
        # Post-processing
        final_mask = self._postprocess_scratch_mask(combined_map)
        
        logger.info(f"Scratch detection completed, found {cv2.countNonZero(final_mask)} scratch pixels")
        
        return final_mask, individual_results
    
    def _gradient_based_detection(self, image: np.ndarray) -> np.ndarray:
        """Gradient-based scratch detection."""
        # Multi-scale gradient analysis
        scales = [1, 2, 3]
        responses = []
        
        for scale in scales:
            # Smooth image
            if SCIPY_AVAILABLE:
                smoothed = gaussian_filter(image, scale)
            else:
                ksize = int(2 * scale * 3 + 1)
                if ksize % 2 == 0:
                    ksize += 1
                smoothed = cv2.GaussianBlur(image, (ksize, ksize), scale)
            
            # Calculate gradients
            grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
            
            # Gradient magnitude
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Gradient direction coherence
            grad_angle = np.arctan2(grad_y, grad_x)
            
            # Local coherence measure
            coherence = self._calculate_local_coherence(grad_angle, window_size=7)
            
            # Combine magnitude and coherence
            response = grad_mag * coherence
            responses.append(response)
        
        # Combine scales
        combined = np.maximum.reduce(responses)
        
        # Threshold
        threshold = np.percentile(combined, 95)
        return (combined > threshold).astype(np.uint8) * 255
    
    def _gabor_based_detection(self, image: np.ndarray) -> np.ndarray:
        """Gabor filter bank for scratch detection."""
        orientations = 8
        frequencies = [0.1, 0.2, 0.3]
        responses = []
        
        for freq in frequencies:
            for i in range(orientations):
                theta = i * np.pi / orientations
                
                # Create Gabor kernel
                kernel = cv2.getGaborKernel((21, 21), 3, theta, 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                response = cv2.filter2D(image, cv2.CV_32F, kernel)
                responses.append(np.abs(response))
        
        # Take maximum response across all orientations and frequencies
        max_response = np.maximum.reduce(responses)
        
        # Threshold
        threshold = np.percentile(max_response, 98)
        return (max_response > threshold).astype(np.uint8) * 255
    
    def _hessian_based_detection(self, image: np.ndarray) -> np.ndarray:
        """Hessian-based ridge detection for scratches."""
        scales = [1.0, 2.0, 3.0]
        responses = []
        
        for scale in scales:
            # Calculate Hessian matrix components using OpenCV
            if SCIPY_AVAILABLE:
                # Use scipy gaussian_filter if available
                sigma = scale
                Ixx = gaussian_filter(image, sigma, order=[0, 2])
                Ixy = gaussian_filter(image, sigma, order=[1, 1])
                Iyy = gaussian_filter(image, sigma, order=[2, 0])
            else:
                # Use OpenCV alternative
                ksize = int(2 * scale * 3 + 1)  # Approximate kernel size
                if ksize % 2 == 0:
                    ksize += 1
                
                # Apply Gaussian blur first
                blurred = cv2.GaussianBlur(image, (ksize, ksize), scale)
                
                # Calculate second derivatives using Sobel
                Ix = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
                Iy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
                
                Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=3)
                Ixy = cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=3)
                Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=3)
            
            # Eigenvalues of Hessian matrix
            trace = Ixx + Iyy
            det = Ixx * Iyy - Ixy * Ixy
            
            discriminant = np.sqrt(np.maximum(trace**2 - 4*det, 0))
            lambda1 = 0.5 * (trace + discriminant)
            lambda2 = 0.5 * (trace - discriminant)
            
            # Ridge strength (for dark ridges on bright background)
            ridge_strength = np.abs(lambda2) * (lambda2 < 0)
            
            # Scale normalization
            ridge_strength *= scale**2
            
            responses.append(ridge_strength)
        
        # Maximum response across scales
        max_response = np.maximum.reduce(responses)
        
        # Threshold
        threshold = np.percentile(max_response, 98)
        return (max_response > threshold).astype(np.uint8) * 255
    
    def _morphological_detection(self, image: np.ndarray) -> np.ndarray:
        """Morphological operations for scratch detection."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Multiple line-shaped structuring elements
        responses = []
        
        # Different line orientations
        angles = np.arange(0, 180, 15)
        
        for angle in angles:
            # Create linear structuring element
            length = 15
            kernel = self._create_linear_kernel(length, float(angle))
            
            # Apply opening (removes structures smaller than kernel)
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            # Difference between original and opened
            diff = cv2.subtract(image, opened)
            
            responses.append(diff)
        
        # Combine responses
        combined = np.maximum.reduce(responses)
        
        # Threshold
        _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _frequency_based_detection(self, image: np.ndarray) -> np.ndarray:
        """Frequency domain scratch detection."""
        # Apply FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create directional filters in frequency domain
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        
        responses = []
        
        # Different orientations
        angles = np.arange(0, 180, 30)
        
        for angle in angles:
            # Create directional filter
            angle_rad = np.radians(angle)
            
            # Line in frequency domain (perpendicular to spatial line)
            perp_angle = angle_rad + np.pi/2
            
            # Distance from line through center
            dist = np.abs((x - center_x) * np.cos(perp_angle) + (y - center_y) * np.sin(perp_angle))
            
            # Create band-pass filter
            filter_mask = np.exp(-dist**2 / (2 * 5**2))  # Gaussian band
            
            # Apply filter
            filtered_freq = f_shift * filter_mask
            
            # Inverse FFT
            filtered_spatial = np.fft.ifft2(np.fft.ifftshift(filtered_freq))
            response = np.abs(filtered_spatial)
            
            responses.append(response)
        
        # Combine responses
        combined = np.maximum.reduce(responses)
        
        # Threshold
        threshold = np.percentile(combined, 95)
        return (combined > threshold).astype(np.uint8) * 255
    
    def _steerable_filter_detection(self, image: np.ndarray) -> np.ndarray:
        """Steerable filter-based scratch detection."""
        # Create steerable filter bank
        angles = np.arange(0, 180, 15)
        responses = []
        
        for angle in angles:
            # Create steerable filter kernel
            kernel = self._create_steerable_filter(float(angle), size=15, sigma=2.0)
            
            # Apply filter
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(np.abs(response))
        
        # Maximum response across orientations
        max_response = np.maximum.reduce(responses)
        
        # Threshold
        threshold = np.percentile(max_response, 98)
        return (max_response > threshold).astype(np.uint8) * 255
    
    def _frangi_vesselness_detection(self, image: np.ndarray) -> np.ndarray:
        """Frangi vesselness filter for scratch detection."""
        scales = [1.0, 2.0, 3.0, 4.0]
        responses = []
        
        beta = 0.5  # Controls sensitivity to blob-like structures
        c = 15      # Controls sensitivity to background
        
        for scale in scales:
            # Calculate Hessian matrix
            if SCIPY_AVAILABLE:
                sigma = scale
                Ixx = gaussian_filter(image, sigma, order=[0, 2])
                Ixy = gaussian_filter(image, sigma, order=[1, 1])
                Iyy = gaussian_filter(image, sigma, order=[2, 0])
            else:
                # Use OpenCV alternative
                ksize = int(2 * scale * 3 + 1)
                if ksize % 2 == 0:
                    ksize += 1
                
                blurred = cv2.GaussianBlur(image, (ksize, ksize), scale)
                Ix = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
                Iy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
                Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=3)
                Ixy = cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=3)
                Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=3)
            
            # Eigenvalues
            trace = Ixx + Iyy
            det = Ixx * Iyy - Ixy * Ixy
            
            discriminant = np.sqrt(np.maximum(trace**2 - 4*det, 0))
            lambda1 = 0.5 * (trace + discriminant)
            lambda2 = 0.5 * (trace - discriminant)
            
            # Sort eigenvalues (|lambda1| >= |lambda2|)
            abs_lambda1 = np.abs(lambda1)
            abs_lambda2 = np.abs(lambda2)
            
            # Vesselness measures
            with np.errstate(divide='ignore', invalid='ignore'):
                RB = abs_lambda2 / abs_lambda1
                S = np.sqrt(lambda1**2 + lambda2**2)
            
            # Replace invalid values
            RB = np.nan_to_num(RB)
            S = np.nan_to_num(S)
            
            # Frangi vesselness
            vesselness = np.exp(-RB**2 / (2 * beta**2)) * (1 - np.exp(-S**2 / (2 * c**2)))
            
            # Only consider dark ridges (lambda2 < 0)
            vesselness[lambda2 > 0] = 0
            
            # Scale normalization
            vesselness *= scale**2
            
            responses.append(vesselness)
        
        # Maximum response across scales
        max_response = np.maximum.reduce(responses)
        
        # Threshold
        threshold = np.percentile(max_response, 98)
        return (max_response > threshold).astype(np.uint8) * 255
    
    def _postprocess_scratch_mask(self, combined_map: np.ndarray) -> np.ndarray:
        """Post-process the combined scratch detection result."""
        # Convert to binary
        threshold = 0.3  # Conservative threshold
        binary = (combined_map > threshold).astype(np.uint8) * 255
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Connect broken lines
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Filter by line properties
        binary = self._filter_by_line_properties(binary)
        
        return binary
    
    def _filter_by_line_properties(self, binary: np.ndarray, 
                                  min_aspect_ratio: float = 3.0, 
                                  min_length: int = 10) -> np.ndarray:
        """Filter detections based on line properties."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = np.zeros_like(binary)
        
        for contour in contours:
            # Calculate properties
            area = cv2.contourArea(contour)
            if area < 5:  # Too small
                continue
            
            # Bounding rectangle
            rect = cv2.boundingRect(contour)
            width, height = rect[2], rect[3]
            
            # Aspect ratio
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            
            # Length (major axis)
            length = max(width, height)
            
            # Check if it's line-like
            if aspect_ratio >= min_aspect_ratio and length >= min_length:
                cv2.drawContours(result, [contour], -1, (255,), -1)
        
        return result
    
    def _calculate_local_coherence(self, angle_map: np.ndarray, window_size: int = 7) -> np.ndarray:
        """Calculate local orientation coherence."""
        # Convert angles to unit vectors
        cos_angles = np.cos(2 * angle_map)  # Double angle for orientation
        sin_angles = np.sin(2 * angle_map)
        
        # Local averaging
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        
        mean_cos = cv2.filter2D(cos_angles, cv2.CV_32F, kernel)
        mean_sin = cv2.filter2D(sin_angles, cv2.CV_32F, kernel)
        
        # Coherence measure
        coherence = np.sqrt(mean_cos**2 + mean_sin**2)
        
        return coherence
    
    def _create_linear_kernel(self, length: int, angle: float) -> np.ndarray:
        """Create a linear morphological kernel."""
        angle_rad = np.radians(angle)
        
        # Create kernel
        kernel = np.zeros((length, length), dtype=np.uint8)
        center = length // 2
        
        # Draw line in kernel
        for i in range(length):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
        
        return kernel
    
    def _create_steerable_filter(self, angle: float, size: int = 15, sigma: float = 2.0) -> np.ndarray:
        """Create steerable filter kernel."""
        angle_rad = np.radians(angle)
        
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                
                # Rotate coordinates
                x_rot = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
                y_rot = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
                
                # Gaussian derivative
                g = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
                val = (x_rot**2 / sigma**2 - 1) * g / (2 * np.pi * sigma**4)
                
                kernel[y, x] = val
        
        # Normalize
        kernel = kernel - np.mean(kernel)
        
        return kernel

def test_scratch_detection():
    """Test the scratch detection module."""
    logger.info("Testing advanced scratch detection...")
    
    # Create synthetic test image with scratches
    test_image = np.ones((256, 256), dtype=np.float32) * 0.5
    
    # Add synthetic scratches
    # Horizontal scratch
    test_image[100:103, 50:200] = 0.2
    
    # Diagonal scratch
    for i in range(150):
        x = 50 + i
        y = 150 + int(i * 0.5)
        if 0 <= x < 256 and 0 <= y < 256:
            test_image[max(0, y-1):min(256, y+2), max(0, x-1):min(256, x+2)] = 0.15
    
    # Vertical scratch
    test_image[30:180, 200:202] = 0.25
    
    # Add noise
    noise = np.random.normal(0, 0.05, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 1)
    
    logger.info(f"Test image shape: {test_image.shape}")
    
    # Initialize detector
    detector = AdvancedScratchDetector()
    
    # Test different method combinations
    method_sets = [
        ['gradient', 'gabor'],
        ['hessian', 'frangi'],
        ['morphological', 'steerable'],
        ['gradient', 'gabor', 'hessian', 'frangi']
    ]
    
    results = {}
    
    for i, methods in enumerate(method_sets):
        logger.info(f"Testing method set {i+1}: {methods}")
        
        try:
            final_mask, individual_results = detector.detect_scratches(
                test_image, methods=methods
            )
            
            scratch_count = cv2.countNonZero(final_mask)
            logger.info(f"Method set {i+1}: {scratch_count} scratch pixels detected")
            
            results[f"set_{i+1}"] = {
                'methods': methods,
                'final_mask': final_mask,
                'individual_results': individual_results,
                'scratch_count': scratch_count
            }
            
        except Exception as e:
            logger.error(f"Error testing method set {i+1}: {e}")
    
    logger.info("All scratch detection tests completed!")
    
    return {
        'test_image': test_image,
        'results': results
    }

if __name__ == "__main__":
    # Run tests
    test_results = test_scratch_detection()
    logger.info("Advanced scratch detection module is ready for use!")
