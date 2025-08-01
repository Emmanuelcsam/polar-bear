#!/usr/bin/env python3
"""
Ensemble Detection Methods Module
=================================

This module provides intelligent ensemble methods for combining multiple
defect detection algorithms with confidence weighting and validation.

Author: Modular Analysis Team
Version: 1.0
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum, auto


class DetectionMethod(Enum):
    """Enumeration of detection methods."""
    DO2MR = auto()
    LEI = auto()
    GRADIENT = auto()
    LAPLACIAN = auto()
    CANNY = auto()
    THRESHOLD = auto()
    MORPHOLOGICAL = auto()
    STATISTICAL = auto()


@dataclass
class DetectionResult:
    """Result from a single detection method."""
    method: DetectionMethod
    detection_mask: np.ndarray
    confidence_map: Optional[np.ndarray] = None
    method_confidence: float = 1.0
    parameters: Optional[Dict[str, Any]] = None


class EnsembleDetector:
    """
    Intelligent ensemble detector that combines multiple detection methods.
    """
    
    def __init__(self, 
                 confidence_weights: Optional[Dict[DetectionMethod, float]] = None,
                 vote_threshold: float = 0.3,
                 min_methods_agreement: int = 2):
        """
        Initialize ensemble detector.
        
        Args:
            confidence_weights: Weights for different detection methods
            vote_threshold: Minimum vote fraction for ensemble detection
            min_methods_agreement: Minimum number of methods that must agree
        """
        self.confidence_weights = confidence_weights or {
            DetectionMethod.DO2MR: 1.0,
            DetectionMethod.LEI: 1.0,
            DetectionMethod.GRADIENT: 0.8,
            DetectionMethod.LAPLACIAN: 0.9,
            DetectionMethod.CANNY: 0.7,
            DetectionMethod.THRESHOLD: 0.6,
            DetectionMethod.MORPHOLOGICAL: 0.8,
            DetectionMethod.STATISTICAL: 0.9
        }
        self.vote_threshold = vote_threshold
        self.min_methods_agreement = min_methods_agreement
        self.logger = logging.getLogger(__name__)
        
        # Initialize detection methods
        self._setup_detection_methods()
    
    def _setup_detection_methods(self):
        """Setup all detection method implementations."""
        self.detection_methods = {
            DetectionMethod.DO2MR: self._do2mr_detection,
            DetectionMethod.LEI: self._lei_detection,
            DetectionMethod.GRADIENT: self._gradient_detection,
            DetectionMethod.LAPLACIAN: self._laplacian_detection,
            DetectionMethod.CANNY: self._canny_detection,
            DetectionMethod.THRESHOLD: self._threshold_detection,
            DetectionMethod.MORPHOLOGICAL: self._morphological_detection,
            DetectionMethod.STATISTICAL: self._statistical_detection
        }
    
    def detect_ensemble(self, image: np.ndarray, 
                       methods: Optional[List[DetectionMethod]] = None,
                       mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform ensemble detection using multiple methods.
        
        Args:
            image: Input grayscale image
            methods: List of methods to use. If None, uses all available methods.
            mask: Optional region mask
            
        Returns:
            Dictionary with ensemble results
        """
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale")
        
        if methods is None:
            methods = list(DetectionMethod)
        
        # Apply individual detection methods
        individual_results = []
        for method in methods:
            try:
                result = self._apply_single_method(image, method, mask)
                individual_results.append(result)
                self.logger.debug(f"Applied {method.name} detection")
            except Exception as e:
                self.logger.warning(f"Method {method.name} failed: {str(e)}")
                continue
        
        if not individual_results:
            self.logger.error("No detection methods succeeded")
            return self._create_empty_result(image.shape)
        
        # Perform ensemble fusion
        ensemble_result = self._fuse_detections(individual_results, image.shape)
        
        # Post-process ensemble result
        final_result = self._post_process_ensemble(ensemble_result, image)
        
        return {
            'ensemble_mask': final_result['ensemble_mask'],
            'confidence_map': final_result['confidence_map'],
            'vote_map': final_result['vote_map'],
            'individual_results': individual_results,
            'methods_used': [r.method.name for r in individual_results],
            'fusion_statistics': final_result['statistics']
        }
    
    def _apply_single_method(self, image: np.ndarray, 
                           method: DetectionMethod,
                           mask: Optional[np.ndarray] = None) -> DetectionResult:
        """Apply a single detection method."""
        if method not in self.detection_methods:
            raise ValueError(f"Unknown detection method: {method}")
        
        detection_func = self.detection_methods[method]
        detection_mask, confidence_map, params = detection_func(image)
        
        # Apply region mask if provided
        if mask is not None:
            detection_mask = cv2.bitwise_and(detection_mask, mask)
            if confidence_map is not None:
                confidence_map = confidence_map * (mask > 0)
        
        return DetectionResult(
            method=method,
            detection_mask=detection_mask,
            confidence_map=confidence_map,
            method_confidence=self.confidence_weights.get(method, 1.0),
            parameters=params
        )
    
    def _fuse_detections(self, results: List[DetectionResult], 
                        image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Fuse multiple detection results using intelligent voting."""
        h, w = image_shape
        vote_map = np.zeros((h, w), dtype=np.float32)
        confidence_map = np.zeros((h, w), dtype=np.float32)
        method_count = np.zeros((h, w), dtype=np.int32)
        
        # Accumulate votes and confidences
        for result in results:
            # Binary vote (weighted by method confidence)
            binary_mask = (result.detection_mask > 0).astype(np.float32)
            vote_weight = result.method_confidence
            vote_map += binary_mask * vote_weight
            
            # Confidence accumulation
            if result.confidence_map is not None:
                confidence_map += result.confidence_map * vote_weight
            else:
                confidence_map += binary_mask * vote_weight * 0.5  # Default confidence
            
            # Count contributing methods
            method_count += (binary_mask > 0).astype(np.int32)
        
        # Normalize by total possible votes
        total_weight = sum(r.method_confidence for r in results)
        if total_weight > 0:
            vote_map /= total_weight
            confidence_map /= total_weight
        
        return {
            'vote_map': vote_map,
            'confidence_map': confidence_map,
            'method_count': method_count,
            'total_methods': len(results)
        }
    
    def _post_process_ensemble(self, fusion_result: Dict[str, Any], 
                             image: np.ndarray) -> Dict[str, Any]:
        """Post-process ensemble results."""
        vote_map = fusion_result['vote_map']
        confidence_map = fusion_result['confidence_map']
        method_count = fusion_result['method_count']
        
        # Create ensemble mask based on voting threshold and method agreement
        ensemble_mask = np.zeros_like(vote_map, dtype=np.uint8)
        
        # Primary condition: vote threshold
        vote_condition = vote_map >= self.vote_threshold
        
        # Secondary condition: minimum method agreement
        agreement_condition = method_count >= self.min_methods_agreement
        
        # Combine conditions
        final_condition = vote_condition & agreement_condition
        ensemble_mask[final_condition] = 255
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ensemble_mask = cv2.morphologyEx(ensemble_mask, cv2.MORPH_OPEN, kernel)
        ensemble_mask = cv2.morphologyEx(ensemble_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate statistics
        statistics = {
            'mean_vote': np.mean(vote_map),
            'max_vote': np.max(vote_map),
            'mean_confidence': np.mean(confidence_map),
            'detection_pixels': np.sum(ensemble_mask > 0),
            'detection_ratio': np.sum(ensemble_mask > 0) / ensemble_mask.size,
            'avg_method_agreement': np.mean(method_count[ensemble_mask > 0]) if np.any(ensemble_mask > 0) else 0
        }
        
        return {
            'ensemble_mask': ensemble_mask,
            'confidence_map': confidence_map,
            'vote_map': vote_map,
            'statistics': statistics
        }
    
    def _create_empty_result(self, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Create empty result when no methods succeed."""
        h, w = image_shape
        return {
            'ensemble_mask': np.zeros((h, w), dtype=np.uint8),
            'confidence_map': np.zeros((h, w), dtype=np.float32),
            'vote_map': np.zeros((h, w), dtype=np.float32),
            'individual_results': [],
            'methods_used': [],
            'fusion_statistics': {'detection_pixels': 0, 'detection_ratio': 0.0}
        }
    
    # Detection method implementations
    def _do2mr_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Difference of Min-Max Ranking detection."""
        kernel_size = 5
        gamma = 3.0
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img_min = cv2.erode(image, kernel)
        img_max = cv2.dilate(image, kernel)
        residual = cv2.subtract(img_max, img_min)
        
        # Threshold based on statistics
        mean_res = np.mean(residual)
        std_res = np.std(residual)
        threshold = mean_res + gamma * std_res
        
        _, detection_mask = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)
        confidence_map = residual.astype(np.float32) / 255.0
        
        params = {'kernel_size': kernel_size, 'gamma': gamma, 'threshold': threshold}
        return detection_mask, confidence_map, params
    
    def _lei_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Linear Enhancement Inspector detection."""
        kernel_length = 15
        angle_step = 15
        gamma = 2.5
        
        max_response = np.zeros_like(image, dtype=np.float32)
        
        for angle_deg in range(0, 180, angle_step):
            # Create linear kernel
            kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
            center = kernel_length // 2
            cv2.line(kernel, (0, center), (kernel_length-1, center), 1.0, 1)
            
            # Normalize kernel
            if np.sum(kernel) > 0:
                kernel /= np.sum(kernel)
            
            # Rotate kernel
            M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
            rotated_kernel = cv2.warpAffine(kernel, M, (kernel_length, kernel_length))
            
            # Apply filter
            response = cv2.filter2D(image.astype(np.float32), -1, rotated_kernel)
            np.maximum(max_response, response, out=max_response)
        
        # Threshold
        mean_resp = np.mean(max_response)
        std_resp = np.std(max_response)
        threshold = mean_resp + gamma * std_resp
        
        _, detection_mask = cv2.threshold(max_response, threshold, 255, cv2.THRESH_BINARY)
        confidence_map = max_response / (np.max(max_response) + 1e-10)
        
        params = {'kernel_length': kernel_length, 'angle_step': angle_step, 'gamma': gamma}
        return detection_mask.astype(np.uint8), confidence_map, params
    
    def _gradient_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Gradient-based detection."""
        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold using Otsu
        _, detection_mask = cv2.threshold(gradient_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map = gradient_mag.astype(np.float32) / 255.0
        
        params = {'method': 'sobel', 'ksize': 3}
        return detection_mask, confidence_map, params
    
    def _laplacian_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Laplacian-based detection."""
        # Apply Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=5)
        laplacian_abs = np.abs(laplacian)
        
        # Normalize
        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold
        _, detection_mask = cv2.threshold(laplacian_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map = laplacian_norm.astype(np.float32) / 255.0
        
        params = {'ksize': 5}
        return detection_mask, confidence_map, params
    
    def _canny_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Canny edge detection."""
        # Auto-compute thresholds
        median = np.median(image)
        lower = int(max(0, 0.7 * median))
        upper = int(min(255, 1.3 * median))
        
        edges = cv2.Canny(image, lower, upper)
        
        # Dilate to make edges more visible
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        detection_mask = cv2.dilate(edges, kernel, iterations=1)
        
        confidence_map = detection_mask.astype(np.float32) / 255.0
        
        params = {'lower_threshold': lower, 'upper_threshold': upper}
        return detection_mask, confidence_map, params
    
    def _threshold_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Adaptive threshold detection."""
        # Multiple threshold methods
        methods = [
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            cv2.THRESH_TRIANGLE,
        ]
        
        best_mask = None
        best_score = 0
        best_params = {}
        
        for method in methods:
            try:
                _, mask = cv2.threshold(image, 0, 255, method)
                
                # Score based on number of features and connectivity
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                score = len(contours) * np.mean([cv2.contourArea(c) for c in contours] or [0])
                
                if score > best_score:
                    best_score = score
                    best_mask = mask
                    best_params = {'method': method, 'score': score}
            except:
                continue
        
        if best_mask is None:
            best_mask = np.zeros_like(image)
            best_params = {'method': 'none'}
        
        confidence_map = best_mask.astype(np.float32) / 255.0
        return best_mask, confidence_map, best_params
    
    def _morphological_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Morphological detection (top-hat)."""
        kernel_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # White top-hat (bright features)
        white_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Black top-hat (dark features)
        black_tophat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine
        combined = cv2.add(white_tophat, black_tophat)
        
        # Threshold
        _, detection_mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        confidence_map = combined.astype(np.float32) / 255.0
        
        params = {'kernel_size': kernel_size}
        return detection_mask, confidence_map, params
    
    def _statistical_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Statistical outlier detection."""
        # Local statistics using sliding window
        window_size = 15
        half_window = window_size // 2
        
        padded = cv2.copyMakeBorder(image, half_window, half_window, half_window, half_window, 
                                   cv2.BORDER_REFLECT)
        
        outlier_map = np.zeros_like(image, dtype=np.float32)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extract local window
                window = padded[i:i+window_size, j:j+window_size]
                
                # Calculate local statistics
                local_mean = np.mean(window)
                local_std = np.std(window)
                
                # Z-score
                center_pixel = image[i, j]
                if local_std > 0:
                    z_score = abs(center_pixel - local_mean) / local_std
                    outlier_map[i, j] = z_score
        
        # Threshold outliers
        threshold = 2.5
        detection_mask = (outlier_map > threshold).astype(np.uint8) * 255
        confidence_map = np.clip(outlier_map / 5.0, 0, 1)  # Normalize confidence
        
        params = {'window_size': window_size, 'threshold': threshold}
        return detection_mask, confidence_map, params


def test_ensemble_detection():
    """Test the ensemble detection methods."""
    print("Testing Ensemble Detection Methods Module")
    print("=" * 50)
    
    def create_test_image_with_defects(size=400):
        """Create a test image with various defects."""
        image = np.ones((size, size), dtype=np.uint8) * 128
        
        # Add background texture
        noise = np.random.normal(0, 10, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Add scratches (linear features)
        for i in range(3):
            angle = np.random.uniform(0, 180)
            length = np.random.randint(80, 150)
            width = np.random.randint(2, 5)
            
            center = (np.random.randint(100, size-100), np.random.randint(100, size-100))
            angle_rad = np.radians(angle)
            
            start_x = int(center[0] - length//2 * np.cos(angle_rad))
            start_y = int(center[1] - length//2 * np.sin(angle_rad))
            end_x = int(center[0] + length//2 * np.cos(angle_rad))
            end_y = int(center[1] + length//2 * np.sin(angle_rad))
            
            cv2.line(image, (start_x, start_y), (end_x, end_y), 80, width)
        
        # Add pits/digs (circular features)
        for i in range(5):
            center = (np.random.randint(50, size-50), np.random.randint(50, size-50))
            radius = np.random.randint(5, 20)
            cv2.circle(image, center, radius, 60, -1)
        
        # Add contamination (irregular features)
        for i in range(4):
            center = (np.random.randint(50, size-50), np.random.randint(50, size-50))
            # Create irregular blob
            blob_mask = np.zeros((size, size), dtype=np.uint8)
            pts = []
            for j in range(8):
                angle = j * 2 * np.pi / 8
                r = np.random.randint(10, 25)
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                pts.append([x, y])
            
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(blob_mask, [pts], 255)
            image[blob_mask > 0] = 90
        
        return image
    
    # Create test image
    test_image = create_test_image_with_defects(400)
    
    # Initialize ensemble detector
    detector = EnsembleDetector(
        vote_threshold=0.3,
        min_methods_agreement=2
    )
    
    # Test with different method combinations
    test_cases = [
        ("all_methods", None),
        ("robust_methods", [DetectionMethod.DO2MR, DetectionMethod.LEI, 
                           DetectionMethod.GRADIENT, DetectionMethod.STATISTICAL]),
        ("edge_methods", [DetectionMethod.GRADIENT, DetectionMethod.LAPLACIAN, 
                         DetectionMethod.CANNY]),
        ("morpho_methods", [DetectionMethod.MORPHOLOGICAL, DetectionMethod.THRESHOLD,
                           DetectionMethod.STATISTICAL])
    ]
    
    results = {}
    
    for case_name, methods in test_cases:
        print(f"\\nTesting {case_name}...")
        
        try:
            result = detector.detect_ensemble(test_image, methods)
            results[case_name] = result
            
            stats = result['fusion_statistics']
            print(f"  Methods used: {len(result['methods_used'])}")
            print(f"  Detection pixels: {stats['detection_pixels']}")
            print(f"  Detection ratio: {stats['detection_ratio']:.4f}")
            print(f"  Mean vote: {stats['mean_vote']:.3f}")
            print(f"  Mean confidence: {stats['mean_confidence']:.3f}")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            continue
    
    # Visualize results
    if results:
        visualize_ensemble_results(test_image, results, "ensemble_detection_test.png")
    
    print("\\nTest completed!")


def visualize_ensemble_results(image: np.ndarray, results: Dict[str, Any], 
                             save_path: Optional[str] = None):
    """Visualize ensemble detection results."""
    n_cases = len(results)
    
    fig, axes = plt.subplots(2, max(2, n_cases), figsize=(4*max(2, n_cases), 8))
    
    if n_cases == 1:
        axes = axes.reshape(2, 1)
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Results for each case
    for i, (case_name, result) in enumerate(results.items()):
        if i >= axes.shape[1]:
            break
            
        # Ensemble mask
        if i == 0:
            col = 1  # Skip original image
        else:
            col = i
            
        if col < axes.shape[1]:
            axes[0, col].imshow(result['ensemble_mask'], cmap='hot')
            axes[0, col].set_title(f'{case_name}\\nEnsemble Mask')
            axes[0, col].axis('off')
            
            # Vote map
            im = axes[1, col].imshow(result['vote_map'], cmap='viridis', vmin=0, vmax=1)
            axes[1, col].set_title(f'{case_name}\\nVote Map')
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    
    # Hide unused axes
    for i in range(max(1, len(results)), axes.shape[1]):
        axes[0, i].axis('off')
        if i > 0:  # Don't hide the first column of second row
            axes[1, i].axis('off')
    
    plt.suptitle('Ensemble Detection Results Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def demo_ensemble_detection():
    """Demonstrate ensemble detection capabilities."""
    print("Ensemble Detection Methods Demo")
    print("=" * 35)
    
    # Test the module
    test_ensemble_detection()
    
    print("\\nDemo completed!")
    print("Generated file: ensemble_detection_test.png")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    demo_ensemble_detection()
