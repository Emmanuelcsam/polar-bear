#!/usr/bin/env python3
"""
Advanced Illumination Correction Module
=======================================

This module provides comprehensive illumination correction techniques
for fiber optic image enhancement including homomorphic filtering,
multi-scale Retinex, and rolling ball background subtraction.

Author: Modular Analysis Team
Version: 1.0
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple, Any
import logging


class AdvancedIlluminationCorrector:
    """
    Comprehensive illumination correction using multiple advanced techniques.
    """
    
    def __init__(self):
        """Initialize the illumination corrector."""
        self.logger = logging.getLogger(__name__)
    
    def correct_illumination(self, image: np.ndarray, 
                           methods: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Apply multiple illumination correction methods.
        
        Args:
            image: Input grayscale image
            methods: List of methods to apply. If None, applies all methods.
                   Available: ['rolling_ball', 'homomorphic', 'retinex', 
                             'clahe', 'top_hat', 'div_normalize', 'gamma']
                   
        Returns:
            Dictionary with corrected images for each method
        """
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale")
        
        if methods is None:
            methods = ['rolling_ball', 'homomorphic', 'retinex', 'clahe', 
                      'top_hat', 'div_normalize', 'gamma']
        
        results = {'original': image.copy()}
        
        for method in methods:
            try:
                if method == 'rolling_ball':
                    results[method] = self.rolling_ball_correction(image)
                elif method == 'homomorphic':
                    results[method] = self.homomorphic_filtering(image)
                elif method == 'retinex':
                    results[method] = self.multi_scale_retinex(image)
                elif method == 'clahe':
                    results[method] = self.advanced_clahe(image)
                elif method == 'top_hat':
                    results[method] = self.top_hat_correction(image)
                elif method == 'div_normalize':
                    results[method] = self.division_normalization(image)
                elif method == 'gamma':
                    results[method] = self.adaptive_gamma_correction(image)
                else:
                    self.logger.warning(f"Unknown method: {method}")
                    
            except Exception as e:
                self.logger.error(f"Method {method} failed: {str(e)}")
                results[method] = image.copy()  # Fallback to original
        
        return results
    
    def rolling_ball_correction(self, image: np.ndarray, 
                               ball_radius: Optional[int] = None) -> np.ndarray:
        """
        Rolling ball background subtraction for illumination correction.
        
        Args:
            image: Input grayscale image
            ball_radius: Radius of the rolling ball. If None, auto-computed.
            
        Returns:
            Illumination-corrected image
        """
        if ball_radius is None:
            ball_radius = min(image.shape) // 8
        
        # Create structuring element (ball)
        kernel_size = ball_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Rolling ball is equivalent to morphological opening
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Subtract background
        corrected = cv2.subtract(image, background)
        
        # Add mean background to maintain brightness
        mean_background = np.mean(background)
        corrected = cv2.add(corrected, np.full_like(corrected, int(mean_background)))
        
        return corrected
    
    def homomorphic_filtering(self, image: np.ndarray, 
                             gamma_l: float = 0.5, gamma_h: float = 2.0,
                             c: float = 1.0, d0: float = 10.0) -> np.ndarray:
        """
        Homomorphic filtering for illumination and reflectance separation.
        
        Args:
            image: Input grayscale image
            gamma_l: Low frequency gain
            gamma_h: High frequency gain  
            c: Sharpness parameter
            d0: Cutoff frequency
            
        Returns:
            Homomorphically filtered image
        """
        # Convert to float and add small value to avoid log(0)
        img_float = image.astype(np.float32) + 1e-10
        
        # Take natural logarithm
        img_log = np.log(img_float)
        
        # Apply FFT
        img_fft = np.fft.fft2(img_log)
        img_fft_shift = np.fft.fftshift(img_fft)
        
        # Create high-pass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create coordinate matrices
        u = np.arange(rows).reshape(-1, 1) - crow
        v = np.arange(cols).reshape(1, -1) - ccol
        d = np.sqrt(u**2 + v**2)
        
        # Homomorphic filter
        h = (gamma_h - gamma_l) * (1 - np.exp(-c * (d**2 / d0**2))) + gamma_l
        
        # Apply filter
        img_filtered_fft = img_fft_shift * h
        
        # Inverse FFT
        img_filtered_fft_shift = np.fft.ifftshift(img_filtered_fft)
        img_filtered = np.real(np.fft.ifft2(img_filtered_fft_shift))
        
        # Take exponential
        img_result = np.exp(img_filtered)
        
        # Normalize to 0-255
        img_result = cv2.normalize(img_result, None, 0, 255, cv2.NORM_MINMAX)
        
        return img_result.astype(np.uint8)
    
    def multi_scale_retinex(self, image: np.ndarray, 
                           scales: Optional[List[float]] = None) -> np.ndarray:
        """
        Multi-scale Retinex enhancement.
        
        Args:
            image: Input grayscale image
            scales: List of Gaussian kernel scales. If None, uses default scales.
            
        Returns:
            Retinex-enhanced image
        """
        if scales is None:
            scales = [15, 80, 250]
        
        # Convert to float
        img_float = image.astype(np.float32) + 1e-10
        
        # Apply multi-scale Retinex
        retinex_result = np.zeros_like(img_float)
        
        for scale in scales:
            # Gaussian blur
            blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
            blurred = blurred + 1e-10  # Avoid division by zero
            
            # Compute single scale Retinex
            ssr = np.log(img_float) - np.log(blurred)
            retinex_result += ssr
        
        # Average across scales
        retinex_result /= len(scales)
        
        # Normalize
        retinex_result = cv2.normalize(retinex_result, None, 0, 255, cv2.NORM_MINMAX)
        
        return retinex_result.astype(np.uint8)
    
    def advanced_clahe(self, image: np.ndarray, 
                      clip_limit: float = 3.0,
                      tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Advanced Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input grayscale image
            clip_limit: Clipping limit for contrast limiting
            tile_size: Size of the tiles for local equalization
            
        Returns:
            CLAHE-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(image)
    
    def top_hat_correction(self, image: np.ndarray, 
                          kernel_sizes: Optional[List[int]] = None) -> np.ndarray:
        """
        Top-hat transformation for illumination correction.
        
        Args:
            image: Input grayscale image
            kernel_sizes: List of kernel sizes for multi-scale top-hat
            
        Returns:
            Top-hat corrected image
        """
        if kernel_sizes is None:
            kernel_sizes = [15, 25, 35]
        
        result = np.zeros_like(image, dtype=np.float32)
        
        for kernel_size in kernel_sizes:
            # Create elliptical kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (kernel_size, kernel_size))
            
            # White top-hat (highlights bright features)
            white_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            
            # Black top-hat (highlights dark features)  
            black_tophat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            
            # Combine
            combined = white_tophat.astype(np.float32) + black_tophat.astype(np.float32)
            result += combined
        
        # Average across scales
        result /= len(kernel_sizes)
        
        # Add to original image
        enhanced = image.astype(np.float32) + result
        
        # Normalize
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        return enhanced.astype(np.uint8)
    
    def division_normalization(self, image: np.ndarray, 
                             blur_size: int = 50) -> np.ndarray:
        """
        Division normalization for illumination correction.
        
        Args:
            image: Input grayscale image
            blur_size: Size of Gaussian blur for background estimation
            
        Returns:
            Division-normalized image
        """
        # Estimate background illumination
        background = cv2.GaussianBlur(image, (blur_size+1, blur_size+1), 
                                    blur_size/3)
        
        # Convert to float to avoid division issues
        img_float = image.astype(np.float32)
        bg_float = background.astype(np.float32) + 1e-10
        
        # Division normalization
        normalized = img_float / bg_float
        
        # Scale to maintain reasonable intensity range
        normalized *= np.mean(bg_float)
        
        # Normalize to 0-255
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized.astype(np.uint8)
    
    def adaptive_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive gamma correction based on image statistics.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Gamma-corrected image
        """
        # Calculate optimal gamma based on image mean
        mean_intensity = np.mean(image) / 255.0
        
        # Adaptive gamma calculation
        if mean_intensity < 0.5:
            gamma = 1.0 / (1.0 - mean_intensity)  # Brighten dark images
        else:
            gamma = 1.0 - mean_intensity  # Darken bright images
        
        # Clamp gamma to reasonable range
        gamma = np.clip(gamma, 0.3, 3.0)
        
        # Apply gamma correction
        normalized = image / 255.0
        corrected = np.power(normalized, gamma)
        result = (corrected * 255).astype(np.uint8)
        
        return result
    
    def estimate_illumination_quality(self, original: np.ndarray, 
                                    corrected: np.ndarray) -> Dict[str, float]:
        """
        Estimate the quality of illumination correction.
        
        Args:
            original: Original image
            corrected: Illumination-corrected image
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Contrast improvement
        orig_contrast = np.std(original)
        corr_contrast = np.std(corrected)
        metrics['contrast_improvement'] = corr_contrast / (orig_contrast + 1e-10)
        
        # Histogram uniformity (lower is better)
        orig_hist = cv2.calcHist([original], [0], None, [256], [0, 256])
        corr_hist = cv2.calcHist([corrected], [0], None, [256], [0, 256])
        
        orig_uniformity = np.std(orig_hist.flatten())
        corr_uniformity = np.std(corr_hist.flatten())
        metrics['uniformity_improvement'] = orig_uniformity / (corr_uniformity + 1e-10)
        
        # Dynamic range utilization
        orig_range = np.max(original) - np.min(original)
        corr_range = np.max(corrected) - np.min(corrected)
        metrics['dynamic_range_ratio'] = corr_range / (orig_range + 1e-10)
        
        # Entropy (information content)
        orig_entropy = self._calculate_entropy(original)
        corr_entropy = self._calculate_entropy(corrected)
        metrics['entropy_improvement'] = corr_entropy / (orig_entropy + 1e-10)
        
        return metrics
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of image."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten() + 1e-10  # Avoid log(0)
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy


def test_illumination_correction():
    """Test the illumination correction methods."""
    print("Testing Advanced Illumination Correction Module")
    print("=" * 50)
    
    def create_uneven_illumination_image(size=400):
        """Create an image with uneven illumination."""
        # Create base pattern
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        
        # Create some features
        features = np.zeros((size, size))
        
        # Add some circles (defects)
        for i in range(5):
            cx = np.random.randint(size//4, 3*size//4)
            cy = np.random.randint(size//4, 3*size//4)
            radius = np.random.randint(10, 30)
            cv2.circle(features, (cx, cy), radius, 100, -1)
        
        # Add some lines (scratches)
        for i in range(3):
            pt1 = (np.random.randint(size//4, 3*size//4), np.random.randint(size//4, 3*size//4))
            pt2 = (np.random.randint(size//4, 3*size//4), np.random.randint(size//4, 3*size//4))
            cv2.line(features, pt1, pt2, 80, 3)
        
        # Create uneven illumination
        illumination = 150 + 50 * np.exp(-(x**2 + y**2)/0.5)  # Bright center
        illumination += 30 * np.sin(3*x) * np.cos(2*y)  # Pattern
        
        # Combine
        image = (illumination + features).astype(np.uint8)
        
        # Add noise
        noise = np.random.normal(0, 5, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    # Create test image
    test_image = create_uneven_illumination_image(400)
    
    # Initialize corrector
    corrector = AdvancedIlluminationCorrector()
    
    # Apply all correction methods
    print("Applying illumination correction methods...")
    results = corrector.correct_illumination(test_image)
    
    # Evaluate quality improvements
    print("\\nQuality Assessment:")
    print("-" * 30)
    
    for method, corrected in results.items():
        if method == 'original':
            continue
            
        metrics = corrector.estimate_illumination_quality(test_image, corrected)
        print(f"{method:15s}: contrast={metrics['contrast_improvement']:.2f}, "
              f"uniformity={metrics['uniformity_improvement']:.2f}, "
              f"entropy={metrics['entropy_improvement']:.2f}")
    
    # Visualize results
    visualize_illumination_results(results, "illumination_correction_test.png")
    
    print("\\nTest completed!")


def visualize_illumination_results(results: Dict[str, np.ndarray], 
                                 save_path: Optional[str] = None):
    """Visualize illumination correction results."""
    methods = list(results.keys())
    n_methods = len(methods)
    
    # Arrange in grid
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (method, image) in enumerate(results.items()):
        row, col = divmod(i, n_cols)
        
        if n_rows == 1 and n_cols == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[col]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        ax.imshow(image, cmap='gray')
        ax.set_title(method.replace('_', ' ').title())
        ax.axis('off')
        
        # Add histogram as inset
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax_hist = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
        ax_hist.plot(hist.flatten(), 'k-', linewidth=1)
        ax_hist.set_xlim(0, 255)
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
        ax_hist.patch.set_alpha(0.7)
    
    # Hide unused subplots
    for i in range(n_methods, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        if n_rows == 1:
            axes[col].axis('off')
        elif n_cols == 1:
            axes[row].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.suptitle('Illumination Correction Methods Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def demo_illumination_correction():
    """Demonstrate illumination correction capabilities."""
    print("Advanced Illumination Correction Demo")
    print("=" * 40)
    
    # Test the module
    test_illumination_correction()
    
    print("\\nDemo completed!")
    print("Generated file: illumination_correction_test.png")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    demo_illumination_correction()
