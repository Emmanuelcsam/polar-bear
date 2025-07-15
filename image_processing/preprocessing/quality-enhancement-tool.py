#!/usr/bin/env python3
"""
Image Enhancement and Preprocessing Module
Advanced image preprocessing for fiber optic analysis.
Includes CLAHE, denoising, sharpening, and contrast enhancement.
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

warnings.filterwarnings('ignore')


class ImageEnhancer:
    """
    Advanced image enhancement for fiber optic images
    """
    
    def __init__(self,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8),
                 gaussian_kernel_size: int = 5,
                 gaussian_sigma: float = 1.0,
                 median_kernel_size: int = 5,
                 bilateral_d: int = 9,
                 bilateral_sigma_color: float = 75,
                 bilateral_sigma_space: float = 75,
                 unsharp_amount: float = 1.5,
                 unsharp_radius: float = 1.0,
                 histogram_equalization: bool = False):
        """
        Initialize the image enhancer
        
        Args:
            clahe_clip_limit: CLAHE contrast limiting threshold
            clahe_tile_size: CLAHE tile grid size
            gaussian_kernel_size: Size of Gaussian blur kernel
            gaussian_sigma: Sigma for Gaussian blur
            median_kernel_size: Size of median filter kernel
            bilateral_d: Diameter for bilateral filtering
            bilateral_sigma_color: Sigma color for bilateral filtering
            bilateral_sigma_space: Sigma space for bilateral filtering
            unsharp_amount: Amount of unsharp masking
            unsharp_radius: Radius for unsharp masking
            histogram_equalization: Whether to apply global histogram equalization
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.median_kernel_size = median_kernel_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.unsharp_amount = unsharp_amount
        self.unsharp_radius = unsharp_radius
        self.histogram_equalization = histogram_equalization
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization
        
        Args:
            image: Input grayscale image
            
        Returns:
            CLAHE enhanced image
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        return clahe.apply(image)
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction
        
        Args:
            image: Input image
            
        Returns:
            Blurred image
        """
        kernel_size = self.gaussian_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), self.gaussian_sigma)
    
    def apply_median_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply median filter for impulse noise removal
        
        Args:
            image: Input image
            
        Returns:
            Filtered image
        """
        kernel_size = self.median_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        return cv2.medianBlur(image, kernel_size)
    
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing
        
        Args:
            image: Input image
            
        Returns:
            Filtered image
        """
        return cv2.bilateralFilter(
            image,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        )
    
    def apply_unsharp_masking(self, image: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking for edge enhancement
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), self.unsharp_radius)
        
        # Create mask
        mask = cv2.subtract(image.astype(np.float32), blurred.astype(np.float32))
        
        # Apply mask
        sharpened = cv2.addWeighted(
            image.astype(np.float32), 1.0,
            mask, self.unsharp_amount,
            0
        )
        
        # Clip values
        sharpened = np.clip(sharpened, 0, 255)
        
        return sharpened.astype(np.uint8)
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply global histogram equalization
        
        Args:
            image: Input grayscale image
            
        Returns:
            Equalized image
        """
        return cv2.equalizeHist(image)
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Apply gamma correction for brightness adjustment
        
        Args:
            image: Input image
            gamma: Gamma value (> 1 darkens, < 1 brightens)
            
        Returns:
            Gamma corrected image
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        
        return cv2.LUT(image, table)
    
    def apply_morphological_operations(self, image: np.ndarray, 
                                     operation: str = 'opening',
                                     kernel_size: int = 3,
                                     kernel_shape: str = 'ellipse') -> np.ndarray:
        """
        Apply morphological operations
        
        Args:
            image: Input binary or grayscale image
            operation: Type of operation ('opening', 'closing', 'gradient', 'tophat', 'blackhat')
            kernel_size: Size of morphological kernel
            kernel_shape: Shape of kernel ('ellipse', 'rect', 'cross')
            
        Returns:
            Processed image
        """
        # Create kernel
        if kernel_shape == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == 'rect':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif kernel_shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply operation
        if operation == 'opening':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'gradient':
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        elif operation == 'tophat':
            return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        elif operation == 'blackhat':
            return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        else:
            return image
    
    def normalize_illumination(self, image: np.ndarray, kernel_size: int = 31) -> np.ndarray:
        """
        Normalize illumination variations using background subtraction
        
        Args:
            image: Input grayscale image
            kernel_size: Size of morphological kernel for background estimation
            
        Returns:
            Illumination normalized image
        """
        # Estimate background using morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Subtract background
        normalized = cv2.subtract(image, background)
        
        # Add offset to avoid negative values
        normalized = cv2.add(normalized, np.ones_like(normalized) * 50)
        
        return normalized
    
    def enhance_edges(self, image: np.ndarray, method: str = 'laplacian') -> np.ndarray:
        """
        Enhance edges in the image
        
        Args:
            image: Input grayscale image
            method: Edge enhancement method ('laplacian', 'sobel', 'scharr')
            
        Returns:
            Edge enhanced image
        """
        if method == 'laplacian':
            # Laplacian edge enhancement
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            laplacian = np.absolute(laplacian)
            enhanced = cv2.addWeighted(
                image.astype(np.float64), 0.8,
                laplacian, 0.2,
                0
            )
        elif method == 'sobel':
            # Sobel edge enhancement
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            enhanced = cv2.addWeighted(
                image.astype(np.float64), 0.7,
                sobel, 0.3,
                0
            )
        elif method == 'scharr':
            # Scharr edge enhancement
            scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
            scharr = np.sqrt(scharr_x**2 + scharr_y**2)
            enhanced = cv2.addWeighted(
                image.astype(np.float64), 0.7,
                scharr, 0.3,
                0
            )
        else:
            enhanced = image.astype(np.float64)
        
        # Normalize and convert back to uint8
        enhanced = np.clip(enhanced, 0, 255)
        return enhanced.astype(np.uint8)
    
    def enhance_image(self, image: np.ndarray, 
                     enhancement_steps: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Apply a sequence of enhancement operations
        
        Args:
            image: Input image (BGR or grayscale)
            enhancement_steps: List of enhancement steps to apply
            
        Returns:
            Dictionary containing intermediate and final results
        """
        if enhancement_steps is None:
            enhancement_steps = [
                'convert_grayscale',
                'clahe',
                'bilateral_filter',
                'unsharp_masking',
                'normalize_illumination'
            ]
        
        results = {'original': image.copy()}
        current_image = image.copy()
        
        for step in enhancement_steps:
            try:
                if step == 'convert_grayscale':
                    if len(current_image.shape) == 3:
                        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                
                elif step == 'clahe':
                    current_image = self.apply_clahe(current_image)
                
                elif step == 'histogram_equalization':
                    current_image = self.apply_histogram_equalization(current_image)
                
                elif step == 'gaussian_blur':
                    current_image = self.apply_gaussian_blur(current_image)
                
                elif step == 'median_filter':
                    current_image = self.apply_median_filter(current_image)
                
                elif step == 'bilateral_filter':
                    current_image = self.apply_bilateral_filter(current_image)
                
                elif step == 'unsharp_masking':
                    current_image = self.apply_unsharp_masking(current_image)
                
                elif step == 'gamma_correction':
                    current_image = self.apply_gamma_correction(current_image)
                
                elif step == 'normalize_illumination':
                    current_image = self.normalize_illumination(current_image)
                
                elif step == 'enhance_edges':
                    current_image = self.enhance_edges(current_image)
                
                elif step.startswith('morphology_'):
                    # Parse morphology operation
                    operation = step.split('_')[1]
                    current_image = self.apply_morphological_operations(current_image, operation)
                
                else:
                    print(f"Warning: Unknown enhancement step: {step}")
                    continue
                
                # Store intermediate result
                results[step] = current_image.copy()
                
            except Exception as e:
                print(f"Warning: Enhancement step '{step}' failed: {e}")
                continue
        
        results['final'] = current_image.copy()
        return results
    
    def auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically enhance image based on its characteristics
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Analyze image characteristics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        contrast_ratio = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        # Determine enhancement strategy based on characteristics
        enhancement_steps = ['convert_grayscale']
        
        # Low contrast images need contrast enhancement
        if contrast_ratio < 0.3:
            enhancement_steps.extend(['clahe', 'unsharp_masking'])
        else:
            enhancement_steps.append('clahe')
        
        # Dark images need brightness adjustment
        if mean_intensity < 100:
            enhancement_steps.append('gamma_correction')
        
        # Noisy images need filtering
        if std_intensity > 50:
            enhancement_steps.append('bilateral_filter')
        else:
            enhancement_steps.append('gaussian_blur')
        
        # Always normalize illumination for fiber optic images
        enhancement_steps.append('normalize_illumination')
        
        # Apply enhancements
        results = self.enhance_image(image, enhancement_steps)
        
        return results['final']


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Image Enhancement for Fiber Optic Analysis')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output-dir', default='enhanced_output',
                       help='Output directory for enhanced images')
    parser.add_argument('--enhancement-steps', nargs='+',
                       choices=['convert_grayscale', 'clahe', 'histogram_equalization',
                               'gaussian_blur', 'median_filter', 'bilateral_filter',
                               'unsharp_masking', 'gamma_correction', 'normalize_illumination',
                               'enhance_edges', 'morphology_opening', 'morphology_closing'],
                       help='Enhancement steps to apply')
    parser.add_argument('--auto-enhance', action='store_true',
                       help='Automatically determine best enhancement')
    parser.add_argument('--clahe-clip', type=float, default=2.0,
                       help='CLAHE clip limit')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate enhancement steps')
    
    args = parser.parse_args()
    
    # Load image
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not read image: {args.image_path}")
        return
    
    # Create enhancer
    enhancer = ImageEnhancer(clahe_clip_limit=args.clahe_clip)
    
    # Apply enhancement
    if args.auto_enhance:
        enhanced = enhancer.auto_enhance(image)
        results = {'auto_enhanced': enhanced}
    else:
        results = enhancer.enhance_image(image, args.enhancement_steps)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = Path(args.image_path).stem
    
    if args.save_intermediate:
        # Save all intermediate steps
        for step_name, step_image in results.items():
            output_path = os.path.join(args.output_dir, f"{base_filename}_{step_name}.png")
            cv2.imwrite(output_path, step_image)
            print(f"Saved: {output_path}")
    else:
        # Save only final result
        final_image = results.get('final', results.get('auto_enhanced', image))
        output_path = os.path.join(args.output_dir, f"{base_filename}_enhanced.png")
        cv2.imwrite(output_path, final_image)
        print(f"Enhanced image saved to: {output_path}")
    
    # Print image statistics
    if len(image.shape) == 3:
        orig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = image
    
    final_image = results.get('final', results.get('auto_enhanced', image))
    if len(final_image.shape) == 3:
        final_gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    else:
        final_gray = final_image
    
    print(f"\nImage Statistics:")
    print(f"Original - Mean: {np.mean(orig_gray):.1f}, Std: {np.std(orig_gray):.1f}")
    print(f"Enhanced - Mean: {np.mean(final_gray):.1f}, Std: {np.std(final_gray):.1f}")
    print(f"Contrast improvement: {np.std(final_gray)/np.std(orig_gray):.2f}x")


if __name__ == "__main__":
    main()
