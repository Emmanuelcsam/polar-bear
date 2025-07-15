#!/usr/bin/env python3
"""
Image Preprocessing Module
==========================
Standalone module for advanced image preprocessing operations.
Extracted from the Advanced Fiber Optic End Face Defect Detection System.

Author: Modularized by AI
Date: July 9, 2025
Version: 1.0
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import argparse
import sys
from pathlib import Path


class ImagePreprocessor:
    """
    A class for applying various preprocessing techniques to images.
    """
    
    def __init__(self, 
                 gaussian_kernel_size: Tuple[int, int] = (7, 7),
                 gaussian_sigma: int = 2,
                 bilateral_d: int = 9,
                 bilateral_sigma_color: int = 75,
                 bilateral_sigma_space: int = 75,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            gaussian_kernel_size: Kernel size for Gaussian blur
            gaussian_sigma: Sigma for Gaussian blur
            bilateral_d: Diameter for bilateral filter
            bilateral_sigma_color: Sigma color for bilateral filter
            bilateral_sigma_space: Sigma space for bilateral filter
            clahe_clip_limit: Clip limit for CLAHE
            clahe_tile_grid_size: Tile grid size for CLAHE
        """
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        
    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Applies various preprocessing techniques to the input image.
        
        Args:
            image: The input BGR or grayscale image
            
        Returns:
            A dictionary of preprocessed images including:
            - 'original_gray': Original grayscale version
            - 'gaussian_blurred': Gaussian blur applied
            - 'bilateral_filtered': Bilateral filter applied
            - 'clahe_enhanced': CLAHE enhanced
            - 'hist_equalized': Histogram equalized
        """
        if image is None:
            print("ERROR: Input image is None")
            return {}

        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image.copy()
        else:
            print(f"ERROR: Unsupported image format: shape {image.shape}")
            return {}

        processed_images = {}
        processed_images['original_gray'] = gray.copy()

        # 1. Gaussian Blur
        try:
            blurred = cv2.GaussianBlur(gray, self.gaussian_kernel_size, self.gaussian_sigma)
            processed_images['gaussian_blurred'] = blurred
            print("INFO: Gaussian blur applied successfully")
        except Exception as e:
            print(f"WARNING: Gaussian blur failed: {e}")
            processed_images['gaussian_blurred'] = gray.copy()

        # 2. Bilateral Filter (Edge-preserving smoothing)
        try:
            bilateral = cv2.bilateralFilter(gray, self.bilateral_d,
                                          self.bilateral_sigma_color,
                                          self.bilateral_sigma_space)
            processed_images['bilateral_filtered'] = bilateral
            print("INFO: Bilateral filter applied successfully")
        except Exception as e:
            print(f"WARNING: Bilateral filter failed: {e}")
            processed_images['bilateral_filtered'] = gray.copy()

        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        try:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                   tileGridSize=self.clahe_tile_grid_size)
            clahe_enhanced = clahe.apply(processed_images.get('bilateral_filtered', gray))
            processed_images['clahe_enhanced'] = clahe_enhanced
            print("INFO: CLAHE enhancement applied successfully")
        except Exception as e:
            print(f"WARNING: CLAHE failed: {e}")
            processed_images['clahe_enhanced'] = gray.copy()

        # 4. Standard Histogram Equalization
        try:
            hist_equalized = cv2.equalizeHist(gray)
            processed_images['hist_equalized'] = hist_equalized
            print("INFO: Histogram equalization applied successfully")
        except Exception as e:
            print(f"WARNING: Histogram equalization failed: {e}")
            processed_images['hist_equalized'] = gray.copy()

        return processed_images


def main():
    """
    Main function for standalone execution.
    """
    parser = argparse.ArgumentParser(description="Image Preprocessing Module")
    parser.add_argument("input_path", help="Path to input image")
    parser.add_argument("--output_dir", default="preprocessed_output", 
                       help="Output directory for processed images")
    parser.add_argument("--gaussian_kernel", type=int, default=7,
                       help="Gaussian kernel size (odd number)")
    parser.add_argument("--gaussian_sigma", type=int, default=2,
                       help="Gaussian sigma value")
    parser.add_argument("--bilateral_d", type=int, default=9,
                       help="Bilateral filter diameter")
    parser.add_argument("--clahe_clip", type=float, default=2.0,
                       help="CLAHE clip limit")
    
    args = parser.parse_args()
    
    # Load input image
    image_path = Path(args.input_path)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)
        
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        sys.exit(1)
        
    print(f"INFO: Loaded image {image_path} with shape {image.shape}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        gaussian_kernel_size=(args.gaussian_kernel, args.gaussian_kernel),
        gaussian_sigma=args.gaussian_sigma,
        bilateral_d=args.bilateral_d,
        clahe_clip_limit=args.clahe_clip
    )
    
    # Apply preprocessing
    print("INFO: Starting image preprocessing...")
    processed_images = preprocessor.preprocess_image(image)
    
    if not processed_images:
        print("ERROR: Preprocessing failed")
        sys.exit(1)
    
    # Save processed images
    base_name = image_path.stem
    for process_name, processed_img in processed_images.items():
        output_path = output_dir / f"{base_name}_{process_name}.jpg"
        cv2.imwrite(str(output_path), processed_img)
        print(f"INFO: Saved {process_name} to {output_path}")
    
    print(f"INFO: Preprocessing complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
