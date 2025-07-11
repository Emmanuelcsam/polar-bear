#!/usr/bin/env python3
"""
Image Transformation Engine - Comprehensive OpenCV Image Processing
Extracted from process.py - Standalone modular script
"""

import cv2
import numpy as np
import os
import sys
import argparse
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageTransformationEngine:
    """Comprehensive image transformation system using OpenCV operations."""
    
    def __init__(self, save_intermediate=True):
        self.save_intermediate = save_intermediate
        self.logger = logger
        self.images_dict = {}
    
    def load_image(self, image_path):
        """Load image from file path."""
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Could not read image: {image_path}")
            raise ValueError(f"Could not read image: {image_path}")
        
        return img
    
    def save_image(self, tag, mat, output_folder):
        """Save image with given tag to output folder and keep in memory."""
        self.images_dict[tag] = mat.copy()
        
        if self.save_intermediate:
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"{tag}.jpg")
            cv2.imwrite(output_path, mat)
    
    def apply_thresholding_transforms(self, gray_img, output_folder):
        """Apply various thresholding techniques."""
        self.logger.info("Applying thresholding transforms...")
        
        # Basic thresholding
        ret, thresh_binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        self.save_image("threshold_binary", thresh_binary, output_folder)
        
        ret, thresh_binary_inv = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        self.save_image("threshold_binary_inv", thresh_binary_inv, output_folder)
        
        ret, thresh_trunc = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TRUNC)
        self.save_image("threshold_trunc", thresh_trunc, output_folder)
        
        ret, thresh_tozero = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TOZERO)
        self.save_image("threshold_tozero", thresh_tozero, output_folder)
        
        ret, thresh_tozero_inv = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TOZERO_INV)
        self.save_image("threshold_tozero_inv", thresh_tozero_inv, output_folder)
        
        # Adaptive thresholding
        adaptive_thresh_mean = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        self.save_image("adaptive_threshold_mean", adaptive_thresh_mean, output_folder)
        
        adaptive_thresh_gaussian = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.save_image("adaptive_threshold_gaussian", adaptive_thresh_gaussian, output_folder)
        
        # Otsu's thresholding
        ret, otsu_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.save_image("otsu_threshold", otsu_thresh, output_folder)
        
        return thresh_binary
    
    def apply_masking_transforms(self, img, output_folder):
        """Apply circular masking to image."""
        self.logger.info("Applying masking transforms...")
        
        mask = np.zeros(img.shape[:2], dtype="uint8")
        (cX, cY) = (img.shape[1] // 2, img.shape[0] // 2)
        radius = int(min(cX, cY) * 0.8)
        cv2.circle(mask, (cX, cY), radius, 255, -1)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        self.save_image("masked_circle", masked_img, output_folder)
        
        return masked_img
    
    def apply_color_transforms(self, img, gray_img, output_folder):
        """Apply various color space transformations and colormaps."""
        self.logger.info("Applying color transforms...")
        
        # Color space conversions
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.save_image("recolor_hsv", hsv_img, output_folder)
        
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        self.save_image("recolor_lab", lab_img, output_folder)
        
        # Apply colormaps to grayscale
        colormaps = [
            (cv2.COLORMAP_AUTUMN, "AUTUMN"),
            (cv2.COLORMAP_BONE, "BONE"),
            (cv2.COLORMAP_JET, "JET"),
            (cv2.COLORMAP_WINTER, "WINTER"),
            (cv2.COLORMAP_RAINBOW, "RAINBOW"),
            (cv2.COLORMAP_OCEAN, "OCEAN"),
            (cv2.COLORMAP_SUMMER, "SUMMER"),
            (cv2.COLORMAP_SPRING, "SPRING"),
            (cv2.COLORMAP_COOL, "COOL"),
            (cv2.COLORMAP_HSV, "HSV"),
            (cv2.COLORMAP_PINK, "PINK"),
            (cv2.COLORMAP_HOT, "HOT")
        ]
        
        for colormap, name in colormaps:
            colormap_img = cv2.applyColorMap(gray_img, colormap)
            self.save_image(f"recolor_colormap_{name.lower()}", colormap_img, output_folder)
    
    def apply_preprocessing_transforms(self, img, gray_img, thresh_binary, output_folder):
        """Apply comprehensive preprocessing transforms."""
        self.logger.info("Applying preprocessing transforms...")
        
        # Blurring techniques
        blurred = cv2.blur(img, (15, 15))
        self.save_image("preprocessing_blur", blurred, output_folder)
        
        gaussian_blurred = cv2.GaussianBlur(img, (15, 15), 0)
        self.save_image("preprocessing_gaussian_blur", gaussian_blurred, output_folder)
        
        median_blurred = cv2.medianBlur(img, 15)
        self.save_image("preprocessing_median_blur", median_blurred, output_folder)
        
        bilateral_filtered = cv2.bilateralFilter(img, 15, 75, 75)
        self.save_image("preprocessing_bilateral_filter", bilateral_filtered, output_folder)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(thresh_binary, kernel, iterations=1)
        self.save_image("preprocessing_erode", eroded, output_folder)
        
        dilated = cv2.dilate(thresh_binary, kernel, iterations=1)
        self.save_image("preprocessing_dilate", dilated, output_folder)
        
        opening = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, kernel)
        self.save_image("preprocessing_opening", opening, output_folder)
        
        closing = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)
        self.save_image("preprocessing_closing", closing, output_folder)
        
        gradient = cv2.morphologyEx(thresh_binary, cv2.MORPH_GRADIENT, kernel)
        self.save_image("preprocessing_gradient", gradient, output_folder)
        
        # Gradient and edge detection
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        self.save_image("preprocessing_laplacian", np.uint8(np.absolute(laplacian)), output_folder)
        
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
        self.save_image("preprocessing_sobel_x", np.uint8(np.absolute(sobel_x)), output_folder)
        self.save_image("preprocessing_sobel_y", np.uint8(np.absolute(sobel_y)), output_folder)
        
        # Edge detection
        canny_edges = cv2.Canny(gray_img, 100, 200)
        self.save_image("preprocessing_canny_edges", canny_edges, output_folder)
        
        # Denoising
        denoised_color = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        self.save_image("preprocessing_denoised_color", denoised_color, output_folder)
        
        # Histogram equalization
        equalized_hist = cv2.equalizeHist(gray_img)
        self.save_image("preprocessing_equalized_hist", equalized_hist, output_folder)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray_img)
        self.save_image("preprocessing_clahe", clahe_img, output_folder)
    
    def apply_geometric_transforms(self, img, output_folder):
        """Apply geometric transformations."""
        self.logger.info("Applying geometric transforms...")
        
        (h, w) = img.shape[:2]
        
        # Resizing with different interpolations
        resized_inter_nearest = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
        self.save_image("retexturing_resize_nearest", resized_inter_nearest, output_folder)
        
        resized_inter_cubic = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        self.save_image("retexturing_resize_cubic", resized_inter_cubic, output_folder)
    
    def apply_intensity_transforms(self, img, output_folder):
        """Apply pixel intensity manipulations."""
        self.logger.info("Applying intensity transforms...")
        
        # Brightness and contrast adjustments
        brighter = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
        self.save_image("intensity_brighter", brighter, output_folder)
        
        darker = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)
        self.save_image("intensity_darker", darker, output_folder)
        
        higher_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        self.save_image("intensity_higher_contrast", higher_contrast, output_folder)
        
        lower_contrast = cv2.convertScaleAbs(img, alpha=0.7, beta=0)
        self.save_image("intensity_lower_contrast", lower_contrast, output_folder)
    
    def apply_bitwise_operations(self, img, output_folder):
        """Apply bitwise operations with created mask."""
        self.logger.info("Applying bitwise operations...")
        
        (h, w) = img.shape[:2]
        
        # Create a simple mask
        img2 = np.zeros(img.shape, dtype="uint8")
        cv2.rectangle(img2, (w//4, h//4), (w*3//4, h*3//4), (255, 255, 255), -1)
        
        bitwise_and_op = cv2.bitwise_and(img, img2)
        self.save_image("binary_bitwise_and", bitwise_and_op, output_folder)
        
        bitwise_or_op = cv2.bitwise_or(img, img2)
        self.save_image("binary_bitwise_or", bitwise_or_op, output_folder)
        
        bitwise_xor_op = cv2.bitwise_xor(img, img2)
        self.save_image("binary_bitwise_xor", bitwise_xor_op, output_folder)
        
        bitwise_not_op = cv2.bitwise_not(img)
        self.save_image("binary_bitwise_not", bitwise_not_op, output_folder)
    
    def transform_image(self, image_path, output_folder="transformed_images"):
        """Apply all transformations to an image."""
        self.logger.info(f"Starting comprehensive image transformation for {image_path}")
        
        # Load image
        img = self.load_image(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply all transformation categories
        thresh_binary = self.apply_thresholding_transforms(gray_img, output_folder)
        self.apply_masking_transforms(img, output_folder)
        self.apply_color_transforms(img, gray_img, output_folder)
        self.apply_preprocessing_transforms(img, gray_img, thresh_binary, output_folder)
        self.apply_geometric_transforms(img, output_folder)
        self.apply_intensity_transforms(img, output_folder)
        self.apply_bitwise_operations(img, output_folder)
        
        if self.save_intermediate:
            count = len(os.listdir(output_folder))
            self.logger.info(f"Processing complete - saved {count} images to {output_folder}")
        else:
            self.logger.info(f"Processing complete in RAM mode with {len(self.images_dict)} frames")
        
        return self.images_dict
    
    def get_transformation_summary(self):
        """Get summary of all transformations applied."""
        return {
            'total_transformations': len(self.images_dict),
            'transformation_types': list(self.images_dict.keys()),
            'memory_usage_mb': sum(img.nbytes for img in self.images_dict.values()) / (1024 * 1024)
        }
    
    def save_summary_report(self, output_path):
        """Save a summary report of transformations."""
        summary = self.get_transformation_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary report saved to: {output_path}")


def main():
    """Command line interface for image transformation."""
    parser = argparse.ArgumentParser(description='Apply comprehensive image transformations')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('-o', '--output', default='transformed_images', help='Output directory')
    parser.add_argument('--ram-only', action='store_true', help='Keep transformations in RAM only')
    parser.add_argument('--summary', help='Save transformation summary to file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Initialize transformer
    transformer = ImageTransformationEngine(save_intermediate=not args.ram_only)
    
    try:
        # Apply transformations
        transformed_images = transformer.transform_image(args.image_path, args.output)
        
        # Print summary
        summary = transformer.get_transformation_summary()
        print(f"Successfully applied {summary['total_transformations']} transformations")
        print(f"Memory usage: {summary['memory_usage_mb']:.2f} MB")
        
        if args.summary:
            transformer.save_summary_report(args.summary)
        
    except Exception as e:
        print(f"Error during transformation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
