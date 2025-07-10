#!/usr/bin/env python3
"""
OpenCV Image Processing Transforms Module
Comprehensive image transformation utilities extracted from process.py.
Includes thresholding, filtering, morphological operations, and color transformations.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple


class ImageProcessor:
    """
    Comprehensive image processing class with multiple transformation methods.
    Can work in memory-only mode or save intermediate results.
    """
    
    def __init__(self, save_intermediate: bool = True, output_folder: str = "processed_images"):
        self.save_intermediate = save_intermediate
        self.output_folder = output_folder
        self.processed_images = {}
        
        if self.save_intermediate:
            os.makedirs(self.output_folder, exist_ok=True)
    
    def _save_image(self, name: str, image: np.ndarray):
        """Save image to disk and/or memory based on configuration."""
        # Always keep a RAM copy
        self.processed_images[name] = image.copy()
        
        # Conditionally save to disk
        if self.save_intermediate:
            filepath = os.path.join(self.output_folder, f"{name}.jpg")
            cv2.imwrite(filepath, image)
    
    def apply_thresholding_suite(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply comprehensive thresholding operations."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        results = {}
        
        # Basic thresholding
        _, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self._save_image("threshold_binary", thresh_binary)
        results["threshold_binary"] = thresh_binary
        
        _, thresh_binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        self._save_image("threshold_binary_inv", thresh_binary_inv)
        results["threshold_binary_inv"] = thresh_binary_inv
        
        _, thresh_trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
        self._save_image("threshold_trunc", thresh_trunc)
        results["threshold_trunc"] = thresh_trunc
        
        _, thresh_tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
        self._save_image("threshold_tozero", thresh_tozero)
        results["threshold_tozero"] = thresh_tozero
        
        _, thresh_tozero_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
        self._save_image("threshold_tozero_inv", thresh_tozero_inv)
        results["threshold_tozero_inv"] = thresh_tozero_inv
        
        # Adaptive thresholding
        adaptive_thresh_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        self._save_image("adaptive_threshold_mean", adaptive_thresh_mean)
        results["adaptive_threshold_mean"] = adaptive_thresh_mean
        
        adaptive_thresh_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self._save_image("adaptive_threshold_gaussian", adaptive_thresh_gaussian)
        results["adaptive_threshold_gaussian"] = adaptive_thresh_gaussian
        
        # Otsu's thresholding
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._save_image("otsu_threshold", otsu_thresh)
        results["otsu_threshold"] = otsu_thresh
        
        return results
    
    def apply_filtering_suite(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply comprehensive filtering operations."""
        results = {}
        
        # Basic blur
        blurred = cv2.blur(image, (15, 15))
        self._save_image("blur", blurred)
        results["blur"] = blurred
        
        # Gaussian blur
        gaussian_blurred = cv2.GaussianBlur(image, (15, 15), 0)
        self._save_image("gaussian_blur", gaussian_blurred)
        results["gaussian_blur"] = gaussian_blurred
        
        # Median blur
        median_blurred = cv2.medianBlur(image, 15)
        self._save_image("median_blur", median_blurred)
        results["median_blur"] = median_blurred
        
        # Bilateral filter
        bilateral_filtered = cv2.bilateralFilter(image, 15, 75, 75)
        self._save_image("bilateral_filter", bilateral_filtered)
        results["bilateral_filter"] = bilateral_filtered
        
        return results
    
    def apply_morphological_suite(self, binary_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply morphological operations to binary image."""
        results = {}
        kernel = np.ones((5, 5), np.uint8)
        
        # Erosion
        eroded = cv2.erode(binary_image, kernel, iterations=1)
        self._save_image("erode", eroded)
        results["erode"] = eroded
        
        # Dilation
        dilated = cv2.dilate(binary_image, kernel, iterations=1)
        self._save_image("dilate", dilated)
        results["dilate"] = dilated
        
        # Opening
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        self._save_image("opening", opening)
        results["opening"] = opening
        
        # Closing
        closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        self._save_image("closing", closing)
        results["closing"] = closing
        
        # Gradient
        gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
        self._save_image("gradient", gradient)
        results["gradient"] = gradient
        
        return results
    
    def apply_edge_detection_suite(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply edge detection operations."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        results = {}
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.uint8(np.absolute(laplacian))
        self._save_image("laplacian", laplacian_abs)
        results["laplacian"] = laplacian_abs
        
        # Sobel X
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_x_abs = np.uint8(np.absolute(sobel_x))
        self._save_image("sobel_x", sobel_x_abs)
        results["sobel_x"] = sobel_x_abs
        
        # Sobel Y
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_y_abs = np.uint8(np.absolute(sobel_y))
        self._save_image("sobel_y", sobel_y_abs)
        results["sobel_y"] = sobel_y_abs
        
        # Canny edge detection
        canny_edges = cv2.Canny(gray, 100, 200)
        self._save_image("canny_edges", canny_edges)
        results["canny_edges"] = canny_edges
        
        return results
    
    def apply_color_transforms(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply color space transformations and colormaps."""
        results = {}
        
        # Color space conversions
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self._save_image("color_hsv", hsv_img)
        results["color_hsv"] = hsv_img
        
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        self._save_image("color_lab", lab_img)
        results["color_lab"] = lab_img
        
        # Apply colormaps to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        colormaps = {
            'AUTUMN': cv2.COLORMAP_AUTUMN,
            'BONE': cv2.COLORMAP_BONE,
            'JET': cv2.COLORMAP_JET,
            'WINTER': cv2.COLORMAP_WINTER,
            'RAINBOW': cv2.COLORMAP_RAINBOW,
            'OCEAN': cv2.COLORMAP_OCEAN,
            'SUMMER': cv2.COLORMAP_SUMMER,
            'SPRING': cv2.COLORMAP_SPRING,
            'COOL': cv2.COLORMAP_COOL,
            'HSV': cv2.COLORMAP_HSV,
            'PINK': cv2.COLORMAP_PINK,
            'HOT': cv2.COLORMAP_HOT
        }
        
        for name, colormap in colormaps.items():
            colormap_img = cv2.applyColorMap(gray, colormap)
            self._save_image(f"colormap_{name.lower()}", colormap_img)
            results[f"colormap_{name.lower()}"] = colormap_img
        
        return results
    
    def apply_enhancement_suite(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply image enhancement operations."""
        results = {}
        
        # Brightness adjustments
        brighter = cv2.convertScaleAbs(image, alpha=1.0, beta=50)
        self._save_image("brighter", brighter)
        results["brighter"] = brighter
        
        darker = cv2.convertScaleAbs(image, alpha=1.0, beta=-50)
        self._save_image("darker", darker)
        results["darker"] = darker
        
        # Contrast adjustments
        higher_contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        self._save_image("higher_contrast", higher_contrast)
        results["higher_contrast"] = higher_contrast
        
        lower_contrast = cv2.convertScaleAbs(image, alpha=0.7, beta=0)
        self._save_image("lower_contrast", lower_contrast)
        results["lower_contrast"] = lower_contrast
        
        # Histogram equalization
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        equalized_hist = cv2.equalizeHist(gray)
        self._save_image("equalized_hist", equalized_hist)
        results["equalized_hist"] = equalized_hist
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        self._save_image("clahe", clahe_img)
        results["clahe"] = clahe_img
        
        # Denoising
        denoised_color = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        self._save_image("denoised_color", denoised_color)
        results["denoised_color"] = denoised_color
        
        return results
    
    def create_circular_mask(self, image: np.ndarray, center: Optional[Tuple[int, int]] = None, 
                           radius_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Create a circular mask and apply it to the image."""
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        
        radius = int(min(center[0], center[1], w - center[0], h - center[1]) * radius_ratio)
        
        mask = np.zeros((h, w), dtype="uint8")
        cv2.circle(mask, center, radius, 255, -1)
        
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        
        self._save_image("circular_mask", mask)
        self._save_image("masked_circle", masked_img)
        
        return mask, masked_img
    
    def process_image_comprehensive(self, image_path: str) -> Dict[str, np.ndarray]:
        """Apply all processing suites to an image."""
        print(f"Processing image: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Apply all processing suites
        all_results = {}
        
        # Thresholding
        thresh_results = self.apply_thresholding_suite(img)
        all_results.update(thresh_results)
        
        # Filtering
        filter_results = self.apply_filtering_suite(img)
        all_results.update(filter_results)
        
        # Morphological (on binary image)
        if "threshold_binary" in thresh_results:
            morph_results = self.apply_morphological_suite(thresh_results["threshold_binary"])
            all_results.update(morph_results)
        
        # Edge detection
        edge_results = self.apply_edge_detection_suite(img)
        all_results.update(edge_results)
        
        # Color transforms
        color_results = self.apply_color_transforms(img)
        all_results.update(color_results)
        
        # Enhancement
        enhance_results = self.apply_enhancement_suite(img)
        all_results.update(enhance_results)
        
        # Circular mask
        mask, masked = self.create_circular_mask(img)
        all_results["circular_mask"] = mask
        all_results["masked_circle"] = masked
        
        # Store all results
        self.processed_images.update(all_results)
        
        print(f"Generated {len(all_results)} processed variations")
        return all_results
    
    def get_processed_images(self) -> Dict[str, np.ndarray]:
        """Get all processed images from memory."""
        return self.processed_images.copy()


def main():
    """Test the ImageProcessor functionality."""
    print("Testing ImageProcessor...")
    
    # Ask for image path
    image_path = input("Enter path to test image (or press Enter to skip): ").strip()
    
    if not image_path:
        print("No image path provided. Creating synthetic test image...")
        # Create a synthetic test image
        test_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.circle(test_img, (150, 150), 100, (255, 255, 255), -1)
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 0, 255), 3)
        
        test_path = "test_synthetic.png"
        cv2.imwrite(test_path, test_img)
        image_path = test_path
        print(f"Created synthetic test image: {test_path}")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Test in memory-only mode
    processor = ImageProcessor(save_intermediate=False)
    results = processor.process_image_comprehensive(image_path)
    
    print(f"Processed {len(results)} image variations")
    print("Available variations:", list(results.keys()))
    
    # Test saving functionality
    processor_save = ImageProcessor(save_intermediate=True, output_folder="test_output")
    processor_save.process_image_comprehensive(image_path)
    print("Images saved to test_output folder")


if __name__ == "__main__":
    main()
