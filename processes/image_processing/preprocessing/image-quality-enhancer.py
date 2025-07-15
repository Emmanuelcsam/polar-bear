#!/usr/bin/env python3
"""
Image Enhancement and Preprocessing - Standalone Module
Extracted from fiber optic defect detection system
Provides various image enhancement and preprocessing techniques
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import hashlib


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image (np.ndarray): Input grayscale image
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (tuple): Size of the neighborhood for local equalization
        
    Returns:
        np.ndarray: Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def gaussian_blur_enhancement(image, kernel_size=(5, 5), sigma_x=0):
    """
    Apply Gaussian blur for noise reduction.
    
    Args:
        image (np.ndarray): Input image
        kernel_size (tuple): Size of the Gaussian kernel
        sigma_x (float): Standard deviation in X direction
        
    Returns:
        np.ndarray: Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, sigma_x)


def bilateral_filter_enhancement(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter for edge-preserving smoothing.
    
    Args:
        image (np.ndarray): Input image
        d (int): Diameter of each pixel neighborhood
        sigma_color (float): Filter sigma in the color space
        sigma_space (float): Filter sigma in the coordinate space
        
    Returns:
        np.ndarray: Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def median_blur_enhancement(image, kernel_size=5):
    """
    Apply median blur for salt-and-pepper noise removal.
    
    Args:
        image (np.ndarray): Input image
        kernel_size (int): Size of the median filter kernel
        
    Returns:
        np.ndarray: Filtered image
    """
    return cv2.medianBlur(image, kernel_size)


def morphological_operations(image, operation='open', kernel_size=(5, 5), kernel_shape='ellipse'):
    """
    Apply morphological operations.
    
    Args:
        image (np.ndarray): Input binary or grayscale image
        operation (str): Type of operation ('open', 'close', 'gradient', 'tophat', 'blackhat')
        kernel_size (tuple): Size of the structuring element
        kernel_shape (str): Shape of kernel ('ellipse', 'rect', 'cross')
        
    Returns:
        np.ndarray: Processed image
    """
    # Create structuring element
    if kernel_shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    elif kernel_shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    # Apply operation
    operation_map = {
        'open': cv2.MORPH_OPEN,
        'close': cv2.MORPH_CLOSE,
        'gradient': cv2.MORPH_GRADIENT,
        'tophat': cv2.MORPH_TOPHAT,
        'blackhat': cv2.MORPH_BLACKHAT,
        'erode': cv2.MORPH_ERODE,
        'dilate': cv2.MORPH_DILATE
    }
    
    if operation in operation_map:
        return cv2.morphologyEx(image, operation_map[operation], kernel)
    else:
        raise ValueError(f"Unknown morphological operation: {operation}")


def adaptive_threshold_enhancement(image, max_value=255, adaptive_method='gaussian', 
                                 threshold_type='binary', block_size=11, c=2):
    """
    Apply adaptive thresholding.
    
    Args:
        image (np.ndarray): Input grayscale image
        max_value (int): Maximum value assigned to pixels
        adaptive_method (str): 'mean' or 'gaussian'
        threshold_type (str): 'binary' or 'binary_inv'
        block_size (int): Size of neighborhood for threshold calculation
        c (float): Constant subtracted from the mean
        
    Returns:
        np.ndarray: Binary image
    """
    # Map string parameters to OpenCV constants
    adaptive_method_map = {
        'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
        'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    }
    
    threshold_type_map = {
        'binary': cv2.THRESH_BINARY,
        'binary_inv': cv2.THRESH_BINARY_INV
    }
    
    return cv2.adaptiveThreshold(
        image, max_value, 
        adaptive_method_map[adaptive_method],
        threshold_type_map[threshold_type],
        block_size, c
    )


def otsu_threshold_enhancement(image, threshold_type='binary'):
    """
    Apply Otsu's automatic threshold selection.
    
    Args:
        image (np.ndarray): Input grayscale image
        threshold_type (str): Type of thresholding
        
    Returns:
        tuple: (threshold_value, binary_image)
    """
    threshold_type_map = {
        'binary': cv2.THRESH_BINARY,
        'binary_inv': cv2.THRESH_BINARY_INV
    }
    
    threshold_val, binary = cv2.threshold(
        image, 0, 255, 
        threshold_type_map[threshold_type] + cv2.THRESH_OTSU
    )
    
    return threshold_val, binary


def canny_edge_detection(image, low_threshold=50, high_threshold=150, aperture_size=3):
    """
    Apply Canny edge detection.
    
    Args:
        image (np.ndarray): Input grayscale image
        low_threshold (int): Lower threshold for edge linking
        high_threshold (int): Upper threshold for edge detection
        aperture_size (int): Aperture size for Sobel operator
        
    Returns:
        np.ndarray: Edge image
    """
    return cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size)


def sobel_gradient_enhancement(image, dx=1, dy=1, kernel_size=3):
    """
    Apply Sobel gradient operator.
    
    Args:
        image (np.ndarray): Input grayscale image
        dx (int): Order of derivative in x direction
        dy (int): Order of derivative in y direction
        kernel_size (int): Size of the Sobel kernel
        
    Returns:
        dict: Dictionary with gradient components and magnitude
    """
    # Calculate gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, dx, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, dy, ksize=kernel_size)
    
    # Calculate magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    # Convert to uint8
    grad_x_uint8 = cv2.convertScaleAbs(grad_x)
    grad_y_uint8 = cv2.convertScaleAbs(grad_y)
    magnitude_uint8 = cv2.convertScaleAbs(magnitude)
    
    return {
        'grad_x': grad_x_uint8,
        'grad_y': grad_y_uint8,
        'magnitude': magnitude_uint8,
        'direction': direction
    }


def laplacian_enhancement(image, kernel_size=3):
    """
    Apply Laplacian operator for edge enhancement.
    
    Args:
        image (np.ndarray): Input grayscale image
        kernel_size (int): Size of the Laplacian kernel
        
    Returns:
        np.ndarray: Laplacian-enhanced image
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    return cv2.convertScaleAbs(laplacian)


def histogram_equalization(image):
    """
    Apply global histogram equalization.
    
    Args:
        image (np.ndarray): Input grayscale image
        
    Returns:
        np.ndarray: Equalized image
    """
    return cv2.equalizeHist(image)


def unsharp_masking(image, gaussian_kernel_size=(5, 5), sigma=1.0, alpha=1.5, threshold=0):
    """
    Apply unsharp masking for image sharpening.
    
    Args:
        image (np.ndarray): Input image
        gaussian_kernel_size (tuple): Size of Gaussian kernel
        sigma (float): Standard deviation for Gaussian blur
        alpha (float): Strength of sharpening
        threshold (int): Threshold for edge detection
        
    Returns:
        np.ndarray: Sharpened image
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(image, gaussian_kernel_size, sigma)
    
    # Calculate unsharp mask
    mask = cv2.subtract(image, blurred)
    
    # Apply threshold if specified
    if threshold > 0:
        mask = np.where(np.abs(mask) < threshold, 0, mask)
    
    # Apply sharpening
    sharpened = cv2.addWeighted(image, 1.0, mask, alpha, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def noise_reduction_nlm(image, h=10, template_window_size=7, search_window_size=21):
    """
    Apply Non-local Means denoising.
    
    Args:
        image (np.ndarray): Input grayscale image
        h (float): Filter strength
        template_window_size (int): Size of template patch
        search_window_size (int): Size of search window
        
    Returns:
        np.ndarray: Denoised image
    """
    return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)


def generate_image_variations(image, variation_types=None):
    """
    Generate multiple variations of an image using different enhancement techniques.
    
    Args:
        image (np.ndarray): Input grayscale image
        variation_types (list, optional): List of variation types to generate
        
    Returns:
        dict: Dictionary with variation names as keys and processed images as values
    """
    if variation_types is None:
        variation_types = [
            'original', 'clahe', 'gaussian_blur', 'bilateral_filter', 'median_blur',
            'morph_open', 'morph_close', 'morph_gradient', 'adaptive_threshold',
            'otsu_threshold', 'canny_edges', 'sobel_magnitude', 'laplacian',
            'histogram_eq', 'unsharp_mask', 'nlm_denoising'
        ]
    
    variations = {}
    
    for var_type in variation_types:
        try:
            if var_type == 'original':
                variations[var_type] = image.copy()
            elif var_type == 'clahe':
                variations[var_type] = clahe_enhancement(image)
            elif var_type == 'gaussian_blur':
                variations[var_type] = gaussian_blur_enhancement(image)
            elif var_type == 'bilateral_filter':
                variations[var_type] = bilateral_filter_enhancement(image)
            elif var_type == 'median_blur':
                variations[var_type] = median_blur_enhancement(image)
            elif var_type == 'morph_open':
                variations[var_type] = morphological_operations(image, 'open')
            elif var_type == 'morph_close':
                variations[var_type] = morphological_operations(image, 'close')
            elif var_type == 'morph_gradient':
                variations[var_type] = morphological_operations(image, 'gradient')
            elif var_type == 'adaptive_threshold':
                variations[var_type] = adaptive_threshold_enhancement(image)
            elif var_type == 'otsu_threshold':
                _, binary = otsu_threshold_enhancement(image)
                variations[var_type] = binary
            elif var_type == 'canny_edges':
                variations[var_type] = canny_edge_detection(image)
            elif var_type == 'sobel_magnitude':
                sobel_result = sobel_gradient_enhancement(image)
                variations[var_type] = sobel_result['magnitude']
            elif var_type == 'laplacian':
                variations[var_type] = laplacian_enhancement(image)
            elif var_type == 'histogram_eq':
                variations[var_type] = histogram_equalization(image)
            elif var_type == 'unsharp_mask':
                variations[var_type] = unsharp_masking(image)
            elif var_type == 'nlm_denoising':
                variations[var_type] = noise_reduction_nlm(image)
        except Exception as e:
            print(f"Warning: Failed to generate {var_type} variation: {e}")
    
    return variations


def comprehensive_image_enhancement(image_path, output_dir='enhanced_images', 
                                  variation_types=None, save_individual=True):
    """
    Apply comprehensive image enhancement with multiple techniques.
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory to save results
        variation_types (list, optional): List of enhancement types to apply
        save_individual (bool): Whether to save individual variations
        
    Returns:
        dict: Results with enhanced images and metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result
    result = {
        'method': 'comprehensive_image_enhancement',
        'image_path': image_path,
        'success': False,
        'variations_generated': [],
        'enhancement_metadata': {}
    }
    
    try:
        # Load image
        if not os.path.exists(image_path):
            result['error'] = f"Image not found: {image_path}"
            return result
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            result['error'] = f"Could not read image: {image_path}"
            return result
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Generate variations
        print("Generating image variations...")
        variations = generate_image_variations(image, variation_types)
        
        result['variations_generated'] = list(variations.keys())
        result['total_variations'] = len(variations)
        
        # Calculate quality metrics for each variation
        for var_name, var_image in variations.items():
            # Calculate image quality metrics
            mean_intensity = np.mean(var_image)
            std_intensity = np.std(var_image)
            contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(var_image, cv2.CV_64F).var()
            
            result['enhancement_metadata'][var_name] = {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'contrast': float(contrast),
                'sharpness': float(laplacian_var),
                'shape': var_image.shape
            }
            
            # Save individual variations if requested
            if save_individual:
                var_path = os.path.join(output_dir, f'{base_filename}_{var_name}.jpg')
                cv2.imwrite(var_path, var_image)
        
        # Create comparison montage
        if len(variations) > 0:
            montage = create_variation_montage(variations)
            montage_path = os.path.join(output_dir, f'{base_filename}_variations_montage.jpg')
            cv2.imwrite(montage_path, montage)
        
        result['success'] = True
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f'{base_filename}_enhancement_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(result, f, indent=4, cls=NumpyEncoder)
        
        print(f"Enhancement complete. Generated {len(variations)} variations.")
        
    except Exception as e:
        result['error'] = f"Enhancement failed: {str(e)}"
        print(f"Error: {e}")
    
    return result


def create_variation_montage(variations, max_cols=4):
    """
    Create a montage showing all image variations.
    
    Args:
        variations (dict): Dictionary of variation images
        max_cols (int): Maximum number of columns in montage
        
    Returns:
        np.ndarray: Montage image
    """
    if not variations:
        return np.zeros((100, 100), dtype=np.uint8)
    
    # Get image dimensions (assume all are same size)
    sample_image = next(iter(variations.values()))
    img_h, img_w = sample_image.shape[:2]
    
    # Calculate montage dimensions
    num_images = len(variations)
    cols = min(num_images, max_cols)
    rows = (num_images + cols - 1) // cols
    
    # Create montage
    montage_h = rows * img_h + (rows - 1) * 10  # 10px spacing
    montage_w = cols * img_w + (cols - 1) * 10
    montage = np.zeros((montage_h, montage_w), dtype=np.uint8)
    
    # Place images
    for idx, (var_name, var_image) in enumerate(variations.items()):
        row = idx // cols
        col = idx % cols
        
        y_start = row * (img_h + 10)
        x_start = col * (img_w + 10)
        
        montage[y_start:y_start+img_h, x_start:x_start+img_w] = var_image
        
        # Add label
        cv2.putText(montage, var_name, (x_start, y_start + img_h + 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,), 1)
    
    return montage


def main():
    """Command line interface for image enhancement"""
    parser = argparse.ArgumentParser(description='Comprehensive Image Enhancement and Preprocessing')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output-dir', default='enhanced_images',
                       help='Output directory (default: enhanced_images)')
    parser.add_argument('--variations', nargs='*', 
                       help='Specific variations to generate (default: all)')
    parser.add_argument('--no-individual', action='store_true',
                       help='Do not save individual variation images')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run enhancement
    result = comprehensive_image_enhancement(
        image_path=args.image_path,
        output_dir=args.output_dir,
        variation_types=args.variations,
        save_individual=not args.no_individual
    )
    
    # Print results
    if args.verbose:
        print(json.dumps(result, indent=2, cls=NumpyEncoder))
    else:
        if result['success']:
            print(f"✓ Image enhancement successful!")
            print(f"  Variations generated: {result['total_variations']}")
            print(f"  Available variations: {', '.join(result['variations_generated'])}")
            
            # Show quality metrics for top variations
            metadata = result['enhancement_metadata']
            if metadata:
                print("  Top variations by sharpness:")
                sorted_vars = sorted(metadata.items(), 
                                   key=lambda x: x[1]['sharpness'], reverse=True)
                for i, (var_name, metrics) in enumerate(sorted_vars[:3]):
                    print(f"    {i+1}. {var_name}: sharpness={metrics['sharpness']:.2f}, "
                          f"contrast={metrics['contrast']:.3f}")
        else:
            print(f"✗ Image enhancement failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
