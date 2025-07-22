#!/usr/bin/env python3
"""
Adaptive Thresholding Module for Fiber Optic Defect Detection
=============================================================

This module implements advanced local thresholding methods for detecting defects
in fiber optic end-face images. Extracted from advanced_defect_analysis.py.

Functions:
- Niblack's adaptive thresholding
- Sauvola's adaptive thresholding
- Local contrast-based thresholding
- Multi-scale adaptive thresholding

Author: Extracted from Advanced Fiber Analysis Team
"""

import numpy as np
import cv2
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')


def niblack_threshold(image, window_size=15, k=-0.2, mask=None):
    """
    Niblack's adaptive thresholding method.
    Threshold = mean + k * standard_deviation
    
    Args:
        image (np.ndarray): Input grayscale image
        window_size (int): Size of local window
        k (float): Niblack parameter (typically negative)
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Contains threshold map, binary result, and parameters
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Convert to float for calculations
    img_float = image.astype(np.float64)
    
    # Calculate local mean
    mean = cv2.blur(img_float, (window_size, window_size))
    
    # Calculate local standard deviation
    mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    std = np.sqrt(np.maximum(0, mean_sq - mean**2))
    
    # Niblack threshold
    threshold_map = mean + k * std
    
    # Apply threshold
    binary_result = (img_float < threshold_map) & mask
    
    return {
        'threshold_map': threshold_map,
        'binary_result': binary_result,
        'defects': binary_result,
        'local_mean': mean,
        'local_std': std,
        'k': k,
        'window_size': window_size
    }


def sauvola_threshold(image, window_size=15, k=0.5, R=128, mask=None):
    """
    Sauvola's adaptive thresholding method.
    Improved version of Niblack's method with dynamic range normalization.
    
    Args:
        image (np.ndarray): Input grayscale image
        window_size (int): Size of local window
        k (float): Sauvola parameter
        R (float): Dynamic range of standard deviation
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Contains threshold map, binary result, and parameters
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Convert to float
    img_float = image.astype(np.float64)
    
    # Calculate local mean
    mean = cv2.blur(img_float, (window_size, window_size))
    
    # Calculate local standard deviation
    mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    std = np.sqrt(np.maximum(0, mean_sq - mean**2))
    
    # Sauvola threshold
    threshold_map = mean * (1 + k * ((std / R) - 1))
    
    # Apply threshold
    binary_result = (img_float < threshold_map) & mask
    
    return {
        'threshold_map': threshold_map,
        'binary_result': binary_result,
        'defects': binary_result,
        'local_mean': mean,
        'local_std': std,
        'k': k,
        'R': R,
        'window_size': window_size
    }


def local_contrast_threshold(image, window_size=15, contrast_threshold=0.3, mask=None):
    """
    Local contrast-based thresholding for detecting regions with high local variation.
    
    Args:
        image (np.ndarray): Input grayscale image
        window_size (int): Size of local window
        contrast_threshold (float): Threshold for contrast detection
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Contains contrast map, binary result, and parameters
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Convert to float
    img_float = image.astype(np.float64)
    
    # Create structuring element
    kernel = np.ones((window_size, window_size), dtype=np.uint8)
    
    # Calculate local min and max
    local_min = cv2.erode(img_float, kernel)
    local_max = cv2.dilate(img_float, kernel)
    
    # Calculate local contrast
    denominator = local_max + local_min + 1e-7  # Avoid division by zero
    local_contrast = (local_max - local_min) / denominator
    
    # Apply threshold
    contrast_defects = (local_contrast > contrast_threshold) & mask
    
    return {
        'contrast_map': local_contrast,
        'binary_result': contrast_defects,
        'defects': contrast_defects,
        'local_min': local_min,
        'local_max': local_max,
        'contrast_threshold': contrast_threshold,
        'window_size': window_size
    }


def phansalkar_threshold(image, window_size=15, k=0.25, p=2.0, q=10.0, mask=None):
    """
    Phansalkar's adaptive thresholding method.
    Extension of Sauvola's method for low-contrast images.
    
    Args:
        image (np.ndarray): Input grayscale image
        window_size (int): Size of local window
        k (float): Phansalkar parameter
        p (float): Power parameter
        q (float): Normalization parameter
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Contains threshold map, binary result, and parameters
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Convert to float
    img_float = image.astype(np.float64)
    
    # Calculate local mean
    mean = cv2.blur(img_float, (window_size, window_size))
    
    # Calculate local standard deviation
    mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    std = np.sqrt(np.maximum(0, mean_sq - mean**2))
    
    # Calculate global mean for normalization
    global_mean = np.mean(img_float[mask]) if np.any(mask) else np.mean(img_float)
    
    # Phansalkar threshold
    threshold_map = mean * (1 + p * np.exp(-q * mean) + k * ((std / global_mean) - 1))
    
    # Apply threshold
    binary_result = (img_float < threshold_map) & mask
    
    return {
        'threshold_map': threshold_map,
        'binary_result': binary_result,
        'defects': binary_result,
        'local_mean': mean,
        'local_std': std,
        'global_mean': global_mean,
        'k': k,
        'p': p,
        'q': q,
        'window_size': window_size
    }


def multiscale_adaptive_threshold(image, window_sizes=[5, 10, 15, 20], method='sauvola', mask=None):
    """
    Multi-scale adaptive thresholding combining results from different window sizes.
    
    Args:
        image (np.ndarray): Input grayscale image
        window_sizes (list): List of window sizes to use
        method (str): Thresholding method ('niblack', 'sauvola', 'contrast')
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Contains results from all scales and combined result
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    scale_results = {}
    combined_defects = np.zeros_like(mask, dtype=bool)
    
    # Run thresholding at each scale
    for window_size in window_sizes:
        if method == 'niblack':
            result = niblack_threshold(image, window_size=window_size, mask=mask)
        elif method == 'sauvola':
            result = sauvola_threshold(image, window_size=window_size, mask=mask)
        elif method == 'contrast':
            result = local_contrast_threshold(image, window_size=window_size, mask=mask)
        elif method == 'phansalkar':
            result = phansalkar_threshold(image, window_size=window_size, mask=mask)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scale_results[f'scale_{window_size}'] = result
        combined_defects |= result['defects']
    
    # Calculate combined statistics
    total_pixels = np.sum(mask)
    defect_count = np.sum(combined_defects)
    defect_percentage = (defect_count / total_pixels * 100) if total_pixels > 0 else 0.0
    
    return {
        'scale_results': scale_results,
        'combined_defects': combined_defects,
        'defect_count': defect_count,
        'defect_percentage': defect_percentage,
        'method': method,
        'window_sizes': window_sizes
    }


def adaptive_threshold_ensemble(image, mask=None, voting_threshold=0.5):
    """
    Ensemble of multiple adaptive thresholding methods with majority voting.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
        voting_threshold (float): Minimum vote fraction for defect detection
    
    Returns:
        dict: Results from all methods and ensemble decision
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    methods = {
        'niblack': niblack_threshold(image, mask=mask),
        'sauvola': sauvola_threshold(image, mask=mask),
        'contrast': local_contrast_threshold(image, mask=mask),
        'phansalkar': phansalkar_threshold(image, mask=mask)
    }
    
    # Create voting map
    vote_count = np.zeros_like(mask, dtype=int)
    
    for method_name, result in methods.items():
        vote_count += result['defects'].astype(int)
    
    # Apply voting threshold
    n_methods = len(methods)
    min_votes = int(voting_threshold * n_methods)
    ensemble_defects = vote_count >= min_votes
    
    # Calculate statistics
    total_pixels = np.sum(mask)
    defect_count = np.sum(ensemble_defects)
    defect_percentage = (defect_count / total_pixels * 100) if total_pixels > 0 else 0.0
    
    return {
        'methods': methods,
        'vote_count': vote_count,
        'ensemble_defects': ensemble_defects,
        'defect_count': defect_count,
        'defect_percentage': defect_percentage,
        'voting_threshold': voting_threshold
    }


def visualize_adaptive_threshold_results(image, results, save_path=None):
    """
    Visualize adaptive thresholding results.
    
    Args:
        image (np.ndarray): Original image
        results (dict): Results from adaptive thresholding functions
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Determine the type of results and create appropriate visualization
    if 'methods' in results:  # Ensemble results
        n_methods = len(results['methods'])
        fig, axes = plt.subplots(2, (n_methods + 2) // 2, figsize=(16, 8))
        fig.suptitle('Adaptive Thresholding Ensemble Results', fontsize=16)
        
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        # Show original
        axes[0].imshow(display_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show individual methods
        for i, (method_name, method_result) in enumerate(results['methods'].items(), 1):
            if i < len(axes):
                axes[i].imshow(method_result['defects'], cmap='hot')
                count = np.sum(method_result['defects'])
                axes[i].set_title(f'{method_name.title()}\n{count} defects')
                axes[i].axis('off')
        
        # Show ensemble result
        if len(axes) > len(results['methods']) + 1:
            axes[-1].imshow(results['ensemble_defects'], cmap='hot')
            axes[-1].set_title(f'Ensemble\n{results["defect_count"]} defects')
            axes[-1].axis('off')
        
    elif 'scale_results' in results:  # Multi-scale results
        n_scales = len(results['scale_results'])
        fig, axes = plt.subplots(2, (n_scales + 2) // 2, figsize=(16, 8))
        fig.suptitle(f'Multi-scale {results["method"].title()} Thresholding', fontsize=16)
        
        axes = axes.flatten() if n_scales > 1 else [axes]
        
        # Show original
        axes[0].imshow(display_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show scale results
        for i, (scale_name, scale_result) in enumerate(results['scale_results'].items(), 1):
            if i < len(axes):
                axes[i].imshow(scale_result['defects'], cmap='hot')
                count = np.sum(scale_result['defects'])
                window_size = scale_result['window_size']
                axes[i].set_title(f'Window {window_size}\n{count} defects')
                axes[i].axis('off')
        
        # Show combined result
        if len(axes) > len(results['scale_results']) + 1:
            axes[-1].imshow(results['combined_defects'], cmap='hot')
            axes[-1].set_title(f'Combined\n{results["defect_count"]} defects')
            axes[-1].axis('off')
    
    else:  # Single method result
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('Adaptive Thresholding Results', fontsize=16)
        
        # Original image
        axes[0].imshow(display_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Threshold map
        if 'threshold_map' in results:
            axes[1].imshow(results['threshold_map'], cmap='viridis')
            axes[1].set_title('Threshold Map')
            axes[1].axis('off')
        elif 'contrast_map' in results:
            axes[1].imshow(results['contrast_map'], cmap='viridis')
            axes[1].set_title('Contrast Map')
            axes[1].axis('off')
        
        # Binary result
        axes[2].imshow(results['defects'], cmap='hot')
        count = np.sum(results['defects'])
        axes[2].set_title(f'Defects\n{count} pixels')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """
    Example usage and testing of adaptive thresholding functions.
    """
    # Create a test image with artificial defects
    test_image = np.random.normal(128, 20, (200, 200)).astype(np.uint8)
    
    # Add some artificial defects
    test_image[50:60, 50:60] = 255  # Bright defect
    test_image[100:110, 100:110] = 50  # Dark defect
    test_image[150, 50:150] = 200  # Bright line
    
    # Add gradient to make it more challenging
    x, y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    gradient = (x + y) * 30
    test_image = np.clip(test_image.astype(float) + gradient, 0, 255).astype(np.uint8)
    
    print("Testing Adaptive Thresholding Module")
    print("=" * 50)
    
    # Test individual methods
    print("\n1. Niblack's method:")
    niblack_result = niblack_threshold(test_image)
    print(f"   Found {np.sum(niblack_result['defects'])} defect pixels")
    
    print("\n2. Sauvola's method:")
    sauvola_result = sauvola_threshold(test_image)
    print(f"   Found {np.sum(sauvola_result['defects'])} defect pixels")
    
    print("\n3. Local contrast method:")
    contrast_result = local_contrast_threshold(test_image)
    print(f"   Found {np.sum(contrast_result['defects'])} defect pixels")
    
    print("\n4. Phansalkar's method:")
    phansalkar_result = phansalkar_threshold(test_image)
    print(f"   Found {np.sum(phansalkar_result['defects'])} defect pixels")
    
    # Test multi-scale
    print("\n5. Multi-scale Sauvola:")
    multiscale_result = multiscale_adaptive_threshold(test_image, method='sauvola')
    print(f"   Combined: {multiscale_result['defect_count']} defects ({multiscale_result['defect_percentage']:.2f}%)")
    
    # Test ensemble
    print("\n6. Ensemble method:")
    ensemble_result = adaptive_threshold_ensemble(test_image)
    print(f"   Ensemble: {ensemble_result['defect_count']} defects ({ensemble_result['defect_percentage']:.2f}%)")
    
    # Visualize results
    visualize_adaptive_threshold_results(test_image, ensemble_result, 'adaptive_threshold_test.png')
    
    return ensemble_result


if __name__ == "__main__":
    results = main()
