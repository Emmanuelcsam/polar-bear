#!/usr/bin/env python3
"""
Texture Analysis Module for Fiber Optic Defect Detection
========================================================

This module implements texture-based defect detection methods using Local Binary Patterns,
Haralick features, and other texture descriptors. Extracted from advanced_defect_analysis.py.

Functions:
- Local Binary Pattern (LBP) analysis
- Haralick texture features
- Local variance analysis
- Texture anomaly detection

Author: Extracted from Advanced Fiber Analysis Team
"""

import numpy as np
import cv2
from scipy import ndimage
try:
    from skimage import feature
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Some texture features will be limited.")
    SKIMAGE_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


def local_binary_pattern_analysis(image, radius=1, n_points=8, mask=None, method='uniform'):
    """
    Compute Local Binary Pattern (LBP) for texture analysis.
    
    Args:
        image (np.ndarray): Input grayscale image
        radius (int): Radius of the circular neighborhood
        n_points (int): Number of sampling points (typically 8*radius)
        mask (np.ndarray, optional): Binary mask for region of interest
        method (str): LBP method ('uniform', 'default', 'var')
    
    Returns:
        dict: LBP histogram, image, and texture features
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    if SKIMAGE_AVAILABLE:
        # Use scikit-image implementation
        lbp = local_binary_pattern(image, n_points, radius, method=method)
    else:
        # Simple manual implementation
        lbp = manual_lbp(image, radius, n_points)
    
    # Calculate LBP histogram
    if method == 'uniform':
        bins = n_points + 2
    else:
        bins = 2**n_points
    
    lbp_values = lbp[mask]
    hist, bin_edges = np.histogram(lbp_values, bins=bins, range=(0, bins), density=True)
    
    # Calculate texture features
    uniformity = np.sum(hist**2)  # Energy/uniformity
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Entropy
    contrast = np.var(lbp_values)  # Local contrast measure
    
    return {
        'lbp_image': lbp,
        'histogram': hist,
        'bin_edges': bin_edges,
        'uniformity': uniformity,
        'entropy': entropy,
        'contrast': contrast,
        'method': method,
        'radius': radius,
        'n_points': n_points
    }


def manual_lbp(image, radius=1, n_points=8):
    """
    Manual implementation of Local Binary Pattern.
    
    Args:
        image (np.ndarray): Input grayscale image
        radius (int): Radius of circular neighborhood
        n_points (int): Number of sampling points
    
    Returns:
        np.ndarray: LBP image
    """
    h, w = image.shape
    lbp = np.zeros_like(image, dtype=np.uint8)
    
    # Calculate sampling points on circle
    angles = 2 * np.pi * np.arange(n_points) / n_points
    dx = radius * np.cos(angles)
    dy = radius * np.sin(angles)
    
    # Process each pixel (excluding border)
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center_value = image[i, j]
            pattern = 0
            
            for k in range(n_points):
                # Calculate neighbor coordinates
                ni = int(round(i + dy[k]))
                nj = int(round(j + dx[k]))
                
                # Ensure coordinates are within bounds
                ni = max(0, min(h-1, ni))
                nj = max(0, min(w-1, nj))
                
                # Compare with center
                if image[ni, nj] >= center_value:
                    pattern |= (1 << k)
            
            lbp[i, j] = pattern
    
    return lbp


def lbp_anomaly_detection(image, mask=None, window_size=15, threshold_percentile=95):
    """
    Detect texture anomalies using LBP with sliding window analysis.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
        window_size (int): Size of sliding window
        threshold_percentile (float): Percentile threshold for anomaly detection
    
    Returns:
        dict: Anomaly map and detected defects
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Calculate LBP
    lbp_result = local_binary_pattern_analysis(image, mask=mask)
    lbp = lbp_result['lbp_image']
    reference_hist = lbp_result['histogram']
    
    # Sliding window analysis
    h, w = image.shape
    anomaly_map = np.zeros_like(image, dtype=np.float32)
    
    half_window = window_size // 2
    
    for i in range(half_window, h - half_window):
        for j in range(half_window, w - half_window):
            if mask[i, j]:
                # Extract window
                window = lbp[i-half_window:i+half_window+1, 
                           j-half_window:j+half_window+1]
                
                # Calculate window histogram
                window_hist, _ = np.histogram(
                    window.flatten(), 
                    bins=len(reference_hist), 
                    range=(0, len(reference_hist)), 
                    density=True
                )
                
                # Calculate chi-square distance
                chi_sq = np.sum((window_hist - reference_hist)**2 / 
                               (window_hist + reference_hist + 1e-10))
                anomaly_map[i, j] = chi_sq
    
    # Threshold anomaly map
    threshold = np.percentile(anomaly_map[mask], threshold_percentile)
    defects = (anomaly_map > threshold) & mask
    
    return {
        'anomaly_map': anomaly_map,
        'defects': defects,
        'threshold': threshold,
        'lbp_result': lbp_result
    }


def haralick_texture_features(image, distances=[1], angles=[0, 45, 90, 135], mask=None):
    """
    Compute Haralick texture features from Gray Level Co-occurrence Matrix (GLCM).
    
    Args:
        image (np.ndarray): Input grayscale image
        distances (list): List of pixel distances
        angles (list): List of angles in degrees
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Haralick texture features
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Quantize image to reduce computational complexity
    quantized = (image // 16).astype(np.uint8)  # 16 gray levels
    
    if SKIMAGE_AVAILABLE:
        # Convert angles to radians
        angles_rad = [np.radians(angle) for angle in angles]
        
        # Compute GLCM
        glcm = graycomatrix(
            quantized, 
            distances=distances, 
            angles=angles_rad,
            levels=16,
            symmetric=True,
            normed=True
        )
        
        # Calculate Haralick features
        features = {
            'contrast': graycoprops(glcm, 'contrast').mean(),
            'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
            'energy': graycoprops(glcm, 'energy').mean(),
            'correlation': graycoprops(glcm, 'correlation').mean(),
            'asm': graycoprops(glcm, 'ASM').mean()  # Angular Second Moment
        }
    else:
        # Manual GLCM computation (simplified)
        features = manual_glcm_features(quantized, mask)
    
    return features


def manual_glcm_features(image, mask):
    """
    Manual computation of basic GLCM features.
    
    Args:
        image (np.ndarray): Quantized grayscale image
        mask (np.ndarray): Binary mask
    
    Returns:
        dict: Basic texture features
    """
    # Simple local variance and contrast measures
    local_var = ndimage.generic_filter(image.astype(float), np.var, size=3)
    local_mean = ndimage.generic_filter(image.astype(float), np.mean, size=3)
    
    # Calculate features over masked region
    masked_var = local_var[mask]
    masked_mean = local_mean[mask]
    
    features = {
        'contrast': np.var(masked_var),
        'homogeneity': 1.0 / (1.0 + np.var(masked_var)),
        'energy': np.mean(masked_var**2),
        'correlation': np.corrcoef(masked_var, masked_mean)[0, 1] if len(masked_var) > 1 else 0,
        'local_variance': np.mean(masked_var),
        'local_mean': np.mean(masked_mean)
    }
    
    return features


def local_variance_analysis(image, window_sizes=[3, 5, 7, 9], mask=None):
    """
    Multi-scale local variance analysis for texture characterization.
    
    Args:
        image (np.ndarray): Input grayscale image
        window_sizes (list): List of window sizes for analysis
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Variance maps and features at different scales
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64)
    results = {}
    
    for window_size in window_sizes:
        # Calculate local variance
        local_var = ndimage.generic_filter(img_float, np.var, size=window_size)
        
        # Calculate statistics
        var_values = local_var[mask]
        mean_var = np.mean(var_values)
        std_var = np.std(var_values)
        
        # Detect high variance regions (potential defects)
        threshold = mean_var + 2 * std_var
        high_var_regions = (local_var > threshold) & mask
        
        results[f'window_{window_size}'] = {
            'variance_map': local_var,
            'high_variance_regions': high_var_regions,
            'mean_variance': mean_var,
            'std_variance': std_var,
            'threshold': threshold,
            'defect_count': np.sum(high_var_regions)
        }
    
    return results


def gabor_texture_analysis(image, frequencies=[0.1, 0.3, 0.5], angles=[0, 45, 90, 135], mask=None):
    """
    Gabor filter bank for texture analysis.
    
    Args:
        image (np.ndarray): Input grayscale image
        frequencies (list): List of spatial frequencies
        angles (list): List of orientations in degrees
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Gabor responses and texture features
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64) / 255.0
    h, w = image.shape
    
    responses = {}
    feature_vector = []
    
    for freq in frequencies:
        for angle in angles:
            # Create Gabor kernel
            kernel_size = 31
            sigma = 3.0
            
            # Generate Gabor kernel
            kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size), 
                sigma, 
                np.radians(angle), 
                2*np.pi*freq, 
                0.5, 
                0, 
                ktype=cv2.CV_32F
            )
            
            # Apply filter
            filtered = cv2.filter2D(img_float, cv2.CV_8UC3, kernel)
            
            # Calculate response statistics
            response_values = filtered[mask]
            if len(response_values) > 0:
                mean_response = np.mean(response_values)
                std_response = np.std(response_values)
                energy = np.mean(response_values**2)
            else:
                mean_response = std_response = energy = 0
            
            key = f'freq_{freq:.1f}_angle_{angle}'
            responses[key] = {
                'filtered_image': filtered,
                'mean_response': mean_response,
                'std_response': std_response,
                'energy': energy
            }
            
            feature_vector.extend([mean_response, std_response, energy])
    
    return {
        'responses': responses,
        'feature_vector': np.array(feature_vector),
        'frequencies': frequencies,
        'angles': angles
    }


def comprehensive_texture_analysis(image, mask=None):
    """
    Comprehensive texture analysis combining multiple methods.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Combined texture analysis results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    results = {}
    
    print("Running comprehensive texture analysis...")
    
    # 1. LBP analysis
    print("  - Local Binary Patterns...")
    lbp_result = local_binary_pattern_analysis(image, mask=mask)
    results['lbp'] = lbp_result
    
    # 2. LBP anomaly detection
    print("  - LBP anomaly detection...")
    lbp_anomaly = lbp_anomaly_detection(image, mask=mask)
    results['lbp_anomaly'] = lbp_anomaly
    
    # 3. Haralick features
    print("  - Haralick texture features...")
    haralick_features = haralick_texture_features(image, mask=mask)
    results['haralick'] = haralick_features
    
    # 4. Local variance analysis
    print("  - Local variance analysis...")
    variance_analysis = local_variance_analysis(image, mask=mask)
    results['variance'] = variance_analysis
    
    # 5. Gabor texture analysis
    print("  - Gabor filter analysis...")
    gabor_analysis = gabor_texture_analysis(image, mask=mask)
    results['gabor'] = gabor_analysis
    
    # Combine defect detections
    combined_defects = np.zeros_like(mask, dtype=bool)
    
    # Add LBP anomalies
    combined_defects |= lbp_anomaly['defects']
    
    # Add high variance regions (from largest window)
    largest_window = max(variance_analysis.keys())
    combined_defects |= variance_analysis[largest_window]['high_variance_regions']
    
    results['combined_defects'] = combined_defects
    results['defect_count'] = np.sum(combined_defects)
    results['defect_percentage'] = (np.sum(combined_defects) / np.sum(mask) * 100) if np.sum(mask) > 0 else 0
    
    return results


def visualize_texture_results(image, results, save_path=None):
    """
    Visualize texture analysis results.
    
    Args:
        image (np.ndarray): Original image
        results (dict): Results from texture analysis
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Texture Analysis Results', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(display_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # LBP image
    if 'lbp' in results:
        axes[0, 1].imshow(results['lbp']['lbp_image'], cmap='gray')
        axes[0, 1].set_title('Local Binary Pattern')
        axes[0, 1].axis('off')
    
    # LBP anomalies
    if 'lbp_anomaly' in results:
        axes[0, 2].imshow(results['lbp_anomaly']['defects'], cmap='hot')
        count = np.sum(results['lbp_anomaly']['defects'])
        axes[0, 2].set_title(f'LBP Anomalies ({count} pixels)')
        axes[0, 2].axis('off')
    
    # Variance map
    if 'variance' in results:
        largest_window = max(results['variance'].keys())
        variance_map = results['variance'][largest_window]['variance_map']
        axes[1, 0].imshow(variance_map, cmap='viridis')
        axes[1, 0].set_title(f'Local Variance ({largest_window})')
        axes[1, 0].axis('off')
    
    # High variance regions
    if 'variance' in results:
        high_var = results['variance'][largest_window]['high_variance_regions']
        axes[1, 1].imshow(high_var, cmap='hot')
        count = np.sum(high_var)
        axes[1, 1].set_title(f'High Variance Regions ({count} pixels)')
        axes[1, 1].axis('off')
    
    # Combined defects
    if 'combined_defects' in results:
        axes[1, 2].imshow(results['combined_defects'], cmap='hot')
        count = results['defect_count']
        percentage = results['defect_percentage']
        axes[1, 2].set_title(f'Combined Defects\n{count} pixels ({percentage:.2f}%)')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """
    Example usage and testing of texture analysis functions.
    """
    # Create a test image with different textures
    test_image = np.random.normal(128, 20, (200, 200)).astype(np.uint8)
    
    # Add textured regions
    # Checkerboard pattern
    for i in range(50, 100, 10):
        for j in range(50, 100, 10):
            if (i//10 + j//10) % 2 == 0:
                test_image[i:i+10, j:j+10] = 200
    
    # Random noise region
    noise_region = np.random.normal(100, 50, (30, 30))
    test_image[120:150, 120:150] = np.clip(noise_region, 0, 255)
    
    # Smooth region
    test_image[150:180, 50:80] = 150
    
    print("Testing Texture Analysis Module")
    print("=" * 50)
    
    # Run comprehensive analysis
    results = comprehensive_texture_analysis(test_image)
    
    # Print summary
    print(f"\nTexture Analysis Summary:")
    print(f"LBP uniformity: {results['lbp']['uniformity']:.4f}")
    print(f"LBP entropy: {results['lbp']['entropy']:.4f}")
    
    if 'haralick' in results:
        print(f"Haralick contrast: {results['haralick']['contrast']:.4f}")
        print(f"Haralick homogeneity: {results['haralick']['homogeneity']:.4f}")
    
    print(f"Total defects found: {results['defect_count']} ({results['defect_percentage']:.2f}%)")
    
    # Visualize results
    visualize_texture_results(test_image, results, 'texture_analysis_test.png')
    
    return results


if __name__ == "__main__":
    results = main()
