#!/usr/bin/env python3

import numpy as np
import cv2
from statistical_functions import (
    compute_correlation, compute_spearman_correlation,
    compute_ks_statistic, compute_wasserstein_distance
)


def compute_exhaustive_comparison(features1, features2):
    """Compute all possible comparison metrics between two feature sets."""
    # Get common feature keys between both sets
    keys = sorted(set(features1.keys()) & set(features2.keys()))
    # Handle case with no common features
    if not keys:
        return {
            'euclidean_distance': float('inf'),
            'manhattan_distance': float('inf'),
            'chebyshev_distance': float('inf'),
            'cosine_distance': 1.0,
            'pearson_correlation': 0.0,
            'spearman_correlation': 0.0,
            'ks_statistic': 1.0,
            'kl_divergence': float('inf'),
            'js_divergence': 1.0,
            'chi_square': float('inf'),
            'wasserstein_distance': float('inf'),
            'feature_ssim': 0.0,
        }
    
    # Convert feature dictionaries to vectors
    vec1 = np.array([features1[k] for k in keys])
    vec2 = np.array([features2[k] for k in keys])
    
    # Handle empty vectors
    if len(vec1) == 0 or len(vec2) == 0:
        return compute_exhaustive_comparison({}, {})
    
    # Normalize vectors to unit length
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    vec1_norm = vec1 / (norm1 + 1e-10)
    vec2_norm = vec2 / (norm2 + 1e-10)
    
    # Initialize comparison dictionary
    comparison = {}
    
    # Distance metrics
    comparison['euclidean_distance'] = float(np.linalg.norm(vec1 - vec2))
    comparison['manhattan_distance'] = float(np.sum(np.abs(vec1 - vec2)))
    comparison['chebyshev_distance'] = float(np.max(np.abs(vec1 - vec2)))
    comparison['cosine_distance'] = float(1 - np.dot(vec1_norm, vec2_norm))
    
    # Correlation measures
    comparison['pearson_correlation'] = float(compute_correlation(vec1, vec2))
    comparison['spearman_correlation'] = float(compute_spearman_correlation(vec1, vec2))
    
    # Statistical tests
    comparison['ks_statistic'] = float(compute_ks_statistic(vec1, vec2))
    
    # Information theoretic measures
    bins = min(30, len(vec1) // 2)  # Adaptive bin count
    if bins > 2:
        # Create normalized histograms for both vectors
        min_val = min(vec1.min(), vec2.min())
        max_val = max(vec1.max(), vec2.max())
        
        # Compute histograms with same bins
        hist1, bin_edges = np.histogram(vec1, bins=bins, range=(min_val, max_val))
        hist2, _ = np.histogram(vec2, bins=bin_edges)
        
        # Normalize to probabilities
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)
        
        # KL divergence: D_KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
        kl_div = 0
        for i in range(len(hist1)):
            if hist1[i] > 0:
                kl_div += hist1[i] * np.log((hist1[i] + 1e-10) / (hist2[i] + 1e-10))
        comparison['kl_divergence'] = float(kl_div)
        
        # JS divergence: symmetric version of KL
        m = 0.5 * (hist1 + hist2)  # Average distribution
        js_div = 0.5 * sum(hist1[i] * np.log((hist1[i] + 1e-10) / (m[i] + 1e-10)) for i in range(len(hist1)) if hist1[i] > 0)
        js_div += 0.5 * sum(hist2[i] * np.log((hist2[i] + 1e-10) / (m[i] + 1e-10)) for i in range(len(hist2)) if hist2[i] > 0)
        comparison['js_divergence'] = float(js_div)
        
        # Chi-square distance: χ² = 0.5 * Σ (P(i) - Q(i))² / (P(i) + Q(i))
        chi_sq = 0.5 * np.sum(np.where(hist1 + hist2 > 0, (hist1 - hist2)**2 / (hist1 + hist2 + 1e-10), 0))
        comparison['chi_square'] = float(chi_sq)
    else:
        # Default values if not enough bins
        comparison['kl_divergence'] = float('inf')
        comparison['js_divergence'] = 1.0
        comparison['chi_square'] = float('inf')
    
    # Wasserstein distance (1D approximation)
    comparison['wasserstein_distance'] = float(compute_wasserstein_distance(vec1, vec2))
    
    # Feature SSIM (simplified structural similarity)
    mean1, mean2 = np.mean(vec1), np.mean(vec2)
    comparison['feature_ssim'] = float((2 * mean1 * mean2 + 1e-10) / (mean1**2 + mean2**2 + 1e-10))
    
    return comparison


def compute_image_structural_comparison(img1, img2):
    """Compute structural similarity between images."""
    # Ensure images have same dimensions
    if img1.shape != img2.shape:
        # Use maximum dimensions
        h, w = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
        # Resize both images
        img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # SSIM implementation constants
    C1 = (0.01 * 255)**2  # Constant to stabilize luminance
    C2 = (0.03 * 255)**2  # Constant to stabilize contrast
    
    # Create Gaussian window for local statistics
    kernel = cv2.getGaussianKernel(11, 1.5)  # 11x11 kernel, sigma=1.5
    window = np.outer(kernel, kernel.transpose())  # 2D kernel
    
    # Compute local means
    mu1 = cv2.filter2D(img1.astype(float), -1, window)
    mu2 = cv2.filter2D(img2.astype(float), -1, window)
    
    # Compute local statistics
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = cv2.filter2D(img1.astype(float)**2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2.astype(float)**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1.astype(float) * img2.astype(float), -1, window) - mu1_mu2
    
    # SSIM components
    # Luminance comparison
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    # Contrast comparison
    contrast = (2 * np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2) / (sigma1_sq + sigma2_sq + C2)
    # Structure comparison
    structure = (sigma12 + C2/2) / (np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2/2)
    
    # Combine components
    ssim_map = luminance * contrast * structure
    # Average SSIM
    ssim_index = np.mean(ssim_map)
    
    # Multi-scale SSIM at different resolutions
    ms_ssim_values = [ssim_index]
    for scale in [2, 4]:
        # Downsample images
        img1_scaled = cv2.resize(img1, (img1.shape[1]//scale, img1.shape[0]//scale))
        img2_scaled = cv2.resize(img2, (img2.shape[1]//scale, img2.shape[0]//scale))
        
        # Simplified SSIM for other scales
        diff = np.abs(img1_scaled.astype(float) - img2_scaled.astype(float))
        ms_ssim = 1 - np.mean(diff) / 255
        ms_ssim_values.append(ms_ssim)
    
    return {
        'ssim': float(ssim_index),                        # Overall SSIM
        'ssim_map': ssim_map,                             # Pixel-wise SSIM
        'ms_ssim': ms_ssim_values,                        # Multi-scale SSIM
        'luminance_map': luminance,                       # Luminance comparison map
        'contrast_map': contrast,                         # Contrast comparison map
        'structure_map': structure,                       # Structure comparison map
        'mean_luminance': float(np.mean(luminance)),     # Average luminance similarity
        'mean_contrast': float(np.mean(contrast)),       # Average contrast similarity
        'mean_structure': float(np.mean(structure)),     # Average structure similarity
    } 