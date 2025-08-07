#!/usr/bin/env python3
"""
Statistical feature extraction module for comprehensive image analysis.
Extracts various statistical measures from grayscale images.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple


def compute_skewness(data: np.ndarray) -> float:
    """
    Compute skewness of data distribution.
    
    Args:
        data: Input data array
        
    Returns:
        Skewness value (float)
    """
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    # Third standardized moment
    return float(np.mean(((data - mean) / std) ** 3))


def compute_kurtosis(data: np.ndarray) -> float:
    """
    Compute kurtosis (excess) of data distribution.
    
    Args:
        data: Input data array
        
    Returns:
        Kurtosis value (float)
    """
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    # Fourth standardized moment minus 3
    return float(np.mean(((data - mean) / std) ** 4) - 3)


def compute_entropy(data: np.ndarray, bins: int = 256) -> float:
    """
    Compute Shannon entropy of data.
    
    Args:
        data: Input data array
        bins: Number of histogram bins (default: 256)
        
    Returns:
        Entropy value (float)
    """
    # Create histogram
    hist, _ = np.histogram(data, bins=bins, range=(0, 256))
    
    # Normalize to probability distribution
    hist = hist / (hist.sum() + 1e-10)
    
    # Remove zero bins
    hist = hist[hist > 0]
    
    # Compute entropy
    return float(-np.sum(hist * np.log2(hist + 1e-10)))


def extract_basic_statistics(gray: np.ndarray) -> Dict[str, float]:
    """
    Extract basic statistical features from grayscale image.
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        Dictionary of statistical features
    """
    flat = gray.flatten()
    percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
    
    return {
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray)),
        'variance': float(np.var(gray)),
        'skewness': compute_skewness(flat),
        'kurtosis': compute_kurtosis(flat),
        'min': float(np.min(gray)),
        'max': float(np.max(gray)),
        'range': float(np.max(gray) - np.min(gray)),
        'median': float(np.median(gray)),
        'mad': float(np.median(np.abs(gray - np.median(gray)))),
        'iqr': float(percentiles[3] - percentiles[1]),
        'entropy': compute_entropy(gray),
        'energy': float(np.sum(gray**2)),
        'p10': float(percentiles[0]),
        'p25': float(percentiles[1]),
        'p50': float(percentiles[2]),
        'p75': float(percentiles[3]),
        'p90': float(percentiles[4])
    }


def extract_histogram_features(gray: np.ndarray, bins: int = 32) -> Dict[str, float]:
    """
    Extract histogram-based features.
    
    Args:
        gray: Grayscale image (uint8)
        bins: Number of histogram bins (default: 32)
        
    Returns:
        Dictionary of histogram features
    """
    # Calculate histogram
    hist, bin_edges = np.histogram(gray.flatten(), bins=bins, range=(0, 256))
    hist_norm = hist / (hist.sum() + 1e-10)
    
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Histogram statistics
    hist_mean = float(np.sum(bin_centers * hist_norm))
    hist_std = float(np.sqrt(np.sum((bin_centers - hist_mean)**2 * hist_norm)))
    
    # Find mode (most frequent bin)
    mode_idx = np.argmax(hist)
    mode_value = float(bin_centers[mode_idx])
    mode_freq = float(hist_norm[mode_idx])
    
    # Histogram uniformity (how evenly distributed)
    uniformity = float(np.sum(hist_norm**2))
    
    return {
        'hist_mean': hist_mean,
        'hist_std': hist_std,
        'hist_mode': mode_value,
        'hist_mode_freq': mode_freq,
        'hist_uniformity': uniformity,
        'hist_max_bin': float(np.max(hist_norm)),
        'hist_min_bin': float(np.min(hist_norm[hist_norm > 0]) if np.any(hist_norm > 0) else 0)
    }


def extract_texture_statistics(gray: np.ndarray, window_size: int = 5) -> Dict[str, float]:
    """
    Extract local texture statistics using sliding window.
    
    Args:
        gray: Grayscale image (uint8)
        window_size: Size of sliding window (default: 5)
        
    Returns:
        Dictionary of texture features
    """
    # Pad image for sliding window
    pad = window_size // 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    # Initialize arrays for local statistics
    h, w = gray.shape
    local_means = np.zeros((h, w))
    local_stds = np.zeros((h, w))
    
    # Compute local statistics
    for i in range(h):
        for j in range(w):
            window = padded[i:i+window_size, j:j+window_size]
            local_means[i, j] = np.mean(window)
            local_stds[i, j] = np.std(window)
    
    return {
        'texture_mean_of_means': float(np.mean(local_means)),
        'texture_std_of_means': float(np.std(local_means)),
        'texture_mean_of_stds': float(np.mean(local_stds)),
        'texture_std_of_stds': float(np.std(local_stds)),
        'texture_contrast': float(np.max(local_means) - np.min(local_means)),
        'texture_homogeneity': float(1.0 / (1.0 + np.var(local_stds)))
    }


def extract_moment_features(gray: np.ndarray) -> Dict[str, float]:
    """
    Extract image moments and Hu moments.
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        Dictionary of moment features
    """
    # Calculate moments
    moments = cv2.moments(gray)
    
    # Hu moments (rotation invariant)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    features = {}
    
    # Store Hu moments with log transform for scale invariance
    for i, hu in enumerate(hu_moments):
        features[f'hu_moment_{i}'] = float(-np.sign(hu) * np.log10(abs(hu) + 1e-10))
    
    # Centroid location (normalized)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        features['centroid_x'] = float(cx / gray.shape[1])
        features['centroid_y'] = float(cy / gray.shape[0])
    else:
        features['centroid_x'] = 0.5
        features['centroid_y'] = 0.5
    
    return features


def extract_all_statistical_features(gray: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive statistical features from image.
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        Dictionary containing all statistical features
    """
    features = {}
    
    # Basic statistics
    features.update(extract_basic_statistics(gray))
    
    # Histogram features
    features.update(extract_histogram_features(gray))
    
    # Texture statistics
    features.update(extract_texture_statistics(gray))
    
    # Moment features
    features.update(extract_moment_features(gray))
    
    return features


def compare_feature_vectors(features1: Dict[str, float], 
                           features2: Dict[str, float]) -> Dict[str, float]:
    """
    Compare two feature vectors and compute similarity metrics.
    
    Args:
        features1: First feature dictionary
        features2: Second feature dictionary
        
    Returns:
        Dictionary of comparison metrics
    """
    # Get common keys
    keys = sorted(set(features1.keys()) & set(features2.keys()))
    
    if not keys:
        return {'similarity': 0.0, 'distance': float('inf')}
    
    # Convert to arrays
    vec1 = np.array([features1[k] for k in keys])
    vec2 = np.array([features2[k] for k in keys])
    
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    
    return {
        'euclidean_distance': float(np.linalg.norm(vec1 - vec2)),
        'manhattan_distance': float(np.sum(np.abs(vec1 - vec2))),
        'cosine_similarity': float(np.dot(vec1_norm, vec2_norm)),
        'correlation': float(np.corrcoef(vec1, vec2)[0, 1]) if len(vec1) > 1 else 0.0
    }


def main():
    """Standalone test function."""
    print("Statistical Features Module - Standalone Test")
    print("-" * 40)
    
    # Create synthetic test image
    test_image = np.zeros((200, 200), dtype=np.uint8)
    
    # Add gradient pattern
    for i in range(200):
        test_image[i, :] = i * 255 // 200
    
    # Add some noise
    noise = np.random.randint(0, 30, test_image.shape, dtype=np.uint8)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    print("Extracting features from test image...")
    features = extract_all_statistical_features(test_image)
    
    print(f"\nExtracted {len(features)} features:")
    
    # Display features by category
    print("\nBasic Statistics:")
    for key in ['mean', 'std', 'skewness', 'kurtosis', 'entropy']:
        if key in features:
            print(f"  {key}: {features[key]:.3f}")
    
    print("\nPercentiles:")
    for key in ['p10', 'p25', 'p50', 'p75', 'p90']:
        if key in features:
            print(f"  {key}: {features[key]:.3f}")
    
    print("\nHistogram Features:")
    for key in features:
        if key.startswith('hist_'):
            print(f"  {key}: {features[key]:.3f}")
    
    print("\nTexture Features:")
    for key in features:
        if key.startswith('texture_'):
            print(f"  {key}: {features[key]:.3f}")
    
    print("\nHu Moments:")
    for i in range(7):
        key = f'hu_moment_{i}'
        if key in features:
            print(f"  {key}: {features[key]:.3f}")
    
    # Create slightly modified version
    test_image2 = cv2.GaussianBlur(test_image, (5, 5), 1.0)
    features2 = extract_all_statistical_features(test_image2)
    
    # Compare features
    print("\nComparing original vs blurred image:")
    comparison = compare_feature_vectors(features, features2)
    for metric, value in comparison.items():
        print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    main()
