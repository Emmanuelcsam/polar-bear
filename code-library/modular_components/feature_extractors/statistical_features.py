#!/usr/bin/env python3

import numpy as np
from typing import Dict


def extract_statistical_features(gray: np.ndarray) -> Dict[str, float]:
    """Extract comprehensive statistical features from grayscale image."""
    # Flatten 2D image to 1D array for statistics
    flat = gray.flatten()
    # Calculate percentiles at 10, 25, 50, 75, 90
    percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
    
    return {
        'stat_mean': float(np.mean(gray)),  # Average pixel value
        'stat_std': float(np.std(gray)),    # Standard deviation
        'stat_variance': float(np.var(gray)),  # Variance
        'stat_skew': float(compute_skewness(flat)),  # Distribution skewness
        'stat_kurtosis': float(compute_kurtosis(flat)),  # Distribution kurtosis
        'stat_min': float(np.min(gray)),    # Minimum value
        'stat_max': float(np.max(gray)),    # Maximum value
        'stat_range': float(np.max(gray) - np.min(gray)),  # Value range
        'stat_median': float(np.median(gray)),  # Median value
        # Median absolute deviation
        'stat_mad': float(np.median(
            np.abs(gray - np.median(gray))
        )),
        'stat_iqr': float(percentiles[3] - percentiles[1]),  # Interquartile range
        'stat_entropy': float(compute_entropy(gray)),  # Information entropy
        'stat_energy': float(np.sum(gray**2)),  # Energy (sum of squares)
        'stat_p10': float(percentiles[0]),  # 10th percentile
        'stat_p25': float(percentiles[1]),  # 25th percentile
        'stat_p50': float(percentiles[2]),  # 50th percentile (median)
        'stat_p75': float(percentiles[3]),  # 75th percentile
        'stat_p90': float(percentiles[4]),  # 90th percentile
    }


def compute_skewness(data: np.ndarray) -> float:
    """Compute skewness of data distribution."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    skewness = np.mean(((data - mean) / std) ** 3)
    return float(skewness)


def compute_kurtosis(data: np.ndarray) -> float:
    """Compute kurtosis of data distribution."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    kurtosis = np.mean(((data - mean) / std) ** 4) - 3
    return float(kurtosis)


def compute_entropy(data: np.ndarray) -> float:
    """Compute information entropy of data."""
    # Create histogram with 256 bins (0-255 for grayscale)
    hist, _ = np.histogram(data, bins=256, range=(0, 256))
    # Normalize to probabilities
    hist = hist / hist.sum()
    # Remove zero probabilities
    hist = hist[hist > 0]
    # Compute entropy: -sum(p * log2(p))
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy)


def test_statistical_features():
    """Test function for statistical feature extraction."""
    # Create a test image
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # Extract features
    features = extract_statistical_features(test_image)
    
    # Print results
    print("Statistical Features Extracted:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    return features


if __name__ == "__main__":
    test_statistical_features() 