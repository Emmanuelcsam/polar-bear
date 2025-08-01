#!/usr/bin/env python3

import numpy as np


def compute_skewness(data):
    """Compute skewness of data."""
    # Calculate mean of data
    mean = np.mean(data)
    # Calculate standard deviation
    std = np.std(data)
    # Handle zero standard deviation case
    if std == 0:
        return 0.0
    # Compute third standardized moment (skewness)
    return np.mean(((data - mean) / std) ** 3)


def compute_kurtosis(data):
    """Compute kurtosis of data."""
    # Calculate mean of data
    mean = np.mean(data)
    # Calculate standard deviation
    std = np.std(data)
    # Handle zero standard deviation case
    if std == 0:
        return 0.0
    # Compute fourth standardized moment minus 3 (excess kurtosis)
    return np.mean(((data - mean) / std) ** 4) - 3


def compute_entropy(data, bins=256):
    """Compute Shannon entropy."""
    # Create histogram with specified bins in range 0-255
    hist, _ = np.histogram(data, bins=bins, range=(0, 256))
    # Normalize histogram to get probability distribution
    hist = hist / (hist.sum() + 1e-10)
    # Remove zero bins to avoid log(0)
    hist = hist[hist > 0]
    # Compute Shannon entropy: -Σ(p * log2(p))
    return -np.sum(hist * np.log2(hist + 1e-10))


def compute_correlation(x, y):
    """Compute Pearson correlation coefficient."""
    # Need at least 2 points for correlation
    if len(x) < 2:
        return 0.0
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # Calculate covariance
    cov = np.mean((x - x_mean) * (y - y_mean))
    # Calculate standard deviations
    std_x = np.std(x)
    std_y = np.std(y)
    # Handle zero standard deviation
    if std_x == 0 or std_y == 0:
        return 0.0
    # Return correlation coefficient
    return cov / (std_x * std_y)


def compute_spearman_correlation(x, y):
    """Compute Spearman rank correlation."""
    # Need at least 2 points
    if len(x) < 2:
        return 0.0
    # Convert values to ranks using double argsort trick
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    # Compute Pearson correlation on ranks
    return compute_correlation(rank_x, rank_y)


def compute_ks_statistic(x, y):
    """Compute Kolmogorov-Smirnov statistic."""
    # Sort both arrays
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Combine and sort all values
    combined = np.concatenate([x_sorted, y_sorted])
    combined_sorted = np.sort(combined)
    
    # Compute empirical CDFs and find maximum difference
    max_diff = 0
    for val in combined_sorted:
        # Compute CDF at this value for both distributions
        cdf_x = np.sum(x_sorted <= val) / len(x_sorted)
        cdf_y = np.sum(y_sorted <= val) / len(y_sorted)
        # Update maximum difference
        max_diff = max(max_diff, abs(cdf_x - cdf_y))
    
    return max_diff


def compute_wasserstein_distance(x, y):
    """Compute 1D Wasserstein distance."""
    # Sort both arrays
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Interpolate to same size for comparison
    n = max(len(x_sorted), len(y_sorted))
    # Create interpolation points
    x_interp = np.interp(np.linspace(0, 1, n), 
                         np.linspace(0, 1, len(x_sorted)), x_sorted)
    y_interp = np.interp(np.linspace(0, 1, n), 
                         np.linspace(0, 1, len(y_sorted)), y_sorted)
    
    # Compute average absolute difference
    return np.mean(np.abs(x_interp - y_interp)) 