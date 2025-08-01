#!/usr/bin/env python3

import numpy as np
from typing import Dict, Tuple


def compute_exhaustive_comparison(features1: Dict, features2: Dict) -> Dict[str, float]:
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
        js_div = 0.5 * sum(hist1[i] * np.log((hist1[i] + 1e-10) / (m[i] + 1e-10)) 
                           for i in range(len(hist1)) if hist1[i] > 0)
        js_div += 0.5 * sum(hist2[i] * np.log((hist2[i] + 1e-10) / (m[i] + 1e-10)) 
                            for i in range(len(hist2)) if hist2[i] > 0)
        comparison['js_divergence'] = float(js_div)
        
        # Chi-square distance: χ² = 0.5 * Σ (P(i) - Q(i))² / (P(i) + Q(i))
        chi_sq = 0.5 * np.sum(np.where(hist1 + hist2 > 0, 
                                       (hist1 - hist2)**2 / (hist1 + hist2 + 1e-10), 0))
        comparison['chi_square'] = float(chi_sq)
    else:
        # Default values if not enough bins
        comparison['kl_divergence'] = float('inf')
        comparison['js_divergence'] = 1.0
        comparison['chi_square'] = float('inf')
    
    # Wasserstein distance (1D approximation)
    comparison['wasserstein_distance'] = float(compute_wasserstein_distance(vec1, vec2))
    
    return comparison


def compute_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(vec1) < 2:
        return 0.0
    
    # Compute means
    mean1 = np.mean(vec1)
    mean2 = np.mean(vec2)
    
    # Compute correlation
    numerator = np.sum((vec1 - mean1) * (vec2 - mean2))
    denominator = np.sqrt(np.sum((vec1 - mean1)**2) * np.sum((vec2 - mean2)**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_spearman_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(vec1) < 2:
        return 0.0
    
    # Convert to ranks
    rank1 = np.argsort(np.argsort(vec1))
    rank2 = np.argsort(np.argsort(vec2))
    
    # Compute correlation on ranks
    return compute_correlation(rank1, rank2)


def compute_ks_statistic(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic."""
    if len(vec1) < 2 or len(vec2) < 2:
        return 1.0
    
    # Sort vectors
    sorted1 = np.sort(vec1)
    sorted2 = np.sort(vec2)
    
    # Compute empirical CDFs
    n1, n2 = len(sorted1), len(sorted2)
    
    # Create combined sorted array
    combined = np.concatenate([sorted1, sorted2])
    combined.sort()
    
    # Compute CDFs at each point
    cdf1 = np.searchsorted(sorted1, combined, side='right') / n1
    cdf2 = np.searchsorted(sorted2, combined, side='right') / n2
    
    # Compute KS statistic
    ks_stat = np.max(np.abs(cdf1 - cdf2))
    
    return float(ks_stat)


def compute_wasserstein_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute 1D Wasserstein distance (Earth Mover's Distance)."""
    if len(vec1) < 1 or len(vec2) < 1:
        return float('inf')
    
    # Sort both vectors
    sorted1 = np.sort(vec1)
    sorted2 = np.sort(vec2)
    
    # Compute cumulative distributions
    n1, n2 = len(sorted1), len(sorted2)
    
    # Create combined sorted array
    combined = np.concatenate([sorted1, sorted2])
    combined.sort()
    
    # Compute CDFs at each point
    cdf1 = np.searchsorted(sorted1, combined, side='right') / n1
    cdf2 = np.searchsorted(sorted2, combined, side='right') / n2
    
    # Compute Wasserstein distance as area between CDFs
    wasserstein = np.sum(np.abs(cdf1 - cdf2)) * (combined[1] - combined[0]) if len(combined) > 1 else 0
    
    return float(wasserstein)


def test_comparison_metrics():
    """Test function for comparison metrics."""
    # Create test feature sets
    features1 = {
        'mean': 100.0,
        'std': 15.0,
        'skew': 0.5,
        'kurtosis': 2.0,
        'entropy': 4.5
    }
    
    features2 = {
        'mean': 105.0,
        'std': 18.0,
        'skew': 0.3,
        'kurtosis': 1.8,
        'entropy': 4.2
    }
    
    # Compute comparison
    comparison = compute_exhaustive_comparison(features1, features2)
    
    # Print results
    print("Comparison Metrics:")
    for key, value in comparison.items():
        print(f"  {key}: {value:.4f}")
    
    return comparison


if __name__ == "__main__":
    test_comparison_metrics() 