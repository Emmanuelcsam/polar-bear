import numpy as np
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
from typing import Dict

def compute_exhaustive_comparison(features1: Dict[str, float], features2: Dict[str, float]) -> Dict[str, float]:
    """Compute all possible comparison metrics between two feature sets."""
    keys = sorted(set(features1.keys()) & set(features2.keys()))
    if not keys:
        return { 'euclidean_distance': float('inf'), 'manhattan_distance': float('inf'), 'cosine_distance': 1.0 }

    vec1 = np.array([features1[k] for k in keys])
    vec2 = np.array([features2[k] for k in keys])
    
    # Normalize for some metrics
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    vec1_norm = vec1 / (norm1 + 1e-10)
    vec2_norm = vec2 / (norm2 + 1e-10)
    
    comparison = {
        'euclidean_distance': float(np.linalg.norm(vec1 - vec2)),
        'manhattan_distance': float(np.sum(np.abs(vec1 - vec2))),
        'chebyshev_distance': float(np.max(np.abs(vec1 - vec2))),
        'cosine_distance': float(1 - np.dot(vec1_norm, vec2_norm)),
        'pearson_correlation': float(np.corrcoef(vec1, vec2)[0, 1]) if len(vec1) > 1 else 1.0,
        'spearman_correlation': float(stats.spearmanr(vec1, vec2)[0]) if len(vec1) > 1 else 1.0,
        'ks_statistic': float(ks_2samp(vec1, vec2)[0]),
        'wasserstein_distance': float(wasserstein_distance(vec1, vec2)),
    }
    
    # Distribution-based metrics
    bins = min(30, len(vec1) // 2)
    if bins > 2:
        hist1, bin_edges = np.histogram(vec1, bins=bins)
        hist2, _ = np.histogram(vec2, bins=bin_edges)
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)
        m = 0.5 * (hist1 + hist2)
        
        comparison['kl_divergence'] = float(stats.entropy(hist1, hist2))
        comparison['js_divergence'] = float(0.5 * (stats.entropy(hist1, m) + stats.entropy(hist2, m)))
        comparison['chi_square'] = float(0.5 * np.sum(np.divide((hist1 - hist2)**2, (hist1 + hist2 + 1e-10))))
    
    return comparison

if __name__ == '__main__':
    # Create two sample feature dictionaries
    features1 = {'stat_mean': 128.5, 'stat_std': 50.2, 'fft_mean_magnitude': 1000.0}
    features2 = {'stat_mean': 135.1, 'stat_std': 55.8, 'fft_mean_magnitude': 950.0}
    
    print("Running exhaustive comparison on two sample feature sets...")
    comparison_results = compute_exhaustive_comparison(features1, features2)
    
    print("\nComparison Results:")
    for key, value in comparison_results.items():
        print(f"  {key}: {value:.4f}")