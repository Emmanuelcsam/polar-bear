import numpy as np
from scipy import stats

def extract_statistical_features(gray: np.ndarray) -> dict:
    """Extract comprehensive statistical features."""
    flat = gray.flatten()
    percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
    
    return {
        'stat_mean': float(np.mean(gray)),
        'stat_std': float(np.std(gray)),
        'stat_variance': float(np.var(gray)),
        'stat_skew': float(stats.skew(flat)),
        'stat_kurtosis': float(stats.kurtosis(flat)),
        'stat_min': float(np.min(gray)),
        'stat_max': float(np.max(gray)),
        'stat_range': float(np.max(gray) - np.min(gray)),
        'stat_median': float(np.median(gray)),
        'stat_mad': float(np.median(np.abs(gray - np.median(gray)))),
        'stat_iqr': float(percentiles[3] - percentiles[1]),
        'stat_entropy': float(stats.entropy(np.histogram(gray, bins=256)[0] + 1e-10)),
        'stat_energy': float(np.sum(gray.astype(np.float64)**2)), # Use float64 to prevent overflow
        'stat_p10': float(percentiles[0]),
        'stat_p25': float(percentiles[1]),
        'stat_p50': float(percentiles[2]),
        'stat_p75': float(percentiles[3]),
        'stat_p90': float(percentiles[4]),
    }

if __name__ == '__main__':
    # Create a sample grayscale image
    sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    print("Running statistical feature extraction on a sample random image...")
    features = extract_statistical_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
