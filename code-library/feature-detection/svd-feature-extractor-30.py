import numpy as np
from scipy import stats

def extract_svd_features(gray: np.ndarray) -> dict:
    """Extract Singular Value Decomposition features."""
    try:
        # Ensure input is float for SVD
        _, s, _ = np.linalg.svd(gray.astype(np.float32), full_matrices=False)
        s_norm = s / (np.sum(s) + 1e-10)
        
        cumsum = np.cumsum(s_norm)
        n_components_90 = np.argmax(cumsum >= 0.9) + 1 if len(cumsum) > 0 else 0
        n_components_95 = np.argmax(cumsum >= 0.95) + 1 if len(cumsum) > 0 else 0
        
        return {
            'svd_largest': float(s[0]) if len(s) > 0 else 0.0,
            'svd_top5_ratio': float(np.sum(s_norm[:5])),
            'svd_top10_ratio': float(np.sum(s_norm[:10])),
            'svd_entropy': float(stats.entropy(s_norm + 1e-10)),
            'svd_energy_ratio': float(s[0] / (s[1] + 1e-10)) if len(s) > 1 else 0.0,
            'svd_n_components_90': float(n_components_90),
            'svd_n_components_95': float(n_components_95),
            'svd_effective_rank': float(np.exp(stats.entropy(s_norm + 1e-10))),
        }
    except np.linalg.LinAlgError:
        print("SVD did not converge.")
        return {f'svd_{k}': 0.0 for k in ['largest', 'top5_ratio', 'top10_ratio', 'entropy', 'energy_ratio', 'n_components_90', 'n_components_95', 'effective_rank']}

if __name__ == '__main__':
    sample_image = np.random.rand(100, 100) * 255
    
    print("Running SVD feature extraction on a sample random image...")
    features = extract_svd_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
