import numpy as np
from skimage.feature import local_binary_pattern
from scipy import stats

def extract_lbp_features(gray: np.ndarray) -> dict:
    """Extract Local Binary Pattern features."""
    features = {}
    for radius in [1, 2, 3, 5]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), density=True, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        
        features[f'lbp_r{radius}_mean'] = float(np.mean(lbp))
        features[f'lbp_r{radius}_std'] = float(np.std(lbp))
        features[f'lbp_r{radius}_entropy'] = float(stats.entropy(hist + 1e-10))
        features[f'lbp_r{radius}_energy'] = float(np.sum(hist**2))
        
    return features

if __name__ == '__main__':
    sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    print("Running LBP feature extraction on a sample random image...")
    features = extract_lbp_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
