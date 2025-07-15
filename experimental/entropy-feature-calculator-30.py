import numpy as np
from scipy import stats, ndimage

def extract_entropy_features(gray: np.ndarray) -> dict:
    """Extract various entropy measures."""
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist_norm = hist / (hist.sum() + 1e-10)
    
    shannon = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
    renyi = -np.log2(np.sum(hist_norm**2) + 1e-10)
    tsallis = (1 - np.sum(hist_norm**2))
    
    def local_entropy_func(region):
        hist_local, _ = np.histogram(region, bins=10, range=(0, 255))
        hist_local_norm = hist_local / (hist_local.sum() + 1e-10)
        return -np.sum(hist_local_norm * np.log2(hist_local_norm + 1e-10))
    
    local_ent = ndimage.generic_filter(gray, local_entropy_func, size=9)
    
    return {
        'entropy_shannon': float(shannon),
        'entropy_renyi': float(renyi),
        'entropy_tsallis': float(tsallis),
        'entropy_local_mean': float(np.mean(local_ent)),
        'entropy_local_std': float(np.std(local_ent)),
        'entropy_local_max': float(np.max(local_ent)),
        'entropy_local_min': float(np.min(local_ent)),
    }

if __name__ == '__main__':
    sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    print("Running entropy feature extraction on a sample random image...")
    features = extract_entropy_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
