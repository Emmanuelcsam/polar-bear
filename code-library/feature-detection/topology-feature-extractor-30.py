import numpy as np
from scipy.ndimage import label

def extract_topological_proxy_features(gray: np.ndarray) -> dict:
    """
    Extract topological proxy features using connected components analysis.
    NOTE: This is a simplified approximation of topological features.
    """
    features = {}
    
    thresholds = np.percentile(gray, np.linspace(5, 95, 20))
    
    n_components, n_holes = [], []
    for t in thresholds:
        _, n_comp = label(gray >= t)
        n_components.append(n_comp)
        
        _, n_hl = label(gray < t)
        n_holes.append(n_hl)
    
    if len(n_components) > 1:
        persistence_b0 = np.diff(n_components)
        features['topo_b0_max_components'] = float(np.max(n_components))
        features['topo_b0_mean_components'] = float(np.mean(n_components))
        features['topo_b0_persistence_sum'] = float(np.sum(np.abs(persistence_b0)))
        features['topo_b0_persistence_max'] = float(np.max(np.abs(persistence_b0)))
    
    if len(n_holes) > 1:
        persistence_b1 = np.diff(n_holes)
        features['topo_b1_max_holes'] = float(np.max(n_holes))
        features['topo_b1_mean_holes'] = float(np.mean(n_holes))
        features['topo_b1_persistence_sum'] = float(np.sum(np.abs(persistence_b1)))
        features['topo_b1_persistence_max'] = float(np.max(np.abs(persistence_b1)))
    
    return features

if __name__ == '__main__':
    # Create an image with some components and holes
    sample_image = np.zeros((100, 100), dtype=np.uint8)
    sample_image[20:40, 20:40] = 100 # component 1
    sample_image[60:80, 60:80] = 150 # component 2
    sample_image[25:35, 25:35] = 0   # hole in component 1

    print("Running topological proxy feature extraction...")
    features = extract_topological_proxy_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
