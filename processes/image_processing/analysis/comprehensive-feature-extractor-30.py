import cv2
import numpy as np
import warnings
from typing import Dict, List, Tuple

# Import all the individual feature extraction functions
from .extract_statistical_features import extract_statistical_features
from .extract_matrix_norms import extract_matrix_norms
from .extract_lbp_features import extract_lbp_features
from .extract_glcm_features import extract_glcm_features
from .extract_fourier_features import extract_fourier_features
from .extract_wavelet_features import extract_wavelet_features
from .extract_morphological_features import extract_morphological_features
from .extract_shape_features import extract_shape_features
from .extract_svd_features import extract_svd_features
from .extract_entropy_features import extract_entropy_features
from .extract_gradient_features import extract_gradient_features
from .extract_topological_proxy_features import extract_topological_proxy_features

def extract_ultra_comprehensive_features(image: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
    """Extract 100+ features using all available methods."""
    features = {}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    print("  Extracting features: ", end='', flush=True)
    
    feature_extractors = [
        ("Stats", extract_statistical_features), ("Norms", extract_matrix_norms),
        ("LBP", extract_lbp_features), ("GLCM", extract_glcm_features),
        ("FFT", extract_fourier_features), ("Wavelet", extract_wavelet_features),
        ("Morph", extract_morphological_features), ("Shape", extract_shape_features),
        ("SVD", extract_svd_features), ("Entropy", extract_entropy_features),
        ("Gradient", extract_gradient_features), ("Topology", extract_topological_proxy_features),
    ]
    
    for name, extractor in feature_extractors:
        print(name, end=' -> ', flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                features.update(extractor(gray))
        except Exception as e:
            print(f"(✗:{e})", end='', flush=True)
    
    print("✓")
    
    feature_names = sorted(features.keys())
    return features, feature_names

if __name__ == '__main__':
    sample_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    print("Running ultra comprehensive feature extraction...")
    features, names = extract_ultra_comprehensive_features(sample_image)
    
    print(f"\nSuccessfully extracted {len(names)} features.")
    print("Sample features:")
    for i, name in enumerate(names[:5]):
        print(f"  - {name}: {features[name]:.4f}")
