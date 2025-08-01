import numpy as np
import pywt

def extract_wavelet_features(gray: np.ndarray) -> dict:
    """Extract multi-level wavelet features."""
    features = {}
    
    for wavelet in ['db4', 'haar', 'sym4']:
        try:
            coeffs = pywt.wavedec2(gray, wavelet, level=3)
            
            cA = coeffs[0]
            features[f'wavelet_{wavelet}_approx_mean'] = float(np.mean(np.abs(cA)))
            features[f'wavelet_{wavelet}_approx_energy'] = float(np.sum(np.square(cA, dtype=np.float64)))
            
            for i, (cH, cV, cD) in enumerate(coeffs[1:], 1):
                for name, coeff in [('H', cH), ('V', cV), ('D', cD)]:
                    features[f'wavelet_{wavelet}_L{i}_{name}_energy'] = float(np.sum(np.square(coeff, dtype=np.float64)))
                    features[f'wavelet_{wavelet}_L{i}_{name}_mean'] = float(np.mean(np.abs(coeff)))
        except Exception as e:
            print(f"Could not compute wavelet {wavelet}: {e}")
            pass
            
    return features

if __name__ == '__main__':
    sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    print("Running wavelet feature extraction on a sample random image...")
    features = extract_wavelet_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
