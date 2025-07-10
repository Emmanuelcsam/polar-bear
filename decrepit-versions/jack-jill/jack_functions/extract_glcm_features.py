import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(gray: np.ndarray) -> dict:
    """Extract Gray-Level Co-occurrence Matrix features."""
    # Quantize image for faster computation
    img_q = (gray // 32).astype(np.uint8)
    
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(img_q, distances=distances, angles=angles, 
                       levels=8, symmetric=True, normed=True)
    
    features = {}
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    for prop in props:
        values = graycoprops(glcm, prop).flatten()
        features[f'glcm_{prop}_mean'] = float(np.mean(values))
        features[f'glcm_{prop}_std'] = float(np.std(values))
        features[f'glcm_{prop}_range'] = float(np.max(values) - np.min(values))
        
    return features

if __name__ == '__main__':
    sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    print("Running GLCM feature extraction on a sample random image...")
    features = extract_glcm_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")