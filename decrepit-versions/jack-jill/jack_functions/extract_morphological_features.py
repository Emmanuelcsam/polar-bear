import numpy as np
import cv2
from skimage.morphology import disk, white_tophat, black_tophat

def extract_morphological_features(gray: np.ndarray) -> dict:
    """Extract morphological features."""
    features = {}
    
    for size in [3, 5, 7, 11]:
        selem = disk(size)
        wth = white_tophat(gray, selem)
        bth = black_tophat(gray, selem)
        
        features[f'morph_wth_{size}_mean'] = float(np.mean(wth))
        features[f'morph_wth_{size}_max'] = float(np.max(wth))
        features[f'morph_wth_{size}_sum'] = float(np.sum(wth))
        features[f'morph_bth_{size}_mean'] = float(np.mean(bth))
        features[f'morph_bth_{size}_max'] = float(np.max(bth))
        features[f'morph_bth_{size}_sum'] = float(np.sum(bth))
        
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    
    binary_sum = np.sum(binary)
    features['morph_binary_area_ratio'] = float(binary_sum / (binary.size * 255))
    features['morph_gradient_sum'] = float(np.sum(gradient))
    features['morph_erosion_ratio'] = float(np.sum(erosion) / (binary_sum + 1e-10))
    features['morph_dilation_ratio'] = float(np.sum(dilation) / (binary_sum + 1e-10))
    
    return features

if __name__ == '__main__':
    sample_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(sample_image, (20, 20), (80, 80), 128, -1)
    
    print("Running morphological feature extraction on a sample image...")
    features = extract_morphological_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")