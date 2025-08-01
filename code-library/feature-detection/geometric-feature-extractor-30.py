import numpy as np
import cv2

def extract_shape_features(gray: np.ndarray) -> dict:
    """Extract shape features using Hu moments."""
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    features = {}
    for i, hu in enumerate(hu_moments):
        features[f'shape_hu_{i}'] = float(-np.sign(hu) * np.log10(abs(hu) + 1e-10))
        
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        features['shape_centroid_x'] = float(cx / gray.shape[1])
        features['shape_centroid_y'] = float(cy / gray.shape[0])
    else:
        features['shape_centroid_x'] = 0.5
        features['shape_centroid_y'] = 0.5
        
    return features

if __name__ == '__main__':
    sample_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(sample_image, (60, 60), 30, 255, -1)
    
    print("Running shape feature extraction on a sample image...")
    features = extract_shape_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

