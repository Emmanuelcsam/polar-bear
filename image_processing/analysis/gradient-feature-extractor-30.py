import numpy as np
import cv2

def extract_gradient_features(gray: np.ndarray) -> dict:
    """Extract gradient-based features."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_orient = np.arctan2(grad_y, grad_x)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (edges.size * 255)
    
    return {
        'gradient_magnitude_mean': float(np.mean(grad_mag)),
        'gradient_magnitude_std': float(np.std(grad_mag)),
        'gradient_magnitude_max': float(np.max(grad_mag)),
        'gradient_magnitude_sum': float(np.sum(grad_mag)),
        'gradient_orientation_mean': float(np.mean(grad_orient)),
        'gradient_orientation_std': float(np.std(grad_orient)),
        'laplacian_mean_abs': float(np.mean(np.abs(laplacian))),
        'laplacian_std': float(np.std(laplacian)),
        'laplacian_sum_abs': float(np.sum(np.abs(laplacian))),
        'edge_density': float(edge_density),
        'edge_count': float(np.sum(edges > 0)),
    }

if __name__ == '__main__':
    sample_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(sample_image, (20, 20), (80, 80), 255, -1)
    
    print("Running gradient feature extraction on a sample image...")
    features = extract_gradient_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
