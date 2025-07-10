import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def determinant_of_hessian(image: np.ndarray, zone_mask: np.ndarray, scales: list[float]) -> np.ndarray:
    """Determinant of Hessian blob detection"""
    doh_response = np.zeros_like(image, dtype=np.float32)
    
    for scale in scales:
        smoothed = gaussian_filter(image.astype(np.float32), scale)
        
        Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
        Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
        Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
        
        # Determinant of Hessian
        det = Hxx * Hyy - Hxy**2
        # Scale normalization
        det *= scale**4
        
        doh_response = np.maximum(doh_response, np.abs(det))
    
    if doh_response.max() > 0:
        cv2.normalize(doh_response, doh_response, 0, 255, cv2.NORM_MINMAX)

    _, doh_mask = cv2.threshold(doh_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    doh_mask = cv2.bitwise_and(doh_mask, doh_mask, mask=zone_mask)
    
    return doh_mask

if __name__ == '__main__':
    from defect_detection_config import DefectDetectionConfig

    config = DefectDetectionConfig()

    # Create a sample image with blobs
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.circle(sample_image, (50, 50), 8, 64, -1)
    cv2.circle(sample_image, (150, 150), 12, 192, -1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running Determinant of Hessian (DoH) detection...")
    # Use a subset of scales for DoH as in the original script
    doh_mask = determinant_of_hessian(sample_image, zone_mask, config.log_scales[::2])

    cv2.imwrite("doh_input.png", sample_image)
    cv2.imwrite("doh_mask.png", doh_mask)
    print("Saved 'doh_input.png' and 'doh_mask.png' for visual inspection.")
