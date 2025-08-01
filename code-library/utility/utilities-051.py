import numpy as np
from scipy import ndimage
import cv2

def scale_normalized_log(image: np.ndarray, zone_mask: np.ndarray, scales: list[float]) -> np.ndarray:
    """Scale-normalized Laplacian of Gaussian for blob detection"""
    blob_response = np.zeros_like(image, dtype=np.float32)
    
    for scale in scales:
        log = ndimage.gaussian_laplace(image.astype(np.float32), sigma=scale)
        # Scale normalization
        log *= scale**2
        blob_response = np.maximum(blob_response, np.abs(log))
    
    if blob_response.max() > 0:
        cv2.normalize(blob_response, blob_response, 0, 255, cv2.NORM_MINMAX)
        
    _, blob_mask = cv2.threshold(blob_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blob_mask = cv2.bitwise_and(blob_mask, blob_mask, mask=zone_mask)
    
    return blob_mask

if __name__ == '__main__':
    from defect_detection_config import DefectDetectionConfig

    config = DefectDetectionConfig()

    # Create a sample image with blobs of different sizes
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.circle(sample_image, (50, 50), 5, 64, -1)
    cv2.circle(sample_image, (150, 150), 10, 192, -1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running scale-normalized LoG detection...")
    log_mask = scale_normalized_log(sample_image, zone_mask, config.log_scales)

    cv2.imwrite("log_input.png", sample_image)
    cv2.imwrite("log_mask.png", log_mask)
    print("Saved 'log_input.png' and 'log_mask.png' for visual inspection.")
