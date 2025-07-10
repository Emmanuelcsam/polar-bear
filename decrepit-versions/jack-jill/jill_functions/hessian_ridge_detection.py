import numpy as np
from scipy.ndimage import gaussian_filter

def hessian_ridge_detection(image: np.ndarray, zone_mask: np.ndarray, scales: list[float]) -> np.ndarray:
    """Multi-scale Hessian ridge detection"""
    ridge_response = np.zeros_like(image, dtype=np.float32)
    
    for scale in scales:
        smoothed = gaussian_filter(image.astype(np.float32), scale)
        
        Hxx = gaussian_filter(smoothed, scale, order=(0, 2))
        Hyy = gaussian_filter(smoothed, scale, order=(2, 0))
        Hxy = gaussian_filter(smoothed, scale, order=(1, 1))
        
        # Eigenvalue calculation
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy * Hxy
        discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
        
        lambda1 = 0.5 * (trace + discriminant)
        lambda2 = 0.5 * (trace - discriminant)
        
        # Sato's vesselness filter variant
        response = np.zeros_like(lambda1)
        dark_lines = (lambda2 < 0) & (lambda1 < lambda2 / 2.0) # Heuristic for dark lines
        response[dark_lines] = np.abs(lambda2[dark_lines])
        
        ridge_response = np.maximum(ridge_response, response)
    
    if ridge_response.max() > 0:
        cv2.normalize(ridge_response, ridge_response, 0, 255, cv2.NORM_MINMAX)
    
    _, ridge_mask = cv2.threshold(ridge_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ridge_mask = cv2.bitwise_and(ridge_mask, ridge_mask, mask=zone_mask)
    
    return ridge_mask

if __name__ == '__main__':
    import cv2
    from defect_detection_config import DefectDetectionConfig

    config = DefectDetectionConfig()

    # Create a sample image with a thin line
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.line(sample_image, (20, 100), (180, 120), 64, 1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running Hessian ridge detection...")
    hessian_mask = hessian_ridge_detection(sample_image, zone_mask, config.hessian_scales)

    cv2.imwrite("hessian_input.png", sample_image)
    cv2.imwrite("hessian_mask.png", hessian_mask)
    print("Saved 'hessian_input.png' and 'hessian_mask.png' for visual inspection.")
