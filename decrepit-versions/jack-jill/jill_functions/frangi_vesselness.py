import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def frangi_vesselness(image: np.ndarray, zone_mask: np.ndarray, scales: list[float]) -> np.ndarray:
    """Frangi vesselness filter"""
    vesselness = np.zeros_like(image, dtype=np.float32)
    
    for scale in scales:
        smoothed = gaussian_filter(image.astype(np.float32), scale)
        
        Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
        Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
        Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
        
        # Corrected eigenvalue calculation
        tmp = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
        lambda1 = 0.5 * (Hxx + Hyy + tmp)
        lambda2 = 0.5 * (Hxx + Hyy - tmp)
        
        # Ensure lambda2 is the smaller eigenvalue
        idx = np.abs(lambda1) < np.abs(lambda2)
        lambda1[idx], lambda2[idx] = lambda2[idx], lambda1[idx]
        
        Rb = np.divide(np.abs(lambda1), np.abs(lambda2), out=np.full_like(lambda1, 1e-10), where=np.abs(lambda2) > 1e-10)
        S = np.sqrt(lambda1**2 + lambda2**2)
        
        beta = 0.5
        gamma = 15
        
        v = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*gamma**2)))
        v[lambda2 > 0] = 0 # Only care about dark lines (vessel-like structures)
        
        vesselness = np.maximum(vesselness, v)
    
    if vesselness.max() > 0:
        cv2.normalize(vesselness, vesselness, 0, 255, cv2.NORM_MINMAX)

    _, vessel_mask = cv2.threshold(vesselness.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vessel_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=zone_mask)
    
    return vessel_mask

if __name__ == '__main__':
    from defect_detection_config import DefectDetectionConfig

    config = DefectDetectionConfig()

    # Create a sample image with a thin line
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.line(sample_image, (20, 100), (180, 120), 64, 1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running Frangi vesselness detection...")
    frangi_mask = frangi_vesselness(sample_image, zone_mask, config.frangi_scales)

    cv2.imwrite("frangi_input.png", sample_image)
    cv2.imwrite("frangi_mask.png", frangi_mask)
    print("Saved 'frangi_input.png' and 'frangi_mask.png' for visual inspection.")
