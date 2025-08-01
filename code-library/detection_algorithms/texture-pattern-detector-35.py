import cv2
import numpy as np

def lbp_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """
    Simplified LBP-based anomaly detection.
    This implementation uses local variance of the image as a proxy for
    texture anomaly, which is related to the concept of LBP.
    """
    # Using texture variance as anomaly measure
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    
    # Ensure image is float for calculations
    image_float = image.astype(np.float32)
    
    local_mean = cv2.filter2D(image_float, -1, kernel)
    local_sq_mean = cv2.filter2D(image_float**2, -1, kernel)
    local_var = local_sq_mean - local_mean**2
    
    if local_var.max() > 0:
        cv2.normalize(local_var, local_var, 0, 255, cv2.NORM_MINMAX)
    
    _, anomaly_mask = cv2.threshold(local_var.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    anomaly_mask = cv2.bitwise_and(anomaly_mask, anomaly_mask, mask=zone_mask)
    
    return anomaly_mask

if __name__ == '__main__':
    # Create a sample image with a textured region
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    # Add a noisy/textured patch
    patch = np.random.randint(100, 150, (50, 50), dtype=np.uint8)
    sample_image[75:125, 75:125] = patch
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running LBP-based anomaly detection...")
    lbp_mask = lbp_detection(sample_image, zone_mask)

    cv2.imwrite("lbp_input.png", sample_image)
    cv2.imwrite("lbp_mask.png", lbp_mask)
    print("Saved 'lbp_input.png' and 'lbp_mask.png' for visual inspection.")
