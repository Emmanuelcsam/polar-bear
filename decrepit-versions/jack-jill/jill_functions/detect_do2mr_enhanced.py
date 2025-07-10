import cv2
import numpy as np

from .defect_detection_config import DefectDetectionConfig

def detect_do2mr_enhanced(image: np.ndarray, zone_mask: np.ndarray, config: DefectDetectionConfig) -> np.ndarray:
    """Enhanced DO2MR detection"""
    vote_map = np.zeros_like(image, dtype=np.float32)
    
    for kernel_size in config.do2mr_kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        img_max = cv2.dilate(image, kernel)
        img_min = cv2.erode(image, kernel)
        residual = cv2.absdiff(img_max, img_min)
        residual_filtered = cv2.medianBlur(residual, 5)
        
        for gamma in config.do2mr_gamma_values:
            masked_values = residual_filtered[zone_mask > 0]
            if len(masked_values) == 0:
                continue
                
            mean_val = np.mean(masked_values)
            std_val = np.std(masked_values)
            threshold = mean_val + gamma * std_val
            
            _, binary = cv2.threshold(residual_filtered, threshold, 255, cv2.THRESH_BINARY)
            binary = cv2.bitwise_and(binary, binary, mask=zone_mask)
            
            # Morphological cleanup
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
            
            vote_map += (binary / 255.0)
    
    # Combine votes
    num_combinations = len(config.do2mr_kernel_sizes) * len(config.do2mr_gamma_values)
    if num_combinations == 0:
        return np.zeros_like(image, dtype=np.uint8)
        
    threshold_ratio = 0.3
    combined_mask = (vote_map >= (num_combinations * threshold_ratio)).astype(np.uint8) * 255
    
    return combined_mask

if __name__ == '__main__':
    config = DefectDetectionConfig()

    # Create a sample image with a blob defect
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.circle(sample_image, (100, 100), 10, 180, -1)
    
    # Create a circular zone mask
    zone_mask = np.zeros_like(sample_image)
    cv2.circle(zone_mask, (100, 100), 90, 255, -1)

    print("Running enhanced DO2MR detection...")
    do2mr_mask = detect_do2mr_enhanced(sample_image, zone_mask, config)

    cv2.imwrite("do2mr_input.png", sample_image)
    cv2.imwrite("do2mr_detection_mask.png", do2mr_mask)
    print("Saved 'do2mr_input.png' and 'do2mr_detection_mask.png' for visual inspection.")
