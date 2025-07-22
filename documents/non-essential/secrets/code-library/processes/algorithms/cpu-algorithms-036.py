
import cv2
import numpy as np
from typing import Tuple

def detect_region_defects_do2mr(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect region-based defects using DO2MR (Difference of Min-Max Ranking) from test3.py
    Returns: (binary_mask, labeled_defects)
    """
    do2mr_params = {
        "kernel_size": (15, 15),
        "gamma": 3.0
    }
    
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        do2mr_params["kernel_size"]
    )
    
    img_max = cv2.dilate(image, kernel)
    img_min = cv2.erode(image, kernel)
    
    residual = img_max.astype(np.float32) - img_min.astype(np.float32)
    
    residual_filtered = cv2.medianBlur(residual.astype(np.uint8), 3)
    
    mean_val = np.mean(residual_filtered)
    std_val = np.std(residual_filtered)
    threshold = mean_val + do2mr_params["gamma"] * std_val
    
    _, binary_mask = cv2.threshold(residual_filtered, threshold, 255, cv2.THRESH_BINARY)
    
    kernel_open = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    
    n_labels, labeled = cv2.connectedComponents(binary_mask)
    
    return binary_mask, labeled

if __name__ == '__main__':
    # Create a dummy image with a region defect
    sz = 500
    dummy_image = np.full((sz, sz), 128, dtype=np.uint8)
    cv2.circle(dummy_image, (sz//2, sz//2), 200, 150, -1)
    # Add a "dig" or "pit"
    cv2.circle(dummy_image, (300, 250), 15, 80, -1)
    dummy_image = cv2.GaussianBlur(dummy_image, (5,5), 0)

    # Detect defects
    binary_mask, labeled_image = detect_region_defects_do2mr(dummy_image)
    
    # Visualize the results
    labeled_display = np.zeros((*dummy_image.shape, 3), dtype=np.uint8)
    # Color labels for visualization
    if labeled_image.max() > 0:
        labeled_display = cv2.normalize(labeled_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        labeled_display = cv2.applyColorMap(labeled_display, cv2.COLORMAP_JET)

    cv2.imshow('Original Image', dummy_image)
    cv2.imshow('Binary Defect Mask', binary_mask)
    cv2.imshow('Labeled Defects', labeled_display)
    
    print(f"Found {labeled_image.max()} potential region defects.")
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
