
import cv2
import numpy as np
from typing import List, Tuple

def _apply_linear_detector(image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Apply linear detector at specific orientation (from test3.py)
    """
    height, width = image.shape
    response = np.zeros_like(image, dtype=np.float32)
    
    pad_size = len(kernel_points) // 2
    padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    for y in range(height):
        for x in range(width):
            center_sum = 0
            surround_sum = 0
            valid_points = 0
            
            for dx, dy in kernel_points:
                px = x + dx + pad_size
                py = y + dy + pad_size
                
                if abs(dx) <= 2 and abs(dy) <= 2:
                    center_sum += padded[py, px]
                else:
                    surround_sum += padded[py, px]
                valid_points += 1
            
            if valid_points > 0:
                center_avg = center_sum / (valid_points * 0.3) if (valid_points * 0.3) > 0 else 0
                surround_avg = surround_sum / (valid_points * 0.7) if (valid_points * 0.7) > 0 else 0
                response[y, x] = max(0, surround_avg - center_avg)
    
    return response

def detect_scratches_lei(image: np.ndarray) -> np.ndarray:
    """
    Detect scratches using LEI (Linear Enhancement Inspector) from test3.py
    """
    lei_params = {
        "kernel_size": 15,
        "angles": np.arange(0, 180, 15),
        "threshold_factor": 2.0
    }
    
    enhanced = cv2.equalizeHist(image)
    scratch_strength = np.zeros_like(enhanced, dtype=np.float32)
    kernel_length = lei_params["kernel_size"]
    
    for angle in lei_params["angles"]:
        angle_rad = np.deg2rad(angle)
        
        kernel_points = []
        for i in range(-kernel_length//2, kernel_length//2 + 1):
            x = int(i * np.cos(angle_rad))
            y = int(i * np.sin(angle_rad))
            kernel_points.append((x, y))
        
        response = _apply_linear_detector(enhanced, kernel_points)
        scratch_strength = np.maximum(scratch_strength, response)
    
    mean_strength = np.mean(scratch_strength)
    std_strength = np.std(scratch_strength)
    threshold = mean_strength + lei_params["threshold_factor"] * std_strength
    
    _, scratch_mask = cv2.threshold(scratch_strength, threshold, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scratch_mask = cv2.morphologyEx(scratch_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    return scratch_mask

if __name__ == '__main__':
    # Create a dummy image with a scratch
    sz = 500
    dummy_image = np.full((sz, sz), 128, dtype=np.uint8)
    cv2.circle(dummy_image, (sz//2, sz//2), 200, 150, -1)
    # Add a scratch
    cv2.line(dummy_image, (200, 200), (350, 350), 100, 4)
    dummy_image = cv2.GaussianBlur(dummy_image, (5,5), 0)

    # Detect scratches
    scratch_mask = detect_scratches_lei(dummy_image)
    
    # Visualize the results
    cv2.imshow('Original Image', dummy_image)
    cv2.imshow('Detected Scratch Mask', scratch_mask)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
