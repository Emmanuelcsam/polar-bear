import numpy as np
import cv2
from typing import List, Tuple

def apply_linear_detector(image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
    """Apply linear detector for scratch detection"""
    h, w = image.shape
    response = np.zeros_like(image, dtype=np.float32)
    
    if not kernel_points:
        return response

    max_offset = max(max(abs(dx), abs(dy)) for dx, dy in kernel_points)
    padded = cv2.copyMakeBorder(image, max_offset, max_offset, max_offset, max_offset, cv2.BORDER_REFLECT)
    
    for y in range(h):
        for x in range(w):
            line_vals = []
            for dx, dy in kernel_points:
                line_vals.append(float(padded[y + max_offset + dy, x + max_offset + dx]))
            
            if line_vals:
                center_val = float(padded[y + max_offset, x + max_offset])
                bright_response = np.mean(line_vals) - center_val
                dark_response = center_val - np.mean(line_vals)
                response[y, x] = max(0, max(bright_response, dark_response))
    
    return response

if __name__ == '__main__':
    # Create a sample image with a dark line on a gray background
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.line(sample_image, (20, 20), (180, 180), 64, 2)

    # Define a kernel for a 45-degree line
    kernel_length = 15
    angle_rad = np.deg2rad(45)
    kernel_points = []
    for i in range(-kernel_length//2, kernel_length//2 + 1):
        if i == 0:
            continue
        dx = int(round(i * np.cos(angle_rad)))
        dy = int(round(i * np.sin(angle_rad)))
        kernel_points.append((dx, dy))

    print("Applying linear detector to a sample image with a line...")
    response = apply_linear_detector(sample_image, kernel_points)

    # Normalize for visualization
    if response.max() > 0:
        cv2.normalize(response, response, 0, 255, cv2.NORM_MINMAX)
    
    response_img = response.astype(np.uint8)

    cv2.imwrite("linear_detector_input.png", sample_image)
    cv2.imwrite("linear_detector_response.png", response_img)
    print("Saved 'linear_detector_input.png' and 'linear_detector_response.png' for visual inspection.")
