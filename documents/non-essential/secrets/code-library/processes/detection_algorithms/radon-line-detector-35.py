import cv2
import numpy as np

def radon_line_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """
    Radon transform for line detection.
    This is a simplified implementation using Hough Lines, as a full Radon
    transform is computationally expensive and often not necessary for this task.
    """
    edges = cv2.Canny(image, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=zone_mask)
    
    # Use Hough lines, which is conceptually related to the Radon transform for lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
    
    line_mask = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    
    line_mask = cv2.bitwise_and(line_mask, line_mask, mask=zone_mask)
    return line_mask

if __name__ == '__main__':
    # Create a sample image with lines
    sample_image = np.full((200, 200), 64, dtype=np.uint8)
    cv2.line(sample_image, (20, 20), (180, 40), 128, 2)
    cv2.line(sample_image, (50, 180), (150, 30), 128, 2)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running simplified Radon line detection (using Hough)...")
    radon_mask = radon_line_detection(sample_image, zone_mask)

    cv2.imwrite("radon_input.png", sample_image)
    cv2.imwrite("radon_mask.png", radon_mask)
    print("Saved 'radon_input.png' and 'radon_mask.png' for visual inspection.")
