
import cv2
import numpy as np
from typing import Tuple

def find_fiber_center(image: np.ndarray) -> Tuple[int, int]:
    """
    Find the center of the fiber using Hough Circle Transform (from test3.py)
    """
    # Apply edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # Find circles using Hough Transform
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=image.shape[0]//4,
        param1=50,
        param2=30,
        minRadius=50,
        maxRadius=image.shape[0]//2
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Select the most prominent circle (usually the largest)
        largest_circle = circles[0][np.argmax(circles[0][:, 2])]
        return (largest_circle[0], largest_circle[1])
    else:
        # Fallback to image center
        print("Hough transform could not find a circle, falling back to image center.")
        return (image.shape[1]//2, image.shape[0]//2)

if __name__ == '__main__':
    # Create a dummy image with a circle
    sz = 500
    dummy_image = np.zeros((sz, sz), dtype=np.uint8)
    center_true = (sz//2 + 10, sz//2 - 10) # Offset center
    radius_true = 150
    cv2.circle(dummy_image, center_true, radius_true, 200, 10)
    dummy_image = cv2.GaussianBlur(dummy_image, (5,5), 0)

    # Find the center
    detected_center = find_fiber_center(dummy_image)
    
    print(f"True Center: {center_true}")
    print(f"Detected Center: {detected_center}")
    
    # Visualize the result
    display_image = cv2.cvtColor(dummy_image, cv2.COLOR_GRAY2BGR)
    cv2.circle(display_image, detected_center, 5, (0, 0, 255), -1) # Red dot for detected center
    cv2.putText(display_image, f"Detected: {detected_center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Fiber Center Detection', display_image)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
