import numpy as np
import cv2

def create_test_image(path="test_image.png"):
    """
    Creates a simple 100x100 grayscale gradient image for testing.
    """
    try:
        height, width = 100, 100
        image = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                image[i, j] = int((i + j) / (height + width) * 255)
        
        cv2.imwrite(path, image)
        print(f"Created test image at: {path}")
        return path
    except Exception as e:
        print(f"Failed to create test image: {e}")
        return None

if __name__ == "__main__":
    create_test_image()