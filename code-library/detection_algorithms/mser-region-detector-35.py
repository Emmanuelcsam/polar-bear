import cv2
import numpy as np

def mser_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """MSER blob detection"""
    # MSER works best on 8-bit single-channel images
    if image.dtype != np.uint8:
        image = np.uint8(image)

    mser = cv2.MSER_create(delta=5, min_area=10, max_area=1000)
    regions, _ = mser.detectRegions(image)
    
    mask = np.zeros_like(image)
    if regions is not None:
        for region in regions:
            cv2.fillPoly(mask, [region], 255)
    
    mask = cv2.bitwise_and(mask, mask, mask=zone_mask)
    return mask

if __name__ == '__main__':
    # Create a sample image with blobs
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.circle(sample_image, (50, 50), 15, 80, -1)
    cv2.rectangle(sample_image, (120, 120), (160, 180), 160, -1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running MSER detection...")
    mser_mask = mser_detection(sample_image, zone_mask)

    cv2.imwrite("mser_input.png", sample_image)
    cv2.imwrite("mser_mask.png", mser_mask)
    print("Saved 'mser_input.png' and 'mser_mask.png' for visual inspection.")
