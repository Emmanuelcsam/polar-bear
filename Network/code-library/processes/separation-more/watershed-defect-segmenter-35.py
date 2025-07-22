import cv2
import numpy as np

def watershed_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """Watershed segmentation for defect detection"""
    # Pre-process for watershed
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    
    # Apply watershed
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    # Create mask from watershed lines
    watershed_mask = np.zeros_like(image)
    watershed_mask[markers == -1] = 255
    watershed_mask = cv2.bitwise_and(watershed_mask, watershed_mask, mask=zone_mask)
    
    return watershed_mask

if __name__ == '__main__':
    # Create a sample image with touching objects
    sample_image = np.full((200, 200), 200, dtype=np.uint8)
    cv2.circle(sample_image, (75, 100), 40, 50, -1)
    cv2.circle(sample_image, (125, 100), 40, 50, -1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running watershed detection...")
    watershed_mask = watershed_detection(sample_image, zone_mask)

    cv2.imwrite("watershed_input.png", sample_image)
    cv2.imwrite("watershed_mask.png", watershed_mask)
    print("Saved 'watershed_input.png' and 'watershed_mask.png' for visual inspection.")
