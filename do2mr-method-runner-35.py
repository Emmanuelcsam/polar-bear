
import cv2
import numpy as np

def run_do2mr_method(gray_image, analysis_mask):
    """
    Implements the DO2MR ("Difference of Min-Max Ranking") filter to find
    region-based defects like dirt, pits, and oil.

    Args:
        gray_image (np.array): The preprocessed grayscale image.
        analysis_mask (np.array): Mask to focus the analysis on the fiber face.

    Returns:
        np.array: A binary mask showing detected region-based defects.
    """
    print("Stage 3A: Running DO2MR for Region-Based Defects...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    img_max = cv2.dilate(gray_image, kernel)
    img_min = cv2.erode(gray_image, kernel)
    residual_map = cv2.subtract(img_max, img_min)

    mu, sigma = cv2.meanStdDev(residual_map, mask=analysis_mask)
    
    gamma = 3.0
    threshold_value = mu[0][0] + gamma * sigma[0][0]
    
    _, defect_mask = cv2.threshold(residual_map, threshold_value, 255, cv2.THRESH_BINARY)

    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, opening_kernel)
    
    defect_mask = cv2.bitwise_and(defect_mask, analysis_mask)

    print("Stage 3A: DO2MR Complete.")
    return defect_mask

if __name__ == '__main__':
    # Create a dummy image and mask
    sz = 600
    gray_image = np.full((sz, sz), 150, dtype=np.uint8)
    cv2.circle(gray_image, (sz//2, sz//2), 200, 180, -1)
    # Add a defect
    cv2.circle(gray_image, (400, 300), 10, 50, -1)
    
    analysis_mask = np.zeros((sz, sz), dtype=np.uint8)
    cv2.circle(analysis_mask, (sz//2, sz//2), 200, 255, -1)
    
    # Run the function
    defect_mask = run_do2mr_method(gray_image, analysis_mask)
    
    cv2.imshow('Original Gray', gray_image)
    cv2.imshow('DO2MR Defect Mask', defect_mask)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
