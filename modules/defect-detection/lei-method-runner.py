
import cv2
import numpy as np

def run_lei_method(gray_image, analysis_mask):
    """
    Implements the LEI ("Linear Enhancement Inspector") method to find
    low-contrast linear scratches at various angles. 

    Args:
        gray_image (np.array): The preprocessed grayscale image.
        analysis_mask (np.array): Mask to focus the analysis on the fiber face.

    Returns:
        np.array: A binary mask showing detected scratch defects.
    """
    print("Stage 3B: Running LEI for Scratch Defects...")
    enhanced_image = cv2.equalizeHist(gray_image)
    final_scratch_map = np.zeros(enhanced_image.shape, dtype=np.uint8)
    
    for angle in range(0, 180, 15):
        kernel_len = 15
        line_kernel = np.zeros((kernel_len, kernel_len), dtype=np.float32)
        line_kernel[int((kernel_len-1)/2), :] = 1
        
        center = (int((kernel_len-1)/2), int((kernel_len-1)/2))
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_kernel = cv2.warpAffine(line_kernel, rot_mat, (kernel_len, kernel_len))
        
        if cv2.sumElems(rotated_kernel)[0] > 0:
            rotated_kernel /= cv2.sumElems(rotated_kernel)[0]
        
        response_map = cv2.filter2D(enhanced_image, -1, rotated_kernel)

        _, thresholded_map = cv2.threshold(response_map, 140, 255, cv2.THRESH_BINARY)
        
        final_scratch_map = cv2.bitwise_or(final_scratch_map, thresholded_map)

    final_scratch_map = cv2.bitwise_and(final_scratch_map, analysis_mask)

    print("Stage 3B: LEI Complete.")
    return final_scratch_map

if __name__ == '__main__':
    # Create a dummy image and mask
    sz = 600
    gray_image = np.full((sz, sz), 150, dtype=np.uint8)
    cv2.circle(gray_image, (sz//2, sz//2), 200, 180, -1)
    # Add a scratch
    cv2.line(gray_image, (250, 250), (400, 400), 120, 3)
    
    analysis_mask = np.zeros((sz, sz), dtype=np.uint8)
    cv2.circle(analysis_mask, (sz//2, sz//2), 200, 255, -1)
    
    # Run the function
    scratch_mask = run_lei_method(gray_image, analysis_mask)
    
    cv2.imshow('Original Gray', gray_image)
    cv2.imshow('LEI Scratch Mask', scratch_mask)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
