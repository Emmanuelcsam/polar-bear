
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

def _create_line_se(length, angle):
    """
    Create line structuring element.
    """
    angle_rad = angle * np.pi / 180
    
    # Create line coordinates
    x = np.arange(length) - length // 2
    y = np.zeros(length)
    
    # Rotate
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    
    # Convert to image coordinates
    x_rot = np.round(x_rot).astype(int)
    y_rot = np.round(y_rot).astype(int)
    
    # Create structuring element
    se_size = length + 2
    se = np.zeros((se_size, se_size), dtype=np.uint8)
    
    # Draw line
    for i in range(len(x_rot)):
        se[y_rot[i] + se_size//2, x_rot[i] + se_size//2] = 1
        
    return se

def morphological_scratch_refinement(mask):
    """
    Morphological refinement for scratch-like structures.
    """
    # Create line structuring elements at different angles
    refined = np.zeros_like(mask, dtype=np.uint8)
    
    for angle in np.arange(0, 180, 15):
        # Create line SE
        length = 15
        se = _create_line_se(length, angle)
        
        # Morphological closing to connect fragments
        closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se)
        
        # Opening to remove small objects
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se)
        
        refined = np.logical_or(refined, opened)
        
    return refined.astype(np.uint8)

if __name__ == '__main__':
    # Create a dummy binary mask with fragmented lines
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(mask, (50, 50), (100, 50), 255, 2)
    cv2.line(mask, (120, 50), (180, 50), 255, 2) # Fragmented line
    cv2.line(mask, (50, 150), (150, 200), 255, 2)
    cv2.circle(mask, (200, 200), 5, 255, -1) # Some noise

    # Apply morphological scratch refinement
    refined_mask = morphological_scratch_refinement(mask)

    # Display the results
    cv2.imshow('Original Mask', mask)
    cv2.imshow('Refined Mask', refined_mask * 255)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
