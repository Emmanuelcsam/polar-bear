import cv2
import numpy as np
from typing import Dict, Any, Optional

from .defect_detection_config import DefectDetectionConfig

def find_fiber_center_enhanced(preprocessed_images: Dict[str, np.ndarray], config: DefectDetectionConfig) -> Optional[Dict[str, Any]]:
    """Find fiber center using multiple methods and vote"""
    candidates = []

    # Method 1: Hough circles on different preprocessed images
    for img_name in ['gaussian_5', 'bilateral_0', 'clahe_0', 'median']:
        if img_name not in preprocessed_images:
            continue
            
        img = preprocessed_images[img_name]
        for dp in config.hough_dp_values:
            for p1 in config.hough_param1_values:
                for p2 in config.hough_param2_values:
                    circles = cv2.HoughCircles(
                        img, cv2.HOUGH_GRADIENT, dp=dp,
                        minDist=img.shape[0]//8,
                        param1=p1, param2=p2,
                        minRadius=img.shape[0]//10,
                        maxRadius=img.shape[0]//2
                    )
                    
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for circle in circles[0]:
                            candidates.append({
                                'center': (int(circle[0]), int(circle[1])),
                                'radius': float(circle[2]),
                                'method': f'hough_{img_name}',
                                'confidence': p2 / max(config.hough_param2_values)
                            })
    
    # Method 2: Contour-based detection
    for img_name in ['bilateral_0', 'nlmeans']:
        if img_name not in preprocessed_images:
            continue
            
        img = preprocessed_images[img_name]
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            candidates.append({
                'center': (int(x), int(y)),
                'radius': float(radius),
                'method': f'contour_{img_name}',
                'confidence': 0.8
            })
    
    if not candidates:
        return None
    
    # Vote for best result using clustering
    centers = np.array([c['center'] for c in candidates])
    radii = np.array([c['radius'] for c in candidates])
    
    # Use median for robustness
    best_center = (int(np.median(centers[:, 0])), int(np.median(centers[:, 1])))
    best_radius = float(np.median(radii))
    avg_confidence = np.mean([c['confidence'] for c in candidates])
    
    return {
        'center': best_center,
        'radius': best_radius,
        'confidence': avg_confidence,
        'num_candidates': len(candidates)
    }

if __name__ == '__main__':
    # This script is not meant to be run standalone as it requires a dictionary of preprocessed images.
    # The main purpose is to be imported by other modules.
    print("This script contains the 'find_fiber_center_enhanced' function.")
    print("It is intended to be used as part of the unified defect detection system.")
    
    # Example of how it might be called
    config = DefectDetectionConfig()
    
    # Create a sample image with a circle
    h, w = 480, 640
    center = (w // 2, h // 2)
    radius = 100
    sample_image = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(sample_image, center, radius, 200, -1)
    sample_image = cv2.GaussianBlur(sample_image, (5, 5), 0)
    
    preprocessed = {
        'gaussian_5': sample_image,
        'bilateral_0': sample_image,
        'clahe_0': sample_image,
        'median': sample_image,
        'nlmeans': sample_image,
    }
    
    print("\nRunning on a sample image...")
    fiber_info = find_fiber_center_enhanced(preprocessed, config)
    
    if fiber_info:
        print("\nDetected fiber info:")
        print(fiber_info)
        
        # Draw result on a blank image
        result_img = cv2.cvtColor(sample_image, cv2.COLOR_GRAY2BGR)
        cv2.circle(result_img, fiber_info['center'], int(fiber_info['radius']), (0, 255, 0), 2)
        cv2.imwrite("fiber_center_detection.png", result_img)
        print("\nSaved 'fiber_center_detection.png' with the detected circle.")
    else:
        print("\nCould not detect fiber center in the sample image.")
