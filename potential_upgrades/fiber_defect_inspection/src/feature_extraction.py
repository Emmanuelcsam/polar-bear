# src/feature_extraction.py
import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops

def extract_features(image, defect_regions, zone_masks, metrics):
    """
    Extracts features from the detected defects and zones.
    
    Args:
        image: The input image as a NumPy array.
        defect_regions: A list of detected defect regions.
        zone_masks: A dictionary of zone masks.
        metrics: A dictionary of metrics from the zone segmentation.
        
    Returns:
        A list of feature vectors for each defect.
    """
    features_list = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for i, defect in enumerate(defect_regions):
        features = {}
        
        # Basic properties
        features['defect_id'] = i
        features['area_px'] = defect['area']
        
        # Location
        cx, cy = defect['centroid']
        features['centroid_x'] = cx
        features['centroid_y'] = cy
        
        # Zone
        zone = "unknown"
        if zone_masks.get('core')[cy, cx] > 0:
            zone = "core"
        elif zone_masks.get('cladding')[cy, cx] > 0:
            zone = "cladding"
        elif zone_masks.get('ferrule')[cy, cx] > 0:
            zone = "ferrule"
        features['zone'] = zone
        
        # Shape descriptors
        rect = cv2.minAreaRect(defect["contour"])
        (x, y), (width, height), angle = rect
        features['rect_width'] = width
        features['rect_height'] = height
        features['rect_angle'] = angle
        features['aspect_ratio'] = max(width, height) / (min(width, height) + 1e-6)
        
        hull = cv2.convexHull(defect["contour"])
        hull_area = cv2.contourArea(hull)
        features['solidity'] = defect['area'] / (hull_area + 1e-6)
        
        # Intensity and Contrast
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [defect["contour"]], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]
        features['mean_intensity'] = mean_val
        
        # Texture (GLCM)
        x, y, w, h = cv2.boundingRect(defect["contour"])
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            glcm = greycomatrix(roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features['contrast'] = greycoprops(glcm, 'contrast')[0, 0]
            features['dissimilarity'] = greycoprops(glcm, 'dissimilarity')[0, 0]
            features['homogeneity'] = greycoprops(glcm, 'homogeneity')[0, 0]
            features['energy'] = greycoprops(glcm, 'energy')[0, 0]
            features['correlation'] = greycoprops(glcm, 'correlation')[0, 0]

        features_list.append(features)
        
    return features_list