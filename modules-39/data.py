
import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict

from .base_module import BaseModule

# ---------------------------------------------------------------------------- #
#                            Feature Extraction Module                         #
# ---------------------------------------------------------------------------- #

class FeatureExtractionModule(BaseModule):

    def __init__(self, config_manager):
        super().__init__(config_manager)
        self._register_tunable_parameter("glcm_distances", [1])
        self._register_tunable_parameter("glcm_angles", [0])
        self._register_tunable_parameter("glcm_levels", 256)

    def execute(self, image_bgr: np.ndarray, defect_contours: List[np.ndarray], zone_masks: Dict[str, np.ndarray] = None) -> List[Dict]:
        """
        Extracts features from a list of defect contours.
        """
        self.logger.info(f"Starting feature extraction for {len(defect_contours)} contours.")
        features_list = []
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        for i, contour in enumerate(defect_contours):
            features = {}
            moments = cv2.moments(contour)
            
            # Basic properties
            area = moments['m00']
            if area == 0:
                continue
            features['defect_id'] = i
            features['area_px'] = area
            
            # Location
            cx = int(moments['m10'] / area)
            cy = int(moments['m01'] / area)
            features['centroid_x'] = cx
            features['centroid_y'] = cy
            
            # Zone (if masks are provided)
            if zone_masks:
                zone = "unknown"
                if zone_masks.get('core')[cy, cx] > 0: zone = "core"
                elif zone_masks.get('cladding')[cy, cx] > 0: zone = "cladding"
                elif zone_masks.get('ferrule')[cy, cx] > 0: zone = "ferrule"
                features['zone'] = zone
            
            # Shape descriptors
            x, y, w, h = cv2.boundingRect(contour)
            features['aspect_ratio'] = max(w, h) / (min(w, h) + 1e-6)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = area / (hull_area + 1e-6)
            
            # Intensity and Contrast
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]
            features['mean_intensity'] = mean_val
            
            # Texture (GLCM)
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
                params = self.get_tunable_parameters()
                glcm = graycomatrix(roi, 
                                    distances=params['glcm_distances'], 
                                    angles=params['glcm_angles'], 
                                    levels=params['glcm_levels'], 
                                    symmetric=True, normed=True)
                
                features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
                features['dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
                features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
                features['energy'] = graycoprops(glcm, 'energy')[0, 0]
                features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]

            features_list.append(features)
        
        self.logger.info(f"Feature extraction complete. Extracted {len(features_list)} feature sets.")
        return features_list

# ---------------------------------------------------------------------------- #
#                             Data Clustering Module                           #
# ---------------------------------------------------------------------------- #

class DataClusteringModule(BaseModule):

    def __init__(self, config_manager):
        super().__init__(config_manager)
        self._register_tunable_parameter("n_clusters", 4)
        self._register_tunable_parameter("random_state", 42)
        self._register_tunable_parameter("features_to_use", [
            'area_px', 'aspect_ratio', 'solidity', 'mean_intensity',
            'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'
        ])

    def execute(self, features_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Starting clustering on {len(features_df)} data points.")
        if features_df.empty:
            self.logger.warning("Input DataFrame is empty. Skipping clustering.")
            return features_df
        
        params = self.get_tunable_parameters()
        features_for_clustering = [col for col in params['features_to_use'] if col in features_df.columns]
        
        missing_cols = set(params['features_to_use']) - set(features_for_clustering)
        if missing_cols:
            self.logger.warning(f"Missing columns for clustering, they will be ignored: {missing_cols}")

        if not features_for_clustering:
            self.logger.error("No valid feature columns found for clustering. Aborting.")
            return features_df

        # Handle missing values and scale
        data_to_cluster = features_df[features_for_clustering].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data_to_cluster)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=params['n_clusters'], 
                        random_state=params['random_state'], 
                        n_init=10)
        
        features_df['cluster'] = kmeans.fit_predict(scaled_features)
        self.logger.info(f"Clustering complete. Assigned data to {params['n_clusters']} clusters.")
        
        return features_df
