#!/usr/bin/env python3
"""
Machine Learning Classifier Module
==================================
Standalone module for ML-based defect classification and anomaly detection.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, ML features disabled")

class DefectClassifier:
    """ML-based classifier for defect types."""
    
    def __init__(self, model_path: Optional[str] = None):
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, classifier disabled")
            self.enabled = False
            return
            
        self.enabled = True
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_names = [
            'area_px', 'aspect_ratio', 'solidity', 'eccentricity',
            'mean_intensity', 'std_intensity', 'perimeter_ratio',
            'hu_moment_1', 'hu_moment_2', 'orientation_deg'
        ]
        self.fitted = False
        
        if model_path:
            self.load_model(model_path)
    
    def extract_features(self, defect_dict: Dict, original_image: np.ndarray) -> np.ndarray:
        """Extract features for classification."""
        if not self.enabled:
            return np.array([])
            
        features = []
        
        # Basic geometric features
        features.append(defect_dict.get('area_px', 0))
        features.append(defect_dict.get('aspect_ratio', 1))
        
        # Calculate additional features
        contour_points = defect_dict.get('contour_points_px', [])
        if len(contour_points) > 5:
            contour_np = np.array(contour_points, dtype=np.int32)
            
            # Solidity
            hull = cv2.convexHull(contour_np)
            hull_area = cv2.contourArea(hull)
            solidity = defect_dict['area_px'] / hull_area if hull_area > 0 else 0
            features.append(solidity)
            
            # Eccentricity
            if len(contour_np) >= 5:
                ellipse = cv2.fitEllipse(contour_np)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
                features.append(eccentricity)
            else:
                features.append(0)
            
            # Intensity features
            mask = np.zeros(original_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour_np], -1, 255, -1)
            
            masked_pixels = original_image[mask > 0]
            if len(masked_pixels) > 0:
                features.append(np.mean(masked_pixels))
                features.append(np.std(masked_pixels))
            else:
                features.extend([0, 0])
            
            # Perimeter ratio
            perimeter = cv2.arcLength(contour_np, True)
            expected_perimeter = 2 * np.pi * np.sqrt(defect_dict['area_px'] / np.pi)
            perimeter_ratio = perimeter / expected_perimeter if expected_perimeter > 0 else 1
            features.append(perimeter_ratio)
            
            # Hu moments (shape descriptors)
            moments = cv2.moments(contour_np)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend([hu_moments[0], hu_moments[1]])
            
            # Orientation
            if len(contour_np) >= 5:
                ellipse = cv2.fitEllipse(contour_np)
                orientation = ellipse[2]
                features.append(orientation)
            else:
                features.append(0)
                
        else:
            # Default values if contour is too small
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        return np.array(features)
    
    def train(self, training_features: List[np.ndarray], labels: List[str]) -> bool:
        """Train the classifier."""
        if not self.enabled:
            return False
            
        try:
            X = np.array(training_features)
            y = np.array(labels)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train classifier
            self.classifier.fit(X_scaled, y)
            self.fitted = True
            
            logger.info(f"Classifier trained on {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def classify(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify a defect based on features."""
        if not self.enabled or not self.fitted:
            return "unknown", 0.0
            
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.classifier.predict(features_scaled)[0]
            confidence = np.max(self.classifier.predict_proba(features_scaled)[0])
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "unknown", 0.0
    
    def save_model(self, path: str) -> bool:
        """Save the trained model."""
        if not self.enabled or not self.fitted:
            return False
            
        try:
            model_data = {
                'classifier': self.classifier,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, path)
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a trained model."""
        if not self.enabled:
            return False
            
        try:
            model_data = joblib.load(path)
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.fitted = True
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

class AnomalyDetector:
    """Machine learning-based anomaly detector."""
    
    def __init__(self):
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, anomaly detection disabled")
            self.enabled = False
            return
            
        self.enabled = True
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.one_class_svm = OneClassSVM(gamma='scale', nu=0.1)
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
        self.fitted = False
    
    def extract_pixel_features(self, image: np.ndarray, patch_size: int = 5) -> np.ndarray:
        """Extract features for each pixel."""
        if not self.enabled:
            return np.array([])
            
        h, w = image.shape
        features = []
        
        # Pad image for edge handling
        padded = cv2.copyMakeBorder(image, patch_size//2, patch_size//2, 
                                   patch_size//2, patch_size//2, cv2.BORDER_REFLECT)
        
        for y in range(h):
            for x in range(w):
                # Extract patch
                patch = padded[y:y+patch_size, x:x+patch_size]
                
                # Calculate features
                pixel_features = [
                    image[y, x],  # Intensity
                    np.mean(patch),  # Local mean
                    np.std(patch),   # Local std
                    np.min(patch),   # Local min
                    np.max(patch),   # Local max
                ]
                
                # Add gradient features
                if y > 0 and y < h-1 and x > 0 and x < w-1:
                    grad_x = float(image[y, x+1]) - float(image[y, x-1])
                    grad_y = float(image[y+1, x]) - float(image[y-1, x])
                    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                    pixel_features.extend([grad_x, grad_y, grad_mag])
                else:
                    pixel_features.extend([0, 0, 0])
                
                features.append(pixel_features)
        
        return np.array(features)
    
    def detect_anomalies(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Detect anomalies in the image."""
        if not self.enabled:
            return np.zeros_like(image, dtype=np.uint8)
            
        logger.info("Running ML-based anomaly detection...")
        
        # Extract features
        features = self.extract_pixel_features(image)
        
        # Apply mask if provided
        if mask is not None:
            valid_pixels = mask.flatten() > 0
            valid_features = features[valid_pixels]
        else:
            valid_features = features
            valid_pixels = np.ones(len(features), dtype=bool)
        
        if len(valid_features) < 10:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Run different anomaly detection methods
        try:
            # 1. Isolation Forest
            iso_labels = self.isolation_forest.fit_predict(valid_features)
            
            # 2. One-class SVM
            svm_labels = self.one_class_svm.fit_predict(valid_features)
            
            # 3. DBSCAN clustering
            cluster_labels = self.dbscan.fit_predict(valid_features)
            
            # Create anomaly mask
            anomaly_pixels = np.zeros(len(features), dtype=bool)
            
            # Mark as anomaly if detected by any method
            for i, (iso, svm, cluster) in enumerate(zip(iso_labels, svm_labels, cluster_labels)):
                if iso == -1 or svm == -1 or cluster == -1:
                    valid_idx = np.where(valid_pixels)[0][i]
                    anomaly_pixels[valid_idx] = True
            
            # Reshape to image
            anomaly_map = anomaly_pixels.reshape(image.shape)
            
            # Convert to uint8
            result = (anomaly_map * 255).astype(np.uint8)
            
            logger.info(f"Anomaly detection found {np.sum(anomaly_map)} anomalous pixels")
            
            return result
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return np.zeros_like(image, dtype=np.uint8)

def simple_defect_classification(area: float, aspect_ratio: float, 
                               solidity: float, mean_intensity: float) -> str:
    """Simple rule-based classification when ML is not available."""
    
    # Classification rules
    if aspect_ratio > 3.0 and area > 20:
        return "scratch"
    elif solidity > 0.8 and area < 100:
        return "dig"
    elif area > 200:
        return "large_defect"
    elif mean_intensity < 0.3:
        return "dark_spot"
    elif mean_intensity > 0.8:
        return "bright_spot"
    else:
        return "small_defect"

def test_ml_classifier():
    """Test the ML classifier module."""
    logger.info("Testing ML classifier module...")
    
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available, testing simple classification only")
        
        # Test simple classification
        test_cases = [
            (50, 5.0, 0.9, 0.2),  # Should be scratch
            (25, 1.1, 0.9, 0.1),  # Should be dig
            (300, 1.5, 0.7, 0.5), # Should be large_defect
        ]
        
        for area, aspect_ratio, solidity, intensity in test_cases:
            classification = simple_defect_classification(area, aspect_ratio, solidity, intensity)
            logger.info(f"Area={area}, AR={aspect_ratio}, Sol={solidity}, Int={intensity} -> {classification}")
        
        return {"simple_classification": "completed"}
    
    # Test with scikit-learn
    classifier = DefectClassifier()
    anomaly_detector = AnomalyDetector()
    
    # Create synthetic training data
    np.random.seed(42)
    
    # Generate synthetic features for different defect types
    n_samples = 100
    
    # Scratches: high aspect ratio, low solidity
    scratch_features = []
    for _ in range(n_samples//4):
        features = [
            np.random.uniform(20, 100),    # area
            np.random.uniform(3, 10),      # aspect_ratio
            np.random.uniform(0.3, 0.7),   # solidity
            np.random.uniform(0.7, 0.9),   # eccentricity
            np.random.uniform(0.1, 0.4),   # mean_intensity
            np.random.uniform(0.05, 0.2),  # std_intensity
            np.random.uniform(1.2, 2.0),   # perimeter_ratio
            np.random.uniform(-0.1, 0.1),  # hu_moment_1
            np.random.uniform(-0.05, 0.05), # hu_moment_2
            np.random.uniform(0, 180)      # orientation
        ]
        scratch_features.append(np.array(features))
    
    # Digs: circular, high solidity
    dig_features = []
    for _ in range(n_samples//4):
        features = [
            np.random.uniform(10, 50),     # area
            np.random.uniform(0.8, 1.5),   # aspect_ratio
            np.random.uniform(0.8, 0.95),  # solidity
            np.random.uniform(0.1, 0.3),   # eccentricity
            np.random.uniform(0.05, 0.3),  # mean_intensity
            np.random.uniform(0.02, 0.1),  # std_intensity
            np.random.uniform(0.9, 1.2),   # perimeter_ratio
            np.random.uniform(-0.05, 0.05), # hu_moment_1
            np.random.uniform(-0.02, 0.02), # hu_moment_2
            np.random.uniform(0, 180)      # orientation
        ]
        dig_features.append(np.array(features))
    
    # Combine training data
    all_features = scratch_features + dig_features
    all_labels = ['scratch'] * len(scratch_features) + ['dig'] * len(dig_features)
    
    # Train classifier
    success = classifier.train(all_features, all_labels)
    logger.info(f"Classifier training success: {success}")
    
    # Test classification
    if success:
        # Test scratch
        test_scratch = np.array([50, 5.0, 0.5, 0.8, 0.2, 0.1, 1.5, 0.0, 0.0, 45])
        pred, conf = classifier.classify(test_scratch)
        logger.info(f"Test scratch prediction: {pred} (confidence: {conf:.3f})")
        
        # Test dig
        test_dig = np.array([30, 1.2, 0.9, 0.2, 0.15, 0.05, 1.0, 0.0, 0.0, 90])
        pred, conf = classifier.classify(test_dig)
        logger.info(f"Test dig prediction: {pred} (confidence: {conf:.3f})")
    
    # Test anomaly detection
    test_image = np.random.rand(64, 64).astype(np.float32)
    
    # Add some anomalies
    test_image[20:25, 20:25] = 0.1  # Dark anomaly
    test_image[40:45, 40:45] = 0.9  # Bright anomaly
    
    anomaly_map = anomaly_detector.detect_anomalies(test_image)
    anomaly_count = np.sum(anomaly_map > 0)
    logger.info(f"Anomaly detection found {anomaly_count} anomalous pixels")
    
    logger.info("ML classifier testing completed!")
    
    return {
        'classifier_trained': success,
        'anomaly_detection_completed': True,
        'test_image': test_image,
        'anomaly_map': anomaly_map
    }

if __name__ == "__main__":
    # Run tests
    test_results = test_ml_classifier()
    logger.info("ML classifier module is ready for use!")
