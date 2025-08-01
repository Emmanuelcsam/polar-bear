#!/usr/bin/env python3
"""
Live Defect Detector
This script merges real-time camera capture with advanced anomaly detection
to find defects on a surface in real-time. It uses a reference image
to build a model of a "good" surface and then compares the live feed against it.
"""

import json
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import logging
import time
import warnings

# Suppress potential warnings from libraries
warnings.filterwarnings('ignore')

# --- Pylon Camera Support ---
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    pass

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

# --- Data Classes and Encoders from detection.py ---

@dataclass
class OmniConfig:
    """Configuration for OmniFiberAnalyzer"""
    knowledge_base_path: Optional[str] = None
    min_defect_size: int = 10
    max_defect_size: int = 5000
    severity_thresholds: Optional[Dict[str, float]] = None
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    enable_visualization: bool = True
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'CRITICAL': 0.9, 'HIGH': 0.7, 'MEDIUM': 0.5,
                'LOW': 0.3, 'NEGLIGIBLE': 0.1
            }

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- Core Analysis Engine (from detection.py) ---

class OmniFiberAnalyzer:
    """The ultimate fiber optic anomaly detection system - adapted for live analysis."""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
        self.reference_model = {}
        self.current_metadata = None
        self.logger = logging.getLogger(__name__)
        if os.path.exists(self.knowledge_base_path):
            self.load_knowledge_base()

    def build_reference_from_image(self, image_path: str):
        """Builds a minimal reference model from a single image file."""
        if not os.path.exists(image_path):
            self.logger.error(f"Reference image not found at {image_path}")
            self.logger.warning("Creating a dummy black image as a fallback reference.")
            dummy_image = np.zeros((480, 640), dtype=np.uint8)
            dummy_path = "dummy_reference.png"
            cv2.imwrite(dummy_path, dummy_image)
            self._build_minimal_reference(dummy_path)
            os.remove(dummy_path)
        else:
            self._build_minimal_reference(image_path)

    def analyze_frame(self, test_image: np.ndarray):
        """Perform exhaustive anomaly detection on a test image (in-memory)."""
        if not self.reference_model.get('statistical_model'):
            self.logger.warning("No reference model available. Build one first.")
            return None

        if test_image is None:
            return None

        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()

        test_features, _ = self.extract_ultra_comprehensive_features(test_image)
        
        # --- Global Analysis ---
        stat_model = self.reference_model['statistical_model']
        feature_names = self.reference_model['feature_names']
        test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
        diff = test_vector - stat_model['robust_mean']
        try:
            mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
        except:
            std_vector = stat_model['std']
            std_vector[std_vector < 1e-6] = 1.0
            normalized_diff = diff / std_vector
            mahalanobis_dist = np.linalg.norm(normalized_diff)

        # --- Structural Analysis ---
        archetype = self.reference_model['archetype_image']
        if isinstance(archetype, list):
            archetype = np.array(archetype, dtype=np.uint8)
        
        if test_gray.shape != archetype.shape:
            test_gray_resized = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
        else:
            test_gray_resized = test_gray
        
        structural_comp = self.compute_image_structural_comparison(test_gray_resized, archetype)
        
        # --- Local Anomaly Detection ---
        anomaly_map = self._compute_local_anomaly_map(test_gray_resized, archetype)
        anomaly_regions = self._find_anomaly_regions(anomaly_map, test_gray.shape)
        
        # --- Specific Defect Detection ---
        specific_defects = self._detect_specific_defects(test_gray)
        
        # --- Determine Overall Status ---
        thresholds = self.reference_model['learned_thresholds']
        is_anomalous = (mahalanobis_dist > thresholds.get('anomaly_threshold', 3.0) or 
                        structural_comp['ssim'] < 0.7 or 
                        len(anomaly_regions) > 3)
        
        confidence = min(1.0, max(
            mahalanobis_dist / thresholds.get('anomaly_threshold', 3.0),
            1 - structural_comp['ssim'],
            len(anomaly_regions) / 10.0
        ))

        return {
            'verdict': {'is_anomalous': is_anomalous, 'confidence': float(confidence)},
            'local_analysis': {'anomaly_regions': anomaly_regions},
            'specific_defects': specific_defects,
        }

    # Most of OmniFiberAnalyzer methods are copied here directly
    # For brevity, only the new/modified methods are shown in full.
    # All helper methods from detection.py are assumed to be here.
    def _build_minimal_reference(self, image_path: str):
        """Build a minimal reference model from a single image"""
        self.logger.info("Building minimal reference model from current image...")
        image = self.load_image(image_path)
        if image is None: return
        features, feature_names = self.extract_ultra_comprehensive_features(image)
        feature_vector = np.array([features[fname] for fname in feature_names])
        self.reference_model = {
            'features': [features], 'feature_names': feature_names,
            'statistical_model': {
                'mean': feature_vector, 'std': np.ones_like(feature_vector) * 0.1,
                'median': feature_vector, 'robust_mean': feature_vector,
                'robust_cov': np.eye(len(feature_vector)),
                'robust_inv_cov': np.eye(len(feature_vector)), 'n_samples': 1,
            },
            'archetype_image': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image,
            'learned_thresholds': {
                'anomaly_mean': 1.0, 'anomaly_std': 0.5, 'anomaly_p90': 1.5,
                'anomaly_p95': 2.0, 'anomaly_p99': 3.0,
                'anomaly_threshold': self.config.anomaly_threshold_multiplier,
            }, 'timestamp': self._get_timestamp(),
        }
    def load_image(self, path):
        self.current_metadata = None
        if path.lower().endswith('.json'): return self._load_from_json(path)
        else:
            img = cv2.imread(path)
            if img is None:
                self.logger.error(f"Could not read image: {path}"); return None
            self.current_metadata = {'filename': os.path.basename(path)}
            return img
    def _load_from_json(self, json_path): return None # Simplified for this app
    def extract_ultra_comprehensive_features(self, image):
        features = {}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # In a real implementation, all _extract_* methods would be here.
        # For this merged script, we'll just use a few key ones for performance.
        features.update(self._extract_statistical_features(gray))
        features.update(self._extract_gradient_features(gray))
        features.update(self._extract_morphological_features(gray))
        sanitized_features = {k: self._sanitize_feature_value(v) for k, v in features.items()}
        return sanitized_features, sorted(sanitized_features.keys())
    def _sanitize_feature_value(self, value):
        if isinstance(value, (list, tuple, np.ndarray)): return float(value[0]) if len(value) > 0 else 0.0
        val = float(value)
        return 0.0 if np.isnan(val) or np.isinf(val) else val
    def _extract_statistical_features(self, gray):
        return {'stat_mean': float(np.mean(gray)), 'stat_std': float(np.std(gray)), 'stat_max': float(np.max(gray)), 'stat_min': float(np.min(gray))}
    def _extract_gradient_features(self, gray):
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return {'gradient_magnitude_mean': float(np.mean(grad_mag)), 'gradient_magnitude_std': float(np.std(grad_mag))}
    def _extract_morphological_features(self, gray):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        return {'morph_bth_5_mean': float(np.mean(bth)), 'morph_bth_5_max': float(np.max(bth))}
    def compute_image_structural_comparison(self, img1, img2):
        from skimage.metrics import structural_similarity as ssim
        score = ssim(img1, img2, data_range=img1.max() - img1.min())
        return {'ssim': score, 'ssim_map': np.zeros_like(img1)} # ssim_map simplified
    def _compute_local_anomaly_map(self, test_img, reference_img):
        diff = cv2.absdiff(test_img, reference_img)
        return cv2.GaussianBlur(diff, (15, 15), 0)
    def _find_anomaly_regions(self, anomaly_map, original_shape):
        if np.max(anomaly_map) == 0: return []
        threshold = np.percentile(anomaly_map[anomaly_map > 0], 80)
        binary_map = (anomaly_map > threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        regions = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > self.config.min_defect_size:
                region_mask = (labels == i)
                confidence = float(np.mean(anomaly_map[region_mask]))
                regions.append({'bbox': (x, y, w, h), 'area': int(area), 'confidence': confidence / 255.0})
        return sorted(regions, key=lambda r: r['confidence'], reverse=True)
    def _detect_specific_defects(self, gray):
        defects = {'scratches': [], 'digs': [], 'blobs': []}
        # Simplified for live performance
        # Digs (dark spots)
        bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        _, thresh = cv2.threshold(bth, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 500:
                x, y, w, h = cv2.boundingRect(cnt)
                defects['digs'].append({'bbox': (x, y, w, h), 'area': area})
        return defects
    def _get_timestamp(self): return time.strftime("%Y-%m-%d_%H:%M:%S")
    def load_knowledge_base(self): pass # Simplified
    def save_knowledge_base(self): pass # Simplified

# --- Camera Handling (from unified_core_detector.py) ---

class PylonCamera:
    """Enhanced camera interface with GPU optimization"""
    def __init__(self, camera_index: int = 0, use_pylon: bool = True):
        self.camera_index = camera_index
        self.use_pylon = use_pylon and PYLON_AVAILABLE
        self.camera = None
        self.is_grabbing = False
        self.setup_camera()
        
    def setup_camera(self):
        pylon_error = None
        if self.use_pylon:
            try:
                tl_factory = pylon.TlFactory.GetInstance()
                devices = tl_factory.EnumerateDevices()
                if not devices:
                    print("No Pylon cameras found. Attempting to use webcam fallback.")
                    self.use_pylon = False
                else:
                    self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
                    self.camera.Open()
                    # Removed rigid PixelFormat setting. Let the camera use its default.
                    self.camera.ExposureAuto.SetValue("Continuous")
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    self.is_grabbing = True
                    print(f"Pylon camera initialized: {self.camera.GetDeviceInfo().GetModelName()}")
                    return # Successfully initialized Pylon camera
            except Exception as e:
                pylon_error = e
                print(f"Error setting up Pylon camera: {e}")
                self.use_pylon = False # Force fallback on Pylon error

        # Fallback to webcam if Pylon is not used or failed
        print("Attempting to initialize webcam...")
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            if pylon_error:
                raise RuntimeError(f"Pylon camera failed ({pylon_error}) and could not open fallback webcam.")
            else:
                raise RuntimeError(f"Failed to open webcam at index {self.camera_index}.")
        print(f"Successfully initialized webcam at index {self.camera_index}")
    
    def read_frame(self) -> Optional[np.ndarray]:
        try:
            if self.use_pylon and self.is_grabbing:
                grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
                if grab_result.GrabSucceeded():
                    image = grab_result.Array; grab_result.Release()
                    return image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                return None
            else:
                ret, frame = self.camera.read()
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
        except Exception as e:
            print(f"Error reading frame: {e}"); return None
    
    def release(self):
        if self.camera:
            try:
                if self.use_pylon and self.is_grabbing: self.camera.StopGrabbing()
                elif hasattr(self.camera, "release"): self.camera.release()
            except Exception as e: print(f"Error releasing camera: {e}")

# --- Main Application Class ---

class LiveDefectDetector:
    """Main application to run the live detection loop."""
    def __init__(self):
        self.camera = PylonCamera(camera_index=0, use_pylon=True)
        config = OmniConfig()
        config.confidence_threshold = 0.2
        config.anomaly_threshold_multiplier = 2.0
        self.analyzer = OmniFiberAnalyzer(config)
        
        self.is_running = False
        self.window_name = "Live Defect Detector"

        print("Building reference model from 'good.bmp'...")
        self.analyzer.build_reference_from_image('good.bmp')
        print("Reference model built.")

    def run(self):
        """Main application loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.is_running = True
        
        while self.is_running:
            frame_rgb = self.camera.read_frame()
            if frame_rgb is None:
                time.sleep(0.01)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            results = self.analyzer.analyze_frame(frame_bgr)
            
            display_frame = self.draw_results(frame_bgr.copy(), results)

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC key
                self.is_running = False
            
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.is_running = False
            except cv2.error: # Catches error when window is closed
                self.is_running = False
        
        self.cleanup()

    def draw_results(self, frame, results: Optional[Dict]) -> np.ndarray:
        """Draws detection results on the frame."""
        if results is None:
            cv2.putText(frame, "ANALYSIS FAILED OR NO MODEL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        verdict = results['verdict']
        color = (0, 255, 0) if not verdict['is_anomalous'] else (0, 0, 255)
        text = f"Status: {'OK' if not verdict['is_anomalous'] else 'ANOMALY'} (Conf: {verdict['confidence']:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        all_defects = self._get_defects_from_results(results)
        for defect in all_defects:
            x, y, w, h = defect['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            label = f"{defect['defect_type']}"
            if 'confidence' in defect:
                label += f" ({defect['confidence']:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        return frame

    def _get_defects_from_results(self, results: Dict) -> List[Dict]:
        """Extracts a simple list of defects from the complex results dictionary."""
        defects = []
        if 'local_analysis' in results and 'anomaly_regions' in results['local_analysis']:
            for region in results['local_analysis']['anomaly_regions']:
                defects.append({'defect_type': 'ANOMALY', 'bbox': region['bbox'], 'confidence': region['confidence']})
        
        if 'specific_defects' in results:
            if 'digs' in results['specific_defects']:
                for dig in results['specific_defects']['digs']:
                    defects.append({'defect_type': 'DIG', 'bbox': dig['bbox'], 'area': dig['area']})
            if 'scratches' in results['specific_defects']:
                 for scratch in results['specific_defects']['scratches']:
                    x1, y1, x2, y2 = scratch['line']
                    bbox = [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)]
                    defects.append({'defect_type': 'SCRATCH', 'bbox': bbox})
        return defects

    def cleanup(self):
        """Cleanup resources."""
        self.camera.release()
        cv2.destroyAllWindows()
        print("Application stopped.")

# --- Main Execution Block ---

if __name__ == "__main__":
    # This script requires 'scikit-image'. Let's check for it.
    try:
        import skimage
    except ImportError:
        print("="*50)
        print("ERROR: 'scikit-image' is not installed.")
        print("Please install it by running: pip install scikit-image")
        print("="*50)
        exit(1)

    app = LiveDefectDetector()
    app.run()