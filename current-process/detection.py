"""
Enhanced Detection Module
- ML-powered defect detection
- Multiple detection algorithms
- In-memory processing
- No argparse
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import concurrent.futures
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from scipy import stats, ndimage
import json

from config_manager import get_config
from enhanced_logging import get_logger, log_execution, log_performance, ProgressLogger

logger = get_logger(__name__)


@dataclass
class Defect:
    """Represents a detected defect"""
    type: str
    location: Tuple[int, int]  # (x, y) center
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    severity: float  # 0-1
    confidence: float  # 0-1
    zone: str  # core, cladding, ferrule
    area: int
    properties: Dict[str, Any] = field(default_factory=dict)
    detection_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type,
            'location': self.location,
            'bbox': self.bbox,
            'severity': self.severity,
            'confidence': self.confidence,
            'zone': self.zone,
            'area': self.area,
            'properties': self.properties,
            'detection_method': self.detection_method
        }


class MLDefectDetector:
    """ML-powered defect detection"""
    
    def __init__(self):
        self.config = get_config()
        self.detector_model = None
        self.anomaly_model = None
        self._load_models()
    
    def _load_models(self):
        """Load ML models for defect detection"""
        if not self.config.processing.ml_enabled:
            return
        
        try:
            if self.config.processing.pytorch_enabled:
                self._load_pytorch_models()
            elif self.config.processing.tensorflow_enabled:
                self._load_tensorflow_models()
        except Exception as e:
            logger.warning(f"Failed to load ML detection models: {e}")
    
    def _load_pytorch_models(self):
        """Load PyTorch models"""
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            
            # Object detection model (simplified Faster R-CNN style)
            class DefectDetector(nn.Module):
                def __init__(self, num_classes=5):  # scratch, pit, contamination, fiber_damage, other
                    super().__init__()
                    # Use ResNet18 as backbone
                    self.backbone = models.resnet18(pretrained=False)
                    self.backbone.fc = nn.Identity()  # Remove final FC
                    
                    # Detection heads
                    self.box_predictor = nn.Linear(512, 4)  # bbox regression
                    self.class_predictor = nn.Linear(512, num_classes)
                    self.confidence_predictor = nn.Linear(512, 1)
                    
                def forward(self, x):
                    features = self.backbone(x)
                    boxes = self.box_predictor(features)
                    classes = self.class_predictor(features)
                    confidence = torch.sigmoid(self.confidence_predictor(features))
                    return boxes, classes, confidence
            
            # Anomaly detection model (Autoencoder)
            class AnomalyDetector(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Encoder
                    self.encoder = nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.ReLU(),
                    )
                    # Decoder
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
                        nn.Sigmoid(),
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded
            
            # Load models
            self.detector_model = DefectDetector()
            self.anomaly_model = AnomalyDetector()
            
            # Try to load weights
            detector_path = self.config.model_dir / "defect_detector.pth"
            if detector_path.exists():
                self.detector_model.load_state_dict(torch.load(detector_path, map_location='cpu'))
                self.detector_model.eval()
                logger.info("Loaded PyTorch defect detector")
            
            anomaly_path = self.config.model_dir / "anomaly_detector.pth"
            if anomaly_path.exists():
                self.anomaly_model.load_state_dict(torch.load(anomaly_path, map_location='cpu'))
                self.anomaly_model.eval()
                logger.info("Loaded PyTorch anomaly detector")
                
        except ImportError:
            logger.warning("PyTorch not available for detection")
    
    def _load_tensorflow_models(self):
        """Load TensorFlow models"""
        try:
            import tensorflow as tf
            
            # Try to load pre-trained models
            detector_path = self.config.model_dir / "defect_detector.h5"
            if detector_path.exists():
                self.detector_model = tf.keras.models.load_model(detector_path)
                logger.info("Loaded TensorFlow defect detector")
            
            anomaly_path = self.config.model_dir / "anomaly_detector.h5"
            if anomaly_path.exists():
                self.anomaly_model = tf.keras.models.load_model(anomaly_path)
                logger.info("Loaded TensorFlow anomaly detector")
                
        except ImportError:
            logger.warning("TensorFlow not available for detection")
    
    def detect_defects(self, image: np.ndarray, zone_mask: np.ndarray, 
                      zone_name: str) -> List[Defect]:
        """Detect defects using ML models"""
        defects = []
        
        # Object detection
        if self.detector_model is not None:
            detected = self._run_object_detection(image, zone_mask, zone_name)
            defects.extend(detected)
        
        # Anomaly detection
        if self.anomaly_model is not None:
            anomalies = self._run_anomaly_detection(image, zone_mask, zone_name)
            defects.extend(anomalies)
        
        return defects
    
    def _run_object_detection(self, image: np.ndarray, zone_mask: np.ndarray, 
                             zone_name: str) -> List[Defect]:
        """Run object detection model"""
        try:
            # Apply zone mask
            masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
            
            # Prepare image
            h, w = image.shape[:2]
            input_size = (224, 224)
            resized = cv2.resize(masked_image, input_size)
            
            if self.config.processing.pytorch_enabled:
                import torch
                x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
                with torch.no_grad():
                    boxes, classes, confidence = self.detector_model(x)
                    
                    # Process predictions
                    boxes = boxes.squeeze().numpy()
                    classes = torch.softmax(classes, dim=1).squeeze().numpy()
                    confidence = confidence.squeeze().item()
                    
                    if confidence > self.config.detection.confidence_threshold:
                        # Convert box coordinates
                        x1, y1, x2, y2 = boxes
                        x1 = int(x1 * w / input_size[0])
                        y1 = int(y1 * h / input_size[1])
                        x2 = int(x2 * w / input_size[0])
                        y2 = int(y2 * h / input_size[1])
                        
                        # Get defect type
                        class_names = ['scratch', 'pit', 'contamination', 'fiber_damage', 'other']
                        defect_type = class_names[np.argmax(classes)]
                        
                        defect = Defect(
                            type=defect_type,
                            location=((x1 + x2) // 2, (y1 + y2) // 2),
                            bbox=(x1, y1, x2 - x1, y2 - y1),
                            severity=0.5,  # TODO: Calculate based on size/type
                            confidence=float(confidence),
                            zone=zone_name,
                            area=(x2 - x1) * (y2 - y1),
                            detection_method="ml_object_detection"
                        )
                        
                        return [defect]
            
            else:  # TensorFlow
                x = resized[np.newaxis, ...] / 255.0
                predictions = self.detector_model.predict(x, verbose=0)
                # Process TensorFlow predictions (implementation depends on model architecture)
                
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
        
        return []
    
    def _run_anomaly_detection(self, image: np.ndarray, zone_mask: np.ndarray, 
                              zone_name: str) -> List[Defect]:
        """Run anomaly detection using autoencoder"""
        try:
            # Apply zone mask
            masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
            
            # Sliding window approach
            window_size = 64
            stride = 32
            h, w = image.shape[:2]
            
            anomaly_map = np.zeros((h, w), dtype=np.float32)
            
            for y in range(0, h - window_size, stride):
                for x in range(0, w - window_size, stride):
                    # Extract window
                    window = masked_image[y:y+window_size, x:x+window_size]
                    
                    # Check if window is in zone
                    if zone_mask[y:y+window_size, x:x+window_size].mean() < 128:
                        continue
                    
                    # Compute reconstruction error
                    if self.config.processing.pytorch_enabled:
                        import torch
                        x_in = torch.from_numpy(window).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        
                        with torch.no_grad():
                            reconstruction = self.anomaly_model(x_in)
                            error = torch.mean((x_in - reconstruction) ** 2).item()
                    else:  # TensorFlow
                        x_in = window[np.newaxis, ...] / 255.0
                        reconstruction = self.anomaly_model.predict(x_in, verbose=0)
                        error = np.mean((x_in - reconstruction) ** 2)
                    
                    # Update anomaly map
                    anomaly_map[y:y+window_size, x:x+window_size] = np.maximum(
                        anomaly_map[y:y+window_size, x:x+window_size],
                        error
                    )
            
            # Threshold anomaly map
            threshold = np.percentile(anomaly_map[anomaly_map > 0], 95)
            anomaly_binary = (anomaly_map > threshold).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(anomaly_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            defects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.config.detection.min_defect_size:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                defect = Defect(
                    type="anomaly",
                    location=(x + w // 2, y + h // 2),
                    bbox=(x, y, w, h),
                    severity=min(1.0, area / 1000),  # Normalize by reference area
                    confidence=0.7,  # Fixed confidence for anomalies
                    zone=zone_name,
                    area=int(area),
                    detection_method="ml_anomaly_detection"
                )
                defects.append(defect)
            
            return defects
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []


class TraditionalDetector:
    """Traditional computer vision defect detection"""
    
    def __init__(self):
        self.config = get_config()
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load reference models and statistics"""
        kb_path = self.config.knowledge_base_path
        
        if kb_path.exists():
            try:
                with open(kb_path, 'r') as f:
                    data = json.load(f)
                    return data.get('detection_models', {})
            except Exception as e:
                logger.warning(f"Failed to load detection knowledge base: {e}")
        
        return {}
    
    def detect_scratches(self, image: np.ndarray, zone_mask: np.ndarray, 
                        zone_name: str) -> List[Defect]:
        """Detect scratch-type defects"""
        defects = []
        
        # Apply zone mask
        masked = cv2.bitwise_and(image, image, mask=zone_mask)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        
        # Line detection using multiple orientations
        angles = np.linspace(0, 180, 12, endpoint=False)
        line_responses = []
        
        for angle in angles:
            # Create oriented kernel
            kernel_size = 15
            kernel = np.zeros((kernel_size, kernel_size))
            cv2.line(kernel, (0, kernel_size//2), (kernel_size-1, kernel_size//2), 1, 1)
            
            # Rotate kernel
            M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            kernel = kernel / kernel.sum()
            
            # Convolve
            response = cv2.filter2D(gray, -1, kernel)
            line_responses.append(response)
        
        # Maximum response across orientations
        max_response = np.max(line_responses, axis=0)
        
        # Threshold
        _, binary = cv2.threshold(max_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.detection.min_defect_size:
                continue
            
            # Check aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
            
            if aspect_ratio > 3:  # Likely a scratch
                # Calculate severity based on length
                length = max(w, h)
                severity = min(1.0, length / 100)
                
                defect = Defect(
                    type="scratch",
                    location=(x + w // 2, y + h // 2),
                    bbox=(x, y, w, h),
                    severity=severity,
                    confidence=0.8,
                    zone=zone_name,
                    area=int(area),
                    properties={
                        'length': length,
                        'aspect_ratio': aspect_ratio
                    },
                    detection_method="traditional_scratch_detection"
                )
                defects.append(defect)
        
        return defects
    
    def detect_pits(self, image: np.ndarray, zone_mask: np.ndarray, 
                    zone_name: str) -> List[Defect]:
        """Detect pit/dig type defects"""
        defects = []
        
        # Apply zone mask
        masked = cv2.bitwise_and(image, image, mask=zone_mask)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        
        # Blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.config.detection.min_defect_size
        params.maxArea = self.config.detection.max_defect_size
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.8
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(cv2.bitwise_not(gray))  # Invert for dark spots
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            
            defect = Defect(
                type="pit",
                location=(x, y),
                bbox=(x - radius, y - radius, radius * 2, radius * 2),
                severity=min(1.0, radius / 20),
                confidence=0.7,
                zone=zone_name,
                area=int(np.pi * radius * radius),
                properties={
                    'radius': radius,
                    'circularity': 0.9  # Approximate
                },
                detection_method="traditional_pit_detection"
            )
            defects.append(defect)
        
        return defects
    
    def detect_contamination(self, image: np.ndarray, zone_mask: np.ndarray, 
                            zone_name: str) -> List[Defect]:
        """Detect contamination/particles"""
        defects = []
        
        # Apply zone mask
        masked = cv2.bitwise_and(image, image, mask=zone_mask)
        
        # Color-based detection
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common contaminants
        # This is simplified - in practice, would use learned color models
        lower_bound = np.array([0, 50, 50])
        upper_bound = np.array([180, 255, 255])
        
        # Create mask for colored regions
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.detection.min_defect_size:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract region for color analysis
            roi = masked[y:y+h, x:x+w]
            mean_color = cv2.mean(roi)[:3]
            
            defect = Defect(
                type="contamination",
                location=(x + w // 2, y + h // 2),
                bbox=(x, y, w, h),
                severity=min(1.0, area / 500),
                confidence=0.6,
                zone=zone_name,
                area=int(area),
                properties={
                    'color': mean_color,
                    'compactness': 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                },
                detection_method="traditional_contamination_detection"
            )
            defects.append(defect)
        
        return defects
    
    def detect_statistical_anomalies(self, image: np.ndarray, zone_mask: np.ndarray, 
                                   zone_name: str) -> List[Defect]:
        """Detect anomalies using statistical methods"""
        defects = []
        
        # Apply zone mask
        masked = cv2.bitwise_and(image, image, mask=zone_mask)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        
        # Get reference statistics from knowledge base
        ref_stats = self.knowledge_base.get(f"{zone_name}_statistics", {})
        
        if not ref_stats:
            # Build reference from current image (fallback)
            pixels = gray[zone_mask > 0]
            ref_stats = {
                'mean': float(np.mean(pixels)),
                'std': float(np.std(pixels))
            }
        
        # Mahalanobis distance for each pixel
        mean = ref_stats['mean']
        std = ref_stats['std'] + 1e-6  # Avoid division by zero
        
        # Compute z-scores
        z_scores = np.abs((gray.astype(np.float32) - mean) / std)
        z_scores[zone_mask == 0] = 0  # Mask out non-zone areas
        
        # Threshold based on Mahalanobis distance
        anomaly_mask = (z_scores > self.config.detection.anomaly_threshold).astype(np.uint8) * 255
        
        # Clean up small regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.detection.min_defect_size:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate severity based on deviation
            roi_scores = z_scores[y:y+h, x:x+w]
            max_deviation = np.max(roi_scores)
            severity = min(1.0, max_deviation / 10)
            
            defect = Defect(
                type="statistical_anomaly",
                location=(x + w // 2, y + h // 2),
                bbox=(x, y, w, h),
                severity=severity,
                confidence=0.7,
                zone=zone_name,
                area=int(area),
                properties={
                    'max_z_score': float(max_deviation),
                    'mean_z_score': float(np.mean(roi_scores))
                },
                detection_method="statistical_anomaly_detection"
            )
            defects.append(defect)
        
        return defects


class DefectMerger:
    """Merges and clusters detected defects"""
    
    def __init__(self):
        self.config = get_config()
    
    def merge_defects(self, all_defects: List[Defect]) -> List[Defect]:
        """Merge overlapping and nearby defects"""
        if not all_defects:
            return []
        
        # Group by zone
        zone_defects = defaultdict(list)
        for defect in all_defects:
            zone_defects[defect.zone].append(defect)
        
        merged_defects = []
        
        for zone, defects in zone_defects.items():
            # Extract centers for clustering
            centers = np.array([defect.location for defect in defects])
            
            # DBSCAN clustering
            clustering = DBSCAN(
                eps=self.config.detection.cluster_eps,
                min_samples=self.config.detection.cluster_min_samples
            ).fit(centers)
            
            # Merge defects in same cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(clustering.labels_):
                if label >= 0:  # Not noise
                    clusters[label].append(defects[idx])
                else:
                    # Keep noise points as individual defects
                    merged_defects.append(defects[idx])
            
            # Merge each cluster
            for cluster_defects in clusters.values():
                merged = self._merge_cluster(cluster_defects)
                merged_defects.append(merged)
        
        return merged_defects
    
    def _merge_cluster(self, defects: List[Defect]) -> Defect:
        """Merge a cluster of defects into one"""
        # Determine type (majority vote)
        type_counts = defaultdict(int)
        for d in defects:
            type_counts[d.type] += 1
        defect_type = max(type_counts, key=type_counts.get)
        
        # Merge bounding boxes
        x_min = min(d.bbox[0] for d in defects)
        y_min = min(d.bbox[1] for d in defects)
        x_max = max(d.bbox[0] + d.bbox[2] for d in defects)
        y_max = max(d.bbox[1] + d.bbox[3] for d in defects)
        
        # Average properties
        avg_severity = np.mean([d.severity for d in defects])
        avg_confidence = np.mean([d.confidence for d in defects])
        total_area = sum(d.area for d in defects)
        
        # Merge properties
        merged_props = {}
        for d in defects:
            for k, v in d.properties.items():
                if k not in merged_props:
                    merged_props[k] = []
                merged_props[k].append(v)
        
        # Average numeric properties
        for k, v in merged_props.items():
            if all(isinstance(x, (int, float)) for x in v):
                merged_props[k] = np.mean(v)
        
        # Detection methods
        methods = list(set(d.detection_method for d in defects))
        
        return Defect(
            type=defect_type,
            location=((x_min + x_max) // 2, (y_min + y_max) // 2),
            bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
            severity=avg_severity,
            confidence=avg_confidence,
            zone=defects[0].zone,
            area=total_area,
            properties=merged_props,
            detection_method="+".join(methods)
        )


class EnhancedDetector:
    """Enhanced defect detector with ML and traditional methods"""
    
    def __init__(self):
        self.config = get_config()
        self.ml_detector = MLDefectDetector()
        self.traditional_detector = TraditionalDetector()
        self.merger = DefectMerger()
        
        # Thread pool for parallel detection
        if self.config.processing.parallel_processing:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.processing.max_workers
            )
        else:
            self.executor = None
    
    @log_performance
    def detect_defects(self, image: np.ndarray, zones: Dict[str, np.ndarray],
                      variation_results: Dict[str, np.ndarray]) -> List[Defect]:
        """Detect defects in all zones"""
        logger.info("Starting defect detection")
        
        all_defects = []
        
        # Process each zone
        for zone_name, zone_mask in zones.items():
            logger.info(f"Processing {zone_name} zone")
            
            # Skip empty zones
            if zone_mask is None or np.sum(zone_mask) == 0:
                logger.warning(f"Skipping empty {zone_name} zone")
                continue
            
            zone_defects = self._detect_in_zone(image, zone_mask, zone_name, variation_results)
            all_defects.extend(zone_defects)
        
        # Merge overlapping defects
        merged_defects = self.merger.merge_defects(all_defects)
        
        logger.info(f"Detection complete", 
                   total_defects=len(all_defects),
                   merged_defects=len(merged_defects))
        
        return merged_defects
    
    def _detect_in_zone(self, image: np.ndarray, zone_mask: np.ndarray, 
                       zone_name: str, variation_results: Dict[str, np.ndarray]) -> List[Defect]:
        """Detect defects in a single zone"""
        defects = []
        
        # ML detection
        if self.config.detection.use_ml_detection:
            ml_defects = self.ml_detector.detect_defects(image, zone_mask, zone_name)
            defects.extend(ml_defects)
        
        # Traditional detection methods
        detection_methods = [
            ('scratches', self.traditional_detector.detect_scratches),
            ('pits', self.traditional_detector.detect_pits),
            ('contamination', self.traditional_detector.detect_contamination),
            ('anomalies', self.traditional_detector.detect_statistical_anomalies)
        ]
        
        enabled_algorithms = self.config.detection.detection_algorithms
        
        if self.executor:
            # Parallel execution
            futures = []
            for name, method in detection_methods:
                if name in enabled_algorithms or f"traditional_{name}" in enabled_algorithms:
                    future = self.executor.submit(method, image, zone_mask, zone_name)
                    futures.append((name, future))
            
            # Collect results
            for name, future in futures:
                try:
                    result = future.result()
                    defects.extend(result)
                except Exception as e:
                    logger.error(f"Detection method {name} failed: {e}")
        else:
            # Sequential execution
            for name, method in detection_methods:
                if name in enabled_algorithms or f"traditional_{name}" in enabled_algorithms:
                    try:
                        result = method(image, zone_mask, zone_name)
                        defects.extend(result)
                    except Exception as e:
                        logger.error(f"Detection method {name} failed: {e}")
        
        # Filter by confidence
        filtered_defects = [
            d for d in defects 
            if d.confidence >= self.config.detection.confidence_threshold
        ]
        
        logger.info(f"Detected in {zone_name}", 
                   total=len(defects),
                   filtered=len(filtered_defects))
        
        return filtered_defects
    
    def visualize_defects(self, image: np.ndarray, defects: List[Defect]) -> np.ndarray:
        """Create visualization with defect overlays"""
        overlay = image.copy()
        
        # Draw each defect
        for defect in defects:
            color = self.config.visualization.defect_colors.get(
                defect.type, 
                self.config.visualization.defect_colors['unknown']
            )
            
            # Draw bounding box
            x, y, w, h = defect.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{defect.type} ({defect.confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for label
            cv2.rectangle(overlay, 
                         (x, y - label_size[1] - 5),
                         (x + label_size[0], y),
                         color, -1)
            
            # Text
            cv2.putText(overlay, label,
                       (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)
            
            # Severity indicator (circle)
            severity_radius = int(5 + defect.severity * 10)
            cv2.circle(overlay, defect.location, severity_radius, color, -1)
        
        return overlay
    
    def generate_heatmap(self, image_shape: Tuple[int, int], defects: List[Defect]) -> np.ndarray:
        """Generate defect density heatmap"""
        h, w = image_shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for defect in defects:
            # Create Gaussian around defect
            x, y = defect.location
            size = max(defect.bbox[2], defect.bbox[3])
            
            # Create meshgrid
            y_range = np.arange(max(0, y - size), min(h, y + size))
            x_range = np.arange(max(0, x - size), min(w, x + size))
            xx, yy = np.meshgrid(x_range, y_range)
            
            # Gaussian
            sigma = size / 3
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            gaussian *= defect.severity
            
            # Add to heatmap
            heatmap[y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1] += gaussian
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert to color
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap_color
    
    def __del__(self):
        """Cleanup"""
        if self.executor:
            self.executor.shutdown(wait=True)


def main():
    """Test the enhanced detector"""
    config = get_config()
    config.interactive_mode = False
    
    detector = EnhancedDetector()
    
    # Test with sample image
    test_image = config.input_dir / "test.jpg"
    if test_image.exists():
        image = cv2.imread(str(test_image))
        
        # Mock zones
        h, w = image.shape[:2]
        zones = {
            'core': np.zeros((h, w), dtype=np.uint8),
            'cladding': np.zeros((h, w), dtype=np.uint8),
            'ferrule': np.zeros((h, w), dtype=np.uint8)
        }
        
        # Simple circular zones
        center = (w // 2, h // 2)
        cv2.circle(zones['core'], center, int(min(center) * 0.2), 255, -1)
        cv2.circle(zones['cladding'], center, int(min(center) * 0.6), 255, -1)
        cv2.circle(zones['cladding'], center, int(min(center) * 0.2), 0, -1)
        cv2.circle(zones['ferrule'], center, int(min(center) * 0.9), 255, -1)
        cv2.circle(zones['ferrule'], center, int(min(center) * 0.6), 0, -1)
        
        # Detect defects
        defects = detector.detect_defects(image, zones, {"original": image})
        
        # Visualize
        vis = detector.visualize_defects(image, defects)
        heatmap = detector.generate_heatmap(image.shape[:2], defects)
        
        # Save results
        output_dir = config.output_dir
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / "detection_overlay.jpg"), vis)
        cv2.imwrite(str(output_dir / "detection_heatmap.jpg"), heatmap)
        
        logger.info(f"Saved visualizations to {output_dir}")
        
        # Print defect summary
        for defect in defects:
            logger.info(f"Defect: {defect.type} at {defect.location} "
                       f"(severity: {defect.severity:.2f}, confidence: {defect.confidence:.2f})")
    else:
        logger.warning("No test image found")


if __name__ == "__main__":
    main()